/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <graph.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <utilities/cuda_utils.cuh>
#include <utilities/graph_utils.cuh>
#include <community/louvain_kernels.hpp>
#include <community/Louvain_modularity.cuh>

#ifdef TIMING
#include <utilities/high_res_timer.hpp>
#endif

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t update_clustering_by_delta_modularity_constrained(
  weight_t m2,
  experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
  rmm::device_vector<vertex_t> const &src_indices,
  rmm::device_vector<weight_t> const &vertex_weights,
  rmm::device_vector<weight_t> &cluster_weights,
  rmm::device_vector<vertex_t> &cluster,
  rmm::device_vector<vertex_t> &constraint,
  cudaStream_t stream)
{
  rmm::device_vector<vertex_t> next_cluster(cluster);
  rmm::device_vector<weight_t> old_cluster_sum(graph.number_of_vertices);
  rmm::device_vector<weight_t> delta_Q(graph.number_of_edges);
  rmm::device_vector<vertex_t> cluster_hash(graph.number_of_edges);
  rmm::device_vector<weight_t> cluster_hash_sum(graph.number_of_edges, weight_t{0.0});

  vertex_t *d_cluster_hash         = cluster_hash.data().get();
  weight_t *d_cluster_hash_sum     = cluster_hash_sum.data().get();
  vertex_t *d_cluster              = cluster.data().get();
  vertex_t const *d_src_indices    = src_indices.data().get();
  vertex_t *d_dst_indices = graph.indices;
  weight_t const *d_vertex_weights = vertex_weights.data().get();
  weight_t *d_cluster_weights      = cluster_weights.data().get();
  weight_t *d_delta_Q              = delta_Q.data().get();
  weight_t *d_old_cluster_sum      = old_cluster_sum.data().get();
  vertex_t *d_constraint = constraint.data().get();
  vertex_t *d_next_cluster  = next_cluster.data().get();

  weight_t new_Q = modularity<vertex_t, edge_t, weight_t>(m2, graph, cluster.data().get(), stream);

  weight_t cur_Q = new_Q - 1;

  // To avoid the potential of having two vertices swap clusters
  // we will only allow vertices to move up (true) or down (false)
  // during each iteration of the loop
  bool up_down = true;

//  while (new_Q > (cur_Q + 0.0001)) {
  while (true) {
    cur_Q = new_Q;

    compute_delta_modularity(graph,
                             cluster_hash,
                             cluster_hash_sum,
                             old_cluster_sum,
                             d_src_indices,
                             d_cluster,
                             d_cluster_weights,
                             m2,
                             d_vertex_weights,
                             d_delta_Q,
                             stream);

//    compute_delta_modularity2(graph,
//                            d_cluster,
//                            d_vertex_weights,
//                            d_delta_Q,
//                            m2,
//                            stream);

    // Before making the swaps we will filter out any positive delta scores that cross boundaries
    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(graph.number_of_edges),
                     [d_delta_Q,
                      d_constraint,
                      d_src_indices,
                      d_dst_indices]
                      __device__ (edge_t loc) {
                        vertex_t start_cluster = d_constraint[d_src_indices[loc]];
                        vertex_t end_cluster = d_constraint[d_dst_indices[loc]];
                        if (start_cluster != end_cluster)
                          d_delta_Q[loc] = 0;});

    // Try to filter out self-swaps
    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(graph.number_of_edges),
                     [d_delta_Q, d_cluster, d_src_indices, d_dst_indices]
                      __device__ (vertex_t i) {
                        vertex_t start = d_src_indices[i];
                        vertex_t end = d_dst_indices[i];
                        vertex_t current_cluster = d_cluster[start];
                        vertex_t new_cluster = d_cluster[end];
                        if (current_cluster == new_cluster && d_delta_Q[i] > 0) {
                          printf("Vertex %d moving from %d to %d with score %f\n", start, current_cluster, new_cluster, d_delta_Q[i]);
                          d_delta_Q[i] = 0;
                        }});

    // Count how many potential swaps there are remaining
    rmm::device_vector<vertex_t> swap_counts(graph.number_of_edges, 0);
    vertex_t* d_swap_counts = swap_counts.data().get();
    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(graph.number_of_edges),
                     [d_swap_counts, d_delta_Q]
                      __device__ (edge_t loc) {
                        if (d_delta_Q[loc] > 0)
                          d_swap_counts[loc] = 1; });
    vertex_t count = thrust::reduce(rmm::exec_policy(stream)->on(stream),
                                    swap_counts.begin(),
                                    swap_counts.end());
    std::cout << "Found " << count << " potential swaps.\n";
    if (count == 0)
      break;

    weight_t delta_score = assign_single_node(graph,
                                              d_delta_Q,
                                              d_cluster_hash,
                                              d_src_indices,
                                              d_next_cluster,
                                              d_vertex_weights,
                                              d_cluster_weights,
                                              up_down,
                                              stream);

    up_down = !up_down;

    new_Q = modularity<vertex_t, edge_t, weight_t>(m2, graph, next_cluster.data().get(), stream);
    auto new_Q2 = modularity2<vertex_t, edge_t, weight_t>(m2, graph, next_cluster.data().get(), stream);
    auto Q_diff = abs(new_Q - new_Q2);
    if (Q_diff > .000001)
      std::cout << "Difference in modularity scores of " << Q_diff << " new score " << new_Q2 << " old score " << new_Q << "\n";


    weight_t actual_delta = new_Q - cur_Q;

    std::cout << "New modularity: " << new_Q << " a difference of: " << new_Q - cur_Q << "\n";
    std::cout << "actual_delta / delta_score = " << actual_delta / delta_score << "\n";

//    if (new_Q > cur_Q) { thrust::copy(next_cluster.begin(), next_cluster.end(), cluster.begin()); }
    thrust::copy(next_cluster.begin(), next_cluster.end(), cluster.begin());
  }

  return cur_Q;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void leiden(experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
             weight_t *final_modularity,
             int *num_level,
             vertex_t *cluster_vec,
             int max_iter,
             cudaStream_t stream)
{
#ifdef TIMING
  HighResTimer hr_timer;
#endif

  *num_level = 0;

  //
  //  Vectors to create a copy of the graph
  //
  rmm::device_vector<edge_t> offsets_v(graph.offsets, graph.offsets + graph.number_of_vertices + 1);
  rmm::device_vector<vertex_t> indices_v(graph.indices, graph.indices + graph.number_of_edges);
  rmm::device_vector<weight_t> weights_v(graph.edge_data, graph.edge_data + graph.number_of_edges);
  rmm::device_vector<vertex_t> src_indices_v(graph.number_of_edges);

  //
  //  Weights and clustering across iterations of algorithm
  //
  rmm::device_vector<weight_t> vertex_weights_v(graph.number_of_vertices);
  rmm::device_vector<weight_t> cluster_weights_v(graph.number_of_vertices);
  rmm::device_vector<vertex_t> cluster_v(graph.number_of_vertices);

  //
  //  Temporaries used within kernels.  Each iteration uses less
  //  of this memory
  //
  rmm::device_vector<vertex_t> tmp_arr_v(graph.number_of_vertices);
  rmm::device_vector<vertex_t> cluster_inverse_v(graph.number_of_vertices);

  weight_t m2 =
    thrust::reduce(rmm::exec_policy(stream)->on(stream), weights_v.begin(), weights_v.end());
  weight_t best_modularity = -1;

  //
  //  Initialize every cluster to reference each vertex to itself
  //
  thrust::sequence(rmm::exec_policy(stream)->on(stream), cluster_v.begin(), cluster_v.end());
  thrust::copy(cluster_v.begin(), cluster_v.end(), cluster_vec);

  //
  //  Our copy of the graph.  Each iteration of the outer loop will
  //  shrink this copy of the graph.
  //
  cugraph::experimental::GraphCSRView<vertex_t, edge_t, weight_t> current_graph(
    offsets_v.data().get(),
    indices_v.data().get(),
    weights_v.data().get(),
    graph.number_of_vertices,
    graph.number_of_edges);

  current_graph.get_source_indices(src_indices_v.data().get());

//  while (true) {
  for (int i = 0; i < max_iter; i++) {
    //
    //  Sum the weights of all edges departing a vertex.  This is
    //  loop invariant, so we'll compute it here.
    //
    //  Cluster weights are equivalent to vertex weights with this initial
    //  graph
    //
#ifdef TIMING
    hr_timer.start("init");
#endif

    cugraph::detail::compute_vertex_sums(current_graph, vertex_weights_v, stream);
    thrust::copy(vertex_weights_v.begin(), vertex_weights_v.end(), cluster_weights_v.begin());

#ifdef TIMING
    hr_timer.stop();

    hr_timer.start("update_clustering");
#endif

    weight_t new_Q = update_clustering_by_delta_modularity(
      m2, current_graph, src_indices_v, vertex_weights_v, cluster_weights_v, cluster_v, stream);

    // Figure out whether or not anything changed
    rmm::device_vector<vertex_t> changes(current_graph.number_of_vertices, 0);
    vertex_t* d_changes = changes.data().get();
    vertex_t* d_cluster = cluster_v.data().get();
    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(current_graph.number_of_vertices),
                     [d_changes, d_cluster]
                      __device__ (edge_t loc) {
                        if (loc != d_cluster[loc])
                          d_changes[loc] = 1;});
    auto change_count = thrust::reduce(rmm::exec_policy(stream)->on(stream),
                                       changes.begin(),
                                       changes.end());
    std::cout << "Change count: " << change_count << "\n";
    if (change_count == 0) {
      break;
    }



    std::cout << i << "New modularity first run: " << new_Q << "\n";

    // After finding the initial clustering we will now "refine" the partition
    rmm::device_vector<vertex_t> constraint(graph.number_of_vertices);
    thrust::copy(cluster_v.begin(), cluster_v.end(), constraint.begin());
    thrust::sequence(rmm::exec_policy(stream)->on(stream), cluster_v.begin(), cluster_v.end());

    new_Q = update_clustering_by_delta_modularity_constrained(m2,
                                                              current_graph,
                                                              src_indices_v,
                                                              vertex_weights_v,
                                                              cluster_weights_v,
                                                              cluster_v,
                                                              constraint,
                                                              stream);


    std::cout << i << "New modularity second run: " << new_Q << "\n\n";

#ifdef TIMING
    hr_timer.stop();
#endif

//    if (new_Q <= best_modularity) {
//      *num_level = i;
//      break;
//    }

    best_modularity = new_Q;

#ifdef TIMING
    hr_timer.start("shrinking graph");
#endif

    // renumber the clusters to the range 0..(num_clusters-1)
    vertex_t num_clusters = renumber_clusters(
      graph.number_of_vertices, cluster_v, tmp_arr_v, cluster_inverse_v, cluster_vec, stream);
    cluster_weights_v.resize(num_clusters);

    std::cout << "There are " << num_clusters << " clusters.\n";

    vertex_t* d_constraint = constraint.data().get();
    rmm::device_vector<vertex_t> constraint_group(current_graph.number_of_vertices);
    vertex_t* d_constraint_group = constraint_group.data().get();

    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(current_graph.number_of_vertices),
                     [d_cluster, d_constraint, d_constraint_group]
                      __device__ (vertex_t i) {
                        vertex_t cluster = d_cluster[i];
                        vertex_t group = d_constraint[i];
                        d_constraint_group[cluster] = group;});

    // shrink our graph to represent the graph of supervertices
    generate_superverticies_graph(current_graph, src_indices_v, num_clusters, cluster_v, stream);

    // assign each new vertex to its own cluster
//    thrust::sequence(rmm::exec_policy(stream)->on(stream), cluster_v.begin(), cluster_v.end());

    // Assign each new vertex to the constrained cluster it belongs to:
    thrust::for_each(rmm::exec_policy(stream)->on(stream),
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(current_graph.number_of_vertices),
                         [d_constraint_group, d_cluster]
                          __device__ (vertex_t i) {
                            d_cluster[i] = d_constraint_group[i];});


    // For debugging purposes
    weight_t after_Q = modularity(m2, current_graph, cluster_v.data().get(), stream);

    std::cout << "Modularity after graph contraction: " << after_Q << "\n";

#ifdef TIMING
    hr_timer.stop();
#endif
    *num_level = i;
  }

#ifdef TIMING
  hr_timer.display(std::cout);
#endif

  *final_modularity = best_modularity;
}

template void leiden(experimental::GraphCSRView<int32_t, int32_t, float> const &,
                      float *,
                      int *,
                      int32_t *,
                      int,
                      cudaStream_t);
template void leiden(experimental::GraphCSRView<int32_t, int32_t, double> const &,
                      double *,
                      int *,
                      int32_t *,
                      int,
                      cudaStream_t);

}  // namespace detail
}  // namespace cugraph
