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

//#include <nvgraph/include/util.cuh>
#include <utilities/cuda_utils.cuh>
#include <utilities/graph_utils.cuh>
#include <community/louvain_kernels.hpp>

#ifdef TIMING
#include <utilities/high_res_timer.hpp>
#endif

namespace cugraph {
namespace detail {

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

  while (true) {
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

#ifdef TIMING
    hr_timer.stop();
#endif

    if (new_Q <= best_modularity) { break; }

    best_modularity = new_Q;

#ifdef TIMING
    hr_timer.start("shrinking graph");
#endif

    // renumber the clusters to the range 0..(num_clusters-1)
    vertex_t num_clusters = renumber_clusters(
      graph.number_of_vertices, cluster_v, tmp_arr_v, cluster_inverse_v, cluster_vec, stream);
    cluster_weights_v.resize(num_clusters);

    // shrink our graph to represent the graph of supervertices
    generate_superverticies_graph(current_graph, src_indices_v, num_clusters, cluster_v, stream);

    // assign each new vertex to its own cluster
    thrust::sequence(rmm::exec_policy(stream)->on(stream), cluster_v.begin(), cluster_v.end());

#ifdef TIMING
    hr_timer.stop();
#endif
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
