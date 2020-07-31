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
#pragma once

#include <graph.hpp>

namespace cugraph {
namespace detail {

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t modularity(weight_t m2,
                    experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                    vertex_t const *d_cluster,
                    cudaStream_t stream);

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t modularity2(weight_t m2,
                     experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
                     vertex_t const *d_cluster,
                     cudaStream_t stream);

template <typename vertex_t, typename edge_t, typename weight_t>
void generate_superverticies_graph(
    cugraph::experimental::GraphCSRView<vertex_t, edge_t, weight_t> &current_graph,
    rmm::device_vector<vertex_t> &src_indices_v,
    vertex_t new_number_of_vertices,
    rmm::device_vector<vertex_t> &cluster_v,
    cudaStream_t stream);

template <typename vertex_t, typename edge_t, typename weight_t>
void compute_vertex_sums(experimental::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                         rmm::device_vector<weight_t>& sums,
                         cudaStream_t stream);

template<typename vertex_t>
vertex_t renumber_clusters(vertex_t graph_num_vertices,
                           rmm::device_vector<vertex_t> &cluster,
                           rmm::device_vector<vertex_t> &temp_array,
                           rmm::device_vector<vertex_t> &cluster_inverse,
                           vertex_t *cluster_vec,
                           cudaStream_t stream);

template<typename vertex_t, typename edge_t, typename weight_t>
void assign_nodes(experimental::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                  weight_t* d_delta_Q,
                  vertex_t* d_cluster_hash,
                  vertex_t const* d_src_indices,
                  vertex_t* d_next_cluster,
                  weight_t const* d_vertex_weights,
                  weight_t* d_cluster_weights,
                  bool up_down,
                  cudaStream_t stream);

template<typename vertex_t, typename edge_t, typename weight_t>
weight_t assign_single_node(experimental::GraphCSRView<vertex_t, edge_t, weight_t> const& graph,
                            weight_t* d_delta_Q,
                            vertex_t* d_cluster_hash,
                            vertex_t const* d_src_indices,
                            vertex_t* d_next_cluster,
                            weight_t const* d_vertex_weights,
                            weight_t* d_cluster_weights,
                            bool up_down,
                            cudaStream_t stream);

template <typename vertex_t, typename edge_t, typename weight_t>
void compute_delta_modularity(
    experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
    rmm::device_vector<vertex_t>& cluster_hash,
    rmm::device_vector<weight_t>& cluster_hash_sum,
    rmm::device_vector<weight_t>& old_cluster_sum,
    vertex_t const *d_src_indices,
    vertex_t *d_cluster,
    weight_t *d_cluster_weights,
    weight_t m2,
    weight_t const *d_vertex_weights,
    weight_t *d_delta_Q,
    cudaStream_t stream);

template <typename vertex_t, typename edge_t, typename weight_t>
weight_t update_clustering_by_delta_modularity(
    weight_t m2,
    experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
    rmm::device_vector<vertex_t> const &src_indices,
    rmm::device_vector<weight_t> const &vertex_weights,
    rmm::device_vector<weight_t> &cluster_weights,
    rmm::device_vector<vertex_t> &cluster,
    cudaStream_t stream);

template <typename vertex_t, typename edge_t, typename weight_t>
void louvain(experimental::GraphCSRView<vertex_t, edge_t, weight_t> const &graph,
             weight_t *final_modularity,
             int *num_level,
             vertex_t *cluster_vec,
             int max_iter,
             cudaStream_t stream = 0);

}  // namespace detail
}  // namespace cugraph
