# Copyright (c) 2019, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

cimport cugraph.traversal.bfs as c_bfs
from cugraph.structure.graph cimport *
from cugraph.structure import graph_wrapper
from cugraph.utilities.column_utils cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t

import cudf
import cudf._lib as libcudf
import numpy as np


def bfs(input_graph, start, directed=True):
    """
    Call bfs
    """
    cdef uintptr_t graph = graph_wrapper.allocate_cpp_graph()
    cdef Graph * g = <Graph*> graph

    if input_graph.adjlist:
        [offsets, indices] = graph_wrapper.datatype_cast([input_graph.adjlist.offsets, input_graph.adjlist.indices], [np.int32])
        [weights] = graph_wrapper.datatype_cast([input_graph.adjlist.weights], [np.float32, np.float64])
        graph_wrapper.add_adj_list(graph, offsets, indices, weights)
    else:
        [src, dst] = graph_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['src'], input_graph.edgelist.edgelist_df['dst']], [np.int32])
        if input_graph.edgelist.weights:
            [weights] = graph_wrapper.datatype_cast([input_graph.edgelist.edgelist_df['weights']], [np.float32, np.float64])
            graph_wrapper.add_edge_list(graph, src, dst, weights)
        else:
            graph_wrapper.add_edge_list(graph, src, dst)
        add_adj_list(g)
        offsets, indices, values = graph_wrapper.get_adj_list(graph)
        input_graph.adjlist = input_graph.AdjList(offsets, indices, values)

    # we should add get_number_of_vertices() to Graph (and this should be
    # used instead of g.adjList.offsets.size - 1)
    num_verts = g.adjList.offsets.size - 1

    if input_graph.renumbered is True:
        start = input_graph.edgelist.renumber_map[input_graph.edgelist.renumber_map==start].index[0]
    if not 0 <= start < num_verts:
        raise ValueError("Starting vertex should be between 0 to number of vertices")

    df = cudf.DataFrame()
    df['vertex'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['distance'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    df['predecessor'] = cudf.Series(np.zeros(num_verts, dtype=np.int32))
    cdef gdf_column c_vertex_col = get_gdf_column_view(df['vertex'])
    #cdef uintptr_t c_distance_ptr = get_column_data_ptr(df['distance']._column)
    cdef uintptr_t c_distance_ptr = df['distance'].__cuda_array_interface__['data'][0]
    #cdef uintptr_t c_predecessors_ptr = get_column_data_ptr(df['predecessor']._column)
    cdef uintptr_t c_predecessors_ptr = df['predecessor'].__cuda_array_interface__['data'][0]

    g.adjList.get_vertex_identifiers(&c_vertex_col)

    c_bfs.bfs[int](g, <int*>c_distance_ptr, <int*>c_predecessors_ptr, <int>start)

    if input_graph.renumbered:
        if isinstance(input_graph.edgelist.renumber_map, cudf.DataFrame):
            n_cols = len(input_graph.edgelist.renumber_map.columns) - 1
            unrenumered_df_ = df.merge(input_graph.edgelist.renumber_map, left_on='vertex', right_on='id', how='left').drop(['id', 'vertex'])
            unrenumered_df = unrenumered_df_.merge(input_graph.edgelist.renumber_map, left_on='predecessor', right_on='id', how='left').drop(['id', 'predecessor'])
            unrenumered_df.columns = ['distance']+['vertex_'+str(i) for i in range(n_cols)]+['predecessor_'+str(i) for i in range(n_cols)]
            cols = unrenumered_df.columns
            df = unrenumered_df[[cols[1:n_cols+1], cols[0], cols[n_cols:]]]
        else:
            df['vertex'] = input_graph.edgelist.renumber_map[df['vertex']]
            df['predecessor'][df['predecessor']>-1] = input_graph.edgelist.renumber_map[df['predecessor'][df['predecessor']>-1]]

    return df
