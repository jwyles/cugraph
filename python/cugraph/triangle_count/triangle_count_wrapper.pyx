# Copyright (c) 2018, NVIDIA CORPORATION.
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

from c_sssp cimport *
from c_graph cimport *
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libc.float cimport FLT_MAX_EXP
import cudf
from librmm_cffi import librmm as rmm
#from pygdf import Column
import numpy as np

cpdef triangle_count(G):
    """
    Compute the number of triangles in the given graph
    
    Parameters
    ----------
    graph : cuGraph.Graph                  
       cuGraph graph descriptor, should contain the connectivity information as an edge list (edge weights are not used for this algorithm). 

    Returns
    -------
    count : 
        64 bit integer value with the number of triangles found
    
    Examples
    --------
    >>> M = ReadMtxFile(graph_file)
    >>> sources = cudf.Series(M.row)
    >>> destinations = cudf.Series(M.col)
    >>> G = cuGraph.Graph()
    >>> G.add_edge_list(sources,destinations,None)
    >>> count = cuGraph.triangle_count(G)
    """

    cdef uintptr_t graph = G.graph_ptr
    cdef gdf_graph* g = <gdf_graph*>graph

    cdef unsigned long long result = 0;
    
    err = gdf_triangle_count_nvgraph(<gdf_graph*>graph, <unsigned long long*>& result)
    cudf.bindings.cudf_cpp.check_gdf_error(err)

    return result
