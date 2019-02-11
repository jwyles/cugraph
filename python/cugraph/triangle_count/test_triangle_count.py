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

import cugraph
import cudf
import time
from scipy.io import mmread
import networkx as nx
import numpy as np
import pytest

print('Networkx version : {} '.format(nx.__version__))


def ReadMtxFile(mmFile):
    print('Reading ' + str(mmFile) + '...')
    return mmread(mmFile).asfptype()


def cugraph_Call(M):

    # Device data
    sources = cudf.Series(M.row)
    destinations = cudf.Series(M.col)
    data = cudf.Series(M.data)

    # cugraph Pagerank Call
    G = cugraph.Graph()
    G.add_edge_list(sources, destinations, data)

    print('cugraph Solving... ')
    t1 = time.time()

    triangles = cugraph.triangle_count(G)

    t2 = time.time() - t1
    print('Time : '+str(t2))

    return triangles


def networkx_Call(M):

    print('Format conversion ... ')
    M = M.tocsr()
    if M is None:
        raise TypeError('Could not read the input graph')
    if M.shape[0] != M.shape[1]:
        raise TypeError('Shape is not square')

    # Directed NetworkX graph
    Gnx = nx.Graph(M)

    print('NX Solving... ')
    t1 = time.time()

    trianglesDic = nx.triangles(Gnx)
    triangles = 0
    for i in range(len(trianglesDic)):
        triangles = triangles + trianglesDic[i]
 
    t2 = time.time() - t1

    print('Time : ' + str(t2))

    return triangles


datasets = ['/datasets/networks/dolphins.mtx',
            '/datasets/networks/karate.mtx',
            '/datasets/golden_data/graphs/dblp.mtx']


@pytest.mark.parametrize('graph_file', datasets)
def test_sssp(graph_file):

    M = ReadMtxFile(graph_file)
    cu_triangles = cugraph_Call(M)
    nx_triangles = networkx_Call(M)

    assert cu_triangles == nx_triangles