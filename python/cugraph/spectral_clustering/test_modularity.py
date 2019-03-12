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

import random

import pytest
from scipy.io import mmread

import cudf
import cugraph


def read_mtx_file(mm_file):
    print('Reading ' + str(mm_file) + '...')
    return mmread(mm_file).asfptype()


def cugraph_call(G, partitions):
    df = cugraph.spectralModularityMaximizationClustering(
        G, partitions, num_eigen_vects=(partitions - 1))
    score = cugraph.analyzeClustering_modularity(G, partitions, df['cluster'])
    return score


def random_call(G, partitions):
    num_verts = G.num_vertices()
    assignment = []
    for i in range(num_verts):
        assignment.append(random.randint(0, partitions-1))
    assignment_cu = cudf.Series(assignment)
    score = cugraph.analyzeClustering_modularity(G, partitions, assignment_cu)
    return score


DATASETS = [
    '/datasets/networks/karate.mtx',
    '/datasets/networks/dolphins.mtx',
    '/datasets/golden_data/graphs/dblp.mtx']
PARTITIONS = [2, 4, 8]


@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('partitions', PARTITIONS)
def test_modularity_clustering_adjacency(graph_file, partitions):
    # Read in the graph and get a cugraph object
    M = read_mtx_file(graph_file).tocsr()
    row_offsets = cudf.Series(M.indptr)
    col_indices = cudf.Series(M.indices)
    values = cudf.Series(M.data)
    G = cugraph.Graph()
    G.add_adj_list(row_offsets, col_indices, values)

    # Get the modularity score for partitioning versus random assignment
    cu_score = cugraph_call(G, partitions)
    rand_score = random_call(G, partitions)

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > rand_score


@pytest.mark.parametrize('graph_file', DATASETS)
@pytest.mark.parametrize('partitions', PARTITIONS)
def test_modularity_clustering_edge(graph_file, partitions):
    # Read in the graph and get a cugraph object
    M = read_mtx_file(graph_file).tocoo()
    row = cudf.Series(M.row)
    col = cudf.Series(M.col)
    values = cudf.Series(M.data)
    G = cugraph.Graph()
    G.add_edge_list(row, col, values)

    # Get the modularity score for partitioning versus random assignment
    cu_score = cugraph_call(G, partitions)
    rand_score = random_call(G, partitions)

    # Assert that the partitioning has better modularity than the random
    # assignment
    assert cu_score > rand_score
