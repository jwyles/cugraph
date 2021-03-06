# Copyright (c) 2020, NVIDIA CORPORATION.
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

import cugraph.dask as dcg
import cugraph.comms as Comms
from dask.distributed import Client
import gc
import cugraph
import dask_cudf
import cudf
from dask_cuda import LocalCUDACluster


def test_dask_bfs():
    gc.collect()
    cluster = LocalCUDACluster()
    client = Client(cluster)
    Comms.initialize()

    input_data_path = r"../datasets/netscience.csv"
    chunksize = dcg.get_chunksize(input_data_path)

    ddf = dask_cudf.read_csv(input_data_path, chunksize=chunksize,
                             delimiter=' ',
                             names=['src', 'dst', 'value'],
                             dtype=['int32', 'int32', 'float32'])

    df = cudf.read_csv(input_data_path,
                       delimiter=' ',
                       names=['src', 'dst', 'value'],
                       dtype=['int32', 'int32', 'float32'])

    g = cugraph.DiGraph()
    g.from_cudf_edgelist(df, 'src', 'dst', renumber=True)

    dg = cugraph.DiGraph()
    dg.from_dask_cudf_edgelist(ddf, renumber=True)

    expected_dist = cugraph.bfs(g, 0)
    result_dist = dcg.bfs(dg, 0, True)

    compare_dist = expected_dist.merge(
        result_dist, on="vertex", suffixes=['_local', '_dask']
    )

    err = 0

    for i in range(len(compare_dist)):
        if (compare_dist['distance_local'].iloc[i] !=
                compare_dist['distance_dask'].iloc[i]):
            err = err + 1
    assert err == 0

    Comms.destroy()
    client.close()
    cluster.close()
