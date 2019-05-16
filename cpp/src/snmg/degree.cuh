/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <omp.h>
#include "graph_utils.cuh"
#include "snmg_utils.cuh"

namespace cugraph {
  /**
   * Single node multi-GPU method for degree calculation on a partitioned graph.
   * @param x Indicates whether to compute in degree, out degree, or the sum of both.
   *    0 = in + out degree
   *    1 = in-degree
   *    2 = out-degree
   * @param part_off The vertex partitioning of the global graph
   * @param off The offsets array of the local partition
   * @param ind The indices array of the local partition
   * @param degree Pointer to pointers to memory on each GPU for the result
   * @return Error code
   */
  template<typename idx_t>
  gdf_error snmg_degree(int x, size_t* part_off, idx_t* off, idx_t* ind, idx_t** degree) {
    sync_all();
    auto i = omp_get_thread_num();
    auto p = omp_get_num_threads();

    // Getting the global and local vertices and edges
    size_t glob_v = part_off[p];
    size_t loc_v = part_off[i + 1] - part_off[i];
    idx_t tmp;
    CUDA_TRY(cudaMemcpy(&tmp, &off[loc_v], sizeof(idx_t), cudaMemcpyDeviceToHost));
    size_t loc_e = tmp;

    // Allocating the local result array, and setting all entries to zero.
    idx_t* local_result;
    ALLOC_MANAGED_TRY((void** )&local_result, glob_v * sizeof(idx_t), nullptr);
    rmm_temp_allocator allocator(nullptr);
    thrust::fill(thrust::cuda::par(allocator).on(nullptr), local_result, local_result + glob_v, 0);

    // In-degree
    if (x == 1 || x == 0) {
      dim3 nthreads, nblocks;
      nthreads.x = min((idx_t) loc_e, (idx_t) CUDA_MAX_KERNEL_THREADS);
      nthreads.y = 1;
      nthreads.z = 1;
      nblocks.x = min((idx_t) ((loc_e + nthreads.x - 1) / nthreads.x), (idx_t) CUDA_MAX_BLOCKS);
      nblocks.y = 1;
      nblocks.z = 1;
      degree_coo<idx_t, idx_t> <<<nblocks, nthreads>>>((idx_t) loc_e,
                                                       (idx_t) loc_e,
                                                       ind,
                                                       local_result);
    }

    // Out-degree
    if (x == 2 || x == 0) {
      dim3 nthreads, nblocks;
      nthreads.x = min((idx_t) loc_v, (idx_t) CUDA_MAX_KERNEL_THREADS);
      nthreads.y = 1;
      nthreads.z = 1;
      nblocks.x = min((idx_t) ((loc_v + nthreads.x - 1) / nthreads.x), (idx_t) CUDA_MAX_BLOCKS);
      nblocks.y = 1;
      nblocks.z = 1;
      degree_offsets<idx_t, idx_t> <<<nblocks, nthreads>>>((idx_t) loc_v,
                                                           (idx_t) loc_e,
                                                           off,
                                                           local_result + part_off[i]);
    }

    // Combining the local results into global results
    treeReduce<idx_t, thrust::plus<idx_t> >(glob_v, local_result, degree);

    // Broadcasting the global result to all GPUs
    treeBroadcast(glob_v, local_result, degree);

    return GDF_SUCCESS;
  }

  template<>
  gdf_error snmg_degree<int64_t>(int x,
                                 size_t* part_off,
                                 int64_t* off,
                                 int64_t* ind,
                                 int64_t** degree) {
    sync_all();
    auto i = omp_get_thread_num();
    auto p = omp_get_num_threads();

    // Getting the global and local vertices and edges
    size_t glob_v = part_off[p];
    size_t loc_v = part_off[i + 1] - part_off[i];
    int64_t tmp;
    CUDA_TRY(cudaMemcpy(&tmp, &off[loc_v], sizeof(int64_t), cudaMemcpyDeviceToHost));
    size_t loc_e = tmp;

    // Allocating the local result array, and setting all entries to zero.
    int64_t* local_result;
    ALLOC_MANAGED_TRY((void** )&local_result, glob_v * sizeof(int64_t), nullptr);
    rmm_temp_allocator allocator(nullptr);
    thrust::fill(thrust::cuda::par(allocator).on(nullptr), local_result, local_result + glob_v, 0);

    // In-degree
    if (x == 1 || x == 0) {
      dim3 nthreads, nblocks;
      nthreads.x = min((int64_t) loc_e, (int64_t) CUDA_MAX_KERNEL_THREADS);
      nthreads.y = 1;
      nthreads.z = 1;
      nblocks.x = min((int64_t)((loc_e + nthreads.x - 1) / nthreads.x), (int64_t) CUDA_MAX_BLOCKS);
      nblocks.y = 1;
      nblocks.z = 1;
      degree_coo<int64_t, double> <<<nblocks, nthreads>>>((int64_t) loc_e,
                                                        (int64_t) loc_e,
                                                        ind,
                                                        (double*) local_result);
    }

    // Out-degree
    if (x == 2 || x == 0) {
      dim3 nthreads, nblocks;
      nthreads.x = min((int64_t) loc_v, (int64_t) CUDA_MAX_KERNEL_THREADS);
      nthreads.y = 1;
      nthreads.z = 1;
      nblocks.x = min((int64_t)((loc_v + nthreads.x - 1) / nthreads.x), (int64_t) CUDA_MAX_BLOCKS);
      nblocks.y = 1;
      nblocks.z = 1;
      degree_offsets<int64_t, double> <<<nblocks, nthreads>>>((int64_t) loc_v,
                                                            (int64_t) loc_e,
                                                            off,
                                                            (double*) (local_result + part_off[i]));
    }

    // Convert the values written as doubles back to int64:
    dim3 nthreads, nblocks;
    nthreads.x = min((int64_t) glob_v, (int64_t) CUDA_MAX_KERNEL_THREADS);
    nthreads.y = 1;
    nthreads.z = 1;
    nblocks.x = min((int64_t)((glob_v + nthreads.x - 1) / nthreads.x), (int64_t) CUDA_MAX_BLOCKS);
    nblocks.y = 1;
    nblocks.z = 1;
    type_convert<double, int64_t> <<<nblocks, nthreads>>>((double*) local_result, glob_v);

    // Combining the local results into global results
    treeReduce<int64_t, thrust::plus<int64_t> >(glob_v, local_result, degree);

    // Broadcasting the global result to all GPUs
    treeBroadcast(glob_v, local_result, degree);

    return GDF_SUCCESS;
  }
}