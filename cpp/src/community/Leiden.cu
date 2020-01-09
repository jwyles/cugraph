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

#include <cugraph.h>
#include "utilities/error_utils.h"
#include <rmm_utils.h>
#include "utilities/graph_utils.cuh"

namespace cugraph {
template<typename IdxT, typename ValT>
void leiden(Graph* graph,
            int metric,
            double gamma,
            IdxT* leiden_parts,
            int max_iter = 100){
  // Assign initial singleton partition
  // Compute metric
  // Compute delta metric
  // Reassign nodes with positive delta metric
  // Repeat until no swaps are made
  // Refine the partition
  // Aggregate the graph according to the refined partition
  // Set initial partitioning according to unrefined partition
  //

}
} // cugraph namespace
