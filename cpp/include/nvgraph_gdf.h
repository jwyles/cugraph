/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
/** ---------------------------------------------------------------------------*
 * @brief Wrapper functions for Nvgraph
 *
 * @file nvgraph_gdf.h
 * ---------------------------------------------------------------------------**/

#pragma once

#include <nvgraph/nvgraph.h>
#include <cugraph.h>

/**
 * Wrapper function to create an nvgraph graph object from a gdf_graph object
 * @param nvg_handle Nvgraph handle
 * @param gdf_G Gdf graph object pointer
 * @param nvgraph_G Nvgraph graph descriptor handle
 * @param use_transposed Flag determining whether to use the transpose of the input graph
 * @return Error code
 */
gdf_error gdf_createGraph_nvgraph(nvgraphHandle_t nvg_handle, gdf_graph* gdf_G, nvgraphGraphDescr_t * nvgraph_G, bool use_transposed = false);

/**
 * Wrapper function around Nvgraph SSSP algorithm.
 * @param gdf_G GDF graph object pointer
 * @param source_vert Pointer to an integer value giving the desired source vertex to use
 * @param sssp_distances GDF column object pointer where the result will be written
 * @return Error code
 */
gdf_error gdf_sssp_nvgraph(gdf_graph *gdf_G, const int *source_vert, gdf_column *sssp_distances);

/**
 * Wrapper function around Nvgraph triangle counting algorithm
 * @param gdf_G GDF graph object pointer
 * @param result Pointer to uint64 value in which the result will be stored
 * @return Error code
 */
gdf_error gdf_triangle_count_nvgraph(gdf_graph*gdf_G, unsigned long long* result);
