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
#include <vector>
#include <map>
#include "utilities/graph_utils.cuh"

namespace cugraph {
  template <typename idx_t>
  class db_pattern_entry {
    bool is_var;
    idx_t constantValue;
    std::string variableName;
  public:
    db_pattern_entry(std::string variable);
    db_pattern_entry(idx_t constant);
    db_pattern_entry(const db_pattern_entry<idx_t>& other);
    bool isVariable() const;
    idx_t getConstant() const;
    std::string getVariable() const;
  };

  template <typename idx_t>
  class db_pattern {
    std::vector<db_pattern_entry<idx_t>> entries;
  public:
    db_pattern();
    db_pattern(const db_pattern<idx_t>& other);
    int getSize() const;
    const db_pattern_entry<idx_t>& getEntry(int position) const;
    void addEntry(db_pattern_entry<idx_t>& entry);
    bool isAllConstants();
  };

  template <typename idx_t>
  class db_column_index {
    gdf_column* offsets;
    gdf_column* indirection;
    void deleteData();
  public:
    db_column_index();
    db_column_index(gdf_column* offsets, gdf_column* indirection);
    ~db_column_index();
    void resetData(gdf_column* offsets, gdf_column* indirection);
  };

  template <typename idx_t>
  class db_table {
    std::vector<gdf_column*> columns;
    std::vector<std::string> names;
    std::vector<db_pattern<idx_t>> inputBuffer;
    std::vector<db_column_index<idx_t>> indices;
  public:
    db_table();
    void addColumn(std::string name);
    void addEntry(db_pattern<idx_t>& pattern);
    void rebuildIndices();
    void flush_input();
  };

  template <typename idx_t>
  class db_object {
    // The dictionary and reverse dictionary encoding strings to ids and vice versa
    std::map<std::string, idx_t> valueToId;
    std::map<idx_t, std::string> idToValue;
    idx_t next_id;

    // The relationship table
    db_table<idx_t> relationshipsTable;

    // The relationship property table
    db_table<idx_t> relationshipPropertiesTable;

  public:
    db_object();
    std::string query(std::string query);
  };
}