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

#include "gtest/gtest.h"
#include "high_res_clock.h"
#include <cugraph.h>
#include "test_utils.h"
#include "db/db_operators.cuh"
#include "utilities/graph_utils.cuh"
#include "db/db_parser_integration_test.cuh"

class Test_Parser: public ::testing::Test {

};

TEST_F(Test_Parser, printOut) {
  std::string input = "LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS csvLine";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";

  input = "MATCH (p:Person {name: 'James'})-[:HasPet]->(z:Pet)\nRETURN z.name";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";

  input = "CREATE (p:Person {name: 'James'})";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";

  input = "LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS csvLine\nCREATE (n:Person {name: csvLine.name})";
  std::cout << input << "\n";
  cugraph::db::printOutAst(input);
  std::cout << "\n";
}
