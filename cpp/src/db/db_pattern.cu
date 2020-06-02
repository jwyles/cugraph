/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <utilities/error_utils.h>
#include <db/db_pattern.cuh>
#include <db/parser_helpers.cuh>
#include <sstream>

namespace cugraph {
namespace db {

value::value()
{
  valType     = value_type::Uninitialized;
  array_index = 0;
}

value::value(value_type type, std::string id)
{
  CUGRAPH_EXPECTS(type == value_type::String || type == value_type::Identifier,
                  "Invalid value_type given");
  valType     = type;
  identifier  = id;
  array_index = 0;
}

value::value(std::string id, std::string prop)
{
  valType         = value_type::Property_Access;
  identifier      = id;
  prop_identifier = prop;
  array_index     = 0;
}

value::value(std::string id, uint32_t idx)
{
  valType     = value_type::Array_Access;
  identifier  = id;
  array_index = idx;
}

value_type value::type() { return valType; }

std::string value::getIdentifier()
{
  CUGRAPH_EXPECTS(valType != value_type::Uninitialized, "Can't access an unitialized value");
  return identifier;
}

std::string value::getPropertyName()
{
  CUGRAPH_EXPECTS(valType != value_type::Uninitialized, "Can't access an unitialized value");
  CUGRAPH_EXPECTS(valType == value_type::Property_Access,
                  "Value type inconsistent with attempted access");
  return prop_identifier;
}

uint32_t value::getArrayIndex()
{
  CUGRAPH_EXPECTS(valType != value_type::Uninitialized, "Can't access an unitialized value");
  CUGRAPH_EXPECTS(valType == value_type::Array_Access,
                  "Value type inconsistent with attempted access");
  return array_index;
}

template <typename idx_t>
node_pattern<idx_t>::node_pattern(std::string id)
{
  identifier = id;
}

template <typename idx_t>
node_pattern<idx_t>::node_pattern(const cypher_astnode_t* astNode, context<idx_t>& ctx)
{
  // Check that the given astnode is the right type:
  std::string typeStr = getTypeString(astNode);
  CUGRAPH_EXPECTS(typeStr == "node pattern", "Wrong type of astnode supplied");

  // Get the identifier for the node
  const cypher_astnode_t* id = cypher_ast_node_pattern_get_identifier(astNode);
  if (id == nullptr) {
    identifier = ctx.getUniqueId();
  } else {
    identifier = cypher_ast_identifier_get_name(id);
  }

  // Get the labels
  uint32_t num_labels = cypher_ast_node_pattern_nlabels(astNode);
  for (uint32_t i = 0; i < num_labels; i++) {
    const cypher_astnode_t* label = cypher_ast_node_pattern_get_label(astNode, i);
    std::string labelName         = cypher_ast_label_get_name(label);
    labels.push_back(labelName);
  }

  // Extract the properties
  const cypher_astnode_t* mapNode = cypher_ast_node_pattern_get_properties(astNode);
  if (mapNode != nullptr) {
    uint32_t num_pairs = cypher_ast_map_nentries(mapNode);
    for (uint32_t i = 0; i < num_pairs; i++) {
      // Getting the key value for the property map entry
      const cypher_astnode_t* propName = cypher_ast_map_get_key(mapNode, i);
      std::string propNameValue        = cypher_ast_prop_name_get_value(propName);

      // Getting the value for the property map entry
      const cypher_astnode_t* propValue = cypher_ast_map_get_value(mapNode, i);
      std::string valueType             = getTypeString(propValue);
      if (valueType == "string") {
        std::string valueStr = cypher_ast_string_get_value(propValue);
        value val(value_type::String, valueStr);
        properties[propNameValue] = val;
      }
      if (valueType == "property") {
        const cypher_astnode_t* expression = cypher_ast_property_operator_get_expression(propValue);
        std::string name                   = cypher_ast_identifier_get_name(expression);

        const cypher_astnode_t* propName2 = cypher_ast_property_operator_get_prop_name(propValue);
        std::string v                     = cypher_ast_prop_name_get_value(propName2);
        value val(name, v);
        properties[propNameValue] = val;
      }
      if (valueType == "subscript") {
        const cypher_astnode_t* expression =
          cypher_ast_subscript_operator_get_expression(propValue);
        std::string exp                   = cypher_ast_identifier_get_name(expression);
        const cypher_astnode_t* subscript = cypher_ast_subscript_operator_get_subscript(propValue);
        std::string subscriptStr          = cypher_ast_integer_get_valuestr(subscript);
        std::stringstream ss(subscriptStr);
        uint32_t index;
        ss >> index;
        value val(exp, index);
        properties[propNameValue] = val;
      }
    }
  }
}

template <typename idx_t>
void node_pattern<idx_t>::setIdentifier(std::string id)
{
  identifier = id;
}

template <typename idx_t>
void node_pattern<idx_t>::addLabel(std::string label)
{
  labels.push_back(label);
}

template <typename idx_t>
void node_pattern<idx_t>::addProperty(std::string name, value val)
{
  properties[name] = val;
}

template <typename idx_t>
std::string node_pattern<idx_t>::getIdentifier()
{
  return identifier;
}

template <typename idx_t>
pattern_type node_pattern<idx_t>::type()
{
  return pattern_type::Node;
}

template <typename idx_t>
std::vector<std::string>& node_pattern<idx_t>::getLabels()
{
  return labels;
}

template <typename idx_t>
std::map<std::string, value>& node_pattern<idx_t>::getProperties()
{
  return properties;
}

template class node_pattern<int32_t>;
template class node_pattern<int64_t>;

template <typename idx_t>
relationship_pattern<idx_t>::relationship_pattern()
{
  direction = 1;
}

template <typename idx_t>
relationship_pattern<idx_t>::relationship_pattern(const cypher_astnode_t* astNode,
                                                  context<idx_t>& ctx)
{
  // Check that the given astnode is the right type:
  std::string typeStr = getTypeString(astNode);
  CUGRAPH_EXPECTS(typeStr == "rel pattern", "Wrong type of astnode supplied");

  // Extract the direction of the relation
  direction = cypher_ast_rel_pattern_get_direction(astNode);

  // Extract the identifier or if null assign a unique identifier
  const cypher_astnode_t* id = cypher_ast_rel_pattern_get_identifier(astNode);
  if (id != nullptr) {
    identifier = cypher_ast_identifier_get_name(id);
  } else {
    identifier = ctx.getUniqueId();
  }

  // Extract the relationship types
  uint32_t num_types = cypher_ast_rel_pattern_nreltypes(astNode);
  for (uint32_t i = 0; i < num_types; i++) {
    const cypher_astnode_t* relType = cypher_ast_rel_pattern_get_reltype(astNode, i);
    std::string relTypeIdentifier   = cypher_ast_reltype_get_name(relType);
    relationshipTypes.emplace_back(value_type::Identifier, relTypeIdentifier);
  }

  // Extract the properties
  const cypher_astnode_t* mapNode = cypher_ast_rel_pattern_get_properties(astNode);
  if (mapNode != nullptr) {
    uint32_t num_pairs = cypher_ast_map_nentries(mapNode);
    for (uint32_t i = 0; i < num_pairs; i++) {
      // Getting the key value for the property map entry
      const cypher_astnode_t* propName = cypher_ast_map_get_key(mapNode, i);
      std::string propNameValue        = cypher_ast_prop_name_get_value(propName);

      // Getting the value for the property map entry
      const cypher_astnode_t* propValue = cypher_ast_map_get_value(mapNode, i);
      std::string valueType             = getTypeString(propValue);
      if (valueType == "string") {
        std::string valueStr = cypher_ast_string_get_value(propValue);
        value val(value_type::String, valueStr);
        properties[propNameValue] = val;
      }
      if (valueType == "property") {
        const cypher_astnode_t* expression = cypher_ast_property_operator_get_expression(propValue);
        std::string name                   = cypher_ast_identifier_get_name(expression);

        const cypher_astnode_t* propName2 = cypher_ast_property_operator_get_prop_name(propValue);
        std::string v                     = cypher_ast_prop_name_get_value(propName2);
        value val(name, v);
        properties[propNameValue] = val;
      }
      if (valueType == "subscript") {
        const cypher_astnode_t* expression =
          cypher_ast_subscript_operator_get_expression(propValue);
        std::string exp                   = cypher_ast_identifier_get_name(expression);
        const cypher_astnode_t* subscript = cypher_ast_subscript_operator_get_subscript(propValue);
        std::string subscriptStr          = cypher_ast_integer_get_valuestr(subscript);
        std::stringstream ss(subscriptStr);
        uint32_t index;
        ss >> index;
        value val(exp, index);
        properties[propNameValue] = val;
      }
    }
  }
}

template <typename idx_t>
void relationship_pattern<idx_t>::addProperty(std::string name, value val)
{
  properties[name] = val;
}

template <typename idx_t>
void relationship_pattern<idx_t>::setStart(std::string start)
{
  startId = start;
}

template <typename idx_t>
void relationship_pattern<idx_t>::setEnd(std::string end)
{
  endId = end;
}

template <typename idx_t>
void relationship_pattern<idx_t>::addType(value type)
{
  relationshipTypes.push_back(type);
}

template <typename idx_t>
void relationship_pattern<idx_t>::setDirection(uint32_t dir)
{
  direction = dir;
}

template <typename idx_t>
std::string relationship_pattern<idx_t>::getStart()
{
  return startId;
}

template <typename idx_t>
std::string relationship_pattern<idx_t>::getEnd()
{
  return endId;
}

template <typename idx_t>
uint32_t relationship_pattern<idx_t>::getDirection()
{
  return direction;
}

template <typename idx_t>
std::vector<value>& relationship_pattern<idx_t>::getTypes()
{
  return relationshipTypes;
}

template <typename idx_t>
std::map<std::string, value>& relationship_pattern<idx_t>::getProperties()
{
  return properties;
}

template <typename idx_t>
std::string relationship_pattern<idx_t>::getIdentifier()
{
  return identifier;
}

template <typename idx_t>
pattern_type relationship_pattern<idx_t>::type()
{
  return pattern_type::Relationship;
}

template class relationship_pattern<int32_t>;
template class relationship_pattern<int64_t>;

template <typename idx_t>
pattern_path<idx_t>::pattern_path(const cypher_astnode_t* astNode, context<idx_t>& ctx)
{
  // Check that the given astnode is the right type:
  std::string typeStr = getTypeString(astNode);
  CUGRAPH_EXPECTS(typeStr == "pattern path", "Wrong type of astnode supplied");

  // For each child of this pattern path create a node pattern or relationship pattern
  uint32_t num_children = cypher_astnode_nchildren(astNode);
  for (uint32_t i = 0; i < num_children; i++) {
    const cypher_astnode_t* child = cypher_astnode_get_child(astNode, i);
    std::string childType         = getTypeString(child);
    if (childType == "rel pattern") {
      pattern_element<idx_t>* pe = new relationship_pattern<idx_t>(child, ctx);
      path.push_back(pe);
    }
    if (childType == "node pattern") {
      pattern_element<idx_t>* pe = new node_pattern<idx_t>(child, ctx);
      path.push_back(pe);
    }
  }

  // Go through the nodes added on to the path to finalize relationship patterns
  for (size_t i = 0; i < path.size(); i++) {
    if (path[i]->type() == pattern_type::Relationship) {
      relationship_pattern<idx_t>* relPat = reinterpret_cast<relationship_pattern<idx_t>*>(path[i]);
      uint32_t dir                        = relPat->getDirection();
      if (dir == 1) {
        relPat->setStart(path[i - 1]->getIdentifier());
        relPat->setEnd(path[i + 1]->getIdentifier());
      }
      if (dir == 0) {
        relPat->setStart(path[i + 1]->getIdentifier());
        relPat->setEnd(path[i - 1]->getIdentifier());
      }
      if (dir == 2) { CUGRAPH_FAIL("Undirected edges not currently supported."); }
    }
  }
}

template <typename idx_t>
pattern_path<idx_t>::pattern_path(pattern_path<idx_t>&& other)
{
  path = std::move(other.path);
  other.path.clear();
}

template <typename idx_t>
pattern_path<idx_t>::~pattern_path()
{
  for (size_t i = 0; i < path.size(); i++) delete path[i];
}

template <typename idx_t>
pattern_path<idx_t>& pattern_path<idx_t>::operator=(pattern_path<idx_t>&& other)
{
  if (this != &other) {
    path = std::move(other.path);
    other.path.clear();
  }
  return *this;
}

template <typename idx_t>
std::vector<pattern_element<idx_t>*>& pattern_path<idx_t>::getPathNodes()
{
  return path;
}

template <typename idx_t>
bool pattern_path<idx_t>::hasBoundVariables(context<idx_t>& ctx)
{
  for (size_t i = 0; i < path.size(); i++) {
    std::string nodeId = path[i]->getIdentifier();
    if (path[i]->type() == pattern_type::Node && ctx.hasVariable(nodeId)) return true;
  }
  return false;
}

template <typename idx_t>
bool pattern_path<idx_t>::hasNamedValues(context<idx_t>& ctx)
{
  for (size_t i = 0; i < path.size(); i++) {
    if (path[i]->type() == pattern_type::Node) {
      node_pattern<idx_t>* node = reinterpret_cast<node_pattern<idx_t>*>(path[i]);
      for (auto it = node->getProperties().begin(); it != node->getProperties().end(); it++) {
        if (it->second.type() == value_type::Array_Access ||
            it->second.type() == value_type::Property_Access)
          return true;
      }
    } else {
      relationship_pattern<idx_t>* rel = reinterpret_cast<relationship_pattern<idx_t>*>(path[i]);
      for (auto it = rel->getProperties().begin(); it != rel->getProperties().end(); it++) {
        if (it->second.type() == value_type::Array_Access ||
            it->second.type() == value_type::Property_Access)
          return true;
      }
    }
  }
  return false;
}

template class pattern_path<int32_t>;
template class pattern_path<int64_t>;

}  // namespace db
}  // namespace cugraph
