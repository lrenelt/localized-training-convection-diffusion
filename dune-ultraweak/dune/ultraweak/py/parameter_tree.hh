#ifndef DUNE_ULTRAWEAK_PY_PARAMETER_TREE_HH
#define DUNE_ULTRAWEAK_PY_PARAMETER_TREE_HH

#include <map>

#include <dune/python/pybind11/pybind11.h>

#include <dune/common/parametertree.hh>

namespace py = pybind11;

class StringConversionException : public Dune::Exception {};

std::string toString(py::handle handle)
{
  std::string type = handle.get_type().attr("__name__").cast<std::string>();
  if (type == "str")
    return handle.cast<std::string>();
  else if (type == "int")
    return std::to_string(handle.cast<int>());
  else if (type == "float") {
    std::stringstream sstr;
    sstr << handle.cast<double>();
    return sstr.str();
  } else if (type == "bool")
    return std::to_string(handle.cast<bool>());
  else if ((type == "list") || (type == "tuple")) {
    std::stringstream str;
    unsigned int i = 0;
    for (auto it = handle.begin(); it != handle.end(); ++it, ++i) {
      if (i > 0)
        str << " ";
      str << toString(*it);
    }
    return str.str();
  }
  DUNE_THROW(StringConversionException, "type \"" << type << "\" not supported");
}

std::map<std::string, std::string> toStringMap(py::dict dict)
{
  std::map<std::string, std::string> map;
  for (const auto& item : dict) {
    std::string type = item.second.get_type().attr("__name__").cast<std::string>();
    std::string key = toString(item.first);
    if (type == "dict") {
      auto sub = toStringMap(item.second.cast<py::dict>());
      for (const auto& k : sub) {
        map[key + "." + k.first] = k.second;
      }
    } else {
      try {
        map[key] = toString(item.second);
      } catch (StringConversionException& ex) {
        // ignore the entry. will be triggered for numpy arrays, which we do not want to be
        // converted to string
      }
    }
  }
  return map;
}

/**
   Converts a python dictionary to a Dune::ParameterTree.
   Copied from dune-hypercut
*/
Dune::ParameterTree toParameterTree(py::dict dict)
{
  Dune::ParameterTree tree;
  auto map = toStringMap(dict);
  for (const auto& k : map) {
    tree[k.first] = k.second;
  }
  return tree;
}

#endif  // DUNE_ULTRAWEAK_PY_PARAMETER_TREE_HH
