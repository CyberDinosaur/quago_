// Derived from efanna2e (Copyright (c) 2017 ZJULearning, MIT license).
#pragma once

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace quago {

// A type-erased string->string parameter bag. Values are stored as strings and
// parsed back on Get<T>(). Supports scalars, std::string, and std::vector of
// either. This is the established configuration mechanism across the codebase.
class Parameters {
 public:
  template <typename ParamType>
  void Set(const std::string& name, const ParamType& value) {
    params_[name] = ToString(value);
  }

  template <typename ParamType>
  ParamType Get(const std::string& name) const {
    auto it = params_.find(name);
    if (it == params_.end()) {
      throw std::invalid_argument("Invalid parameter name: " + name);
    }
    return FromString<ParamType>(it->second);
  }

  template <typename ParamType>
  ParamType Get(const std::string& name, const ParamType& default_value) const {
    auto it = params_.find(name);
    if (it == params_.end()) return default_value;
    return FromString<ParamType>(it->second);
  }

 private:
  std::unordered_map<std::string, std::string> params_;

  /* ------------------- serialization ------------------- */

  template <typename T>
  std::string ToString(const T& value) const {
    std::ostringstream oss;
    oss << value;
    return oss.str();
  }

  std::string ToString(const std::vector<std::string>& vec) const {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i + 1 < vec.size()) oss << ',';
    }
    return oss.str();
  }

  template <typename T>
  std::string ToString(const std::vector<T>& vec) const {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i + 1 < vec.size()) oss << ',';
    }
    return oss.str();
  }

  /* ------------------- deserialization ------------------- */

  template <typename T>
  T FromStringBasic(const std::string& s) const {
    std::istringstream iss(s);
    T value;
    if (!(iss >> value)) {
      throw std::runtime_error("Failed to parse parameter value: " + s);
    }
    return value;
  }

  std::vector<std::string> FromStringVectorString(const std::string& s) const {
    std::vector<std::string> result;
    std::string temp = s;
    std::replace(temp.begin(), temp.end(), ';', ',');
    std::replace(temp.begin(), temp.end(), ' ', ',');
    std::istringstream ss(temp);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) result.push_back(token);
    }
    return result;
  }

  template <typename T>
  std::vector<T> FromStringVector(const std::string& s) const {
    std::vector<T> result;
    std::string temp = s;
    std::replace(temp.begin(), temp.end(), ';', ',');
    std::replace(temp.begin(), temp.end(), ' ', ',');
    std::istringstream ss(temp);
    std::string token;
    while (std::getline(ss, token, ',')) {
      if (!token.empty()) {
        std::istringstream ts(token);
        T v;
        if (ts >> v) result.push_back(v);
      }
    }
    return result;
  }

  /* ------------------- dispatcher ------------------- */

  template <typename T>
  T FromString(const std::string& s) const {
    // std::string is returned verbatim: the empty string is a valid value
    // (default for optional path parameters) and would fail istringstream >>.
    if constexpr (std::is_same_v<T, std::string>) {
      return s;
    } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
      return FromStringVectorString(s);
    } else if constexpr (is_vector_of_number<T>::value) {
      using ElemType = typename T::value_type;
      return FromStringVector<ElemType>(s);
    } else {
      return FromStringBasic<T>(s);
    }
  }

  /* ------------------- type trait ------------------- */

  template <typename T>
  struct is_vector_of_number : std::false_type {};

  template <typename U>
  struct is_vector_of_number<std::vector<U>> {
    static constexpr bool value = std::is_arithmetic<U>::value;
  };
};

}  // namespace quago
