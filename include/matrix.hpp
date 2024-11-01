#pragma once
#include <vector>

class Matrix {
  std::vector<float> data_;
  std::size_t rows_;
  std::size_t cols_;

 public:
  Matrix(const std::size_t rows, const std::size_t cols);

  float &operator()(const std::size_t i, const std::size_t j);

  const float &operator()(const std::size_t i, const std::size_t j) const;

  float &operator[](const std::size_t i);

  const float &operator[](const std::size_t i) const;

  std::size_t rows() const;

  std::size_t cols() const;
};