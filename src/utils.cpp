#include "utils.hpp"

#include <random>

Matrix generate_random_matrix(const std::size_t rows, const std::size_t cols,
                              const float min_val, const float max_val) {
  Matrix matrix(rows, cols);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min_val, max_val);

  for (std::size_t i = 0; i < rows * cols; ++i) {
    matrix[i] = dis(gen);
  }

  return matrix;
}