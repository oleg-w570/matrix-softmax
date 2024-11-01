#include "softmax.hpp"

#include <cmath>

Matrix softmax_seq(const Matrix &m) {
  Matrix res(m.rows(), m.cols());

  for (std::size_t i = 0; i < m.rows(); ++i) {
    float sum_exp = 0.0f;

    for (std::size_t j = 0; j < m.cols(); ++j) {
      res(i, j) = std::exp(m(i, j));
      sum_exp += res(i, j);
    }
    for (std::size_t j = 0; j < m.cols(); ++j) {
      res(i, j) /= sum_exp;
    }
  }

  return res;
}

Matrix softmax_par(const Matrix &m) {
  Matrix res(m.rows(), m.cols());

#pragma omp parallel for
  for (std::size_t i = 0; i < m.rows(); ++i) {
    float sum_exp = 0.0f;

    for (std::size_t j = 0; j < m.cols(); ++j) {
      res(i, j) = std::exp(m(i, j));
      sum_exp += res(i, j);
    }
    for (std::size_t j = 0; j < m.cols(); ++j) {
      res(i, j) /= sum_exp;
    }
  }

  return res;
}

Matrix softmax_simd(const Matrix &m) { return m; }

Matrix softmax_par_simd(const Matrix &m) { return m; }

Matrix softmax_simt(const Matrix &m) { return m; }
