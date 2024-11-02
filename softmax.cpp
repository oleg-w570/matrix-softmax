#include <immintrin.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

void fillRandomMatrix(float *const matrix, const std::size_t n,
                      const float min_val, const float max_val) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min_val, max_val);

  for (std::size_t i = 0; i < n * n; ++i) {
    matrix[i] = dis(gen);
  }
}

void softmaxSequence(const float *input_matrix, float *const output_matrix,
                     const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum_exp = 0.0f;

    for (std::size_t j = 0; j < n; ++j) {
      output_matrix[i * n + j] = std::exp(input_matrix[i * n + j]);
      sum_exp += output_matrix[i * n + j];
    }
    for (std::size_t j = 0; j < n; ++j) {
      output_matrix[i * n + j] /= sum_exp;
    }
  }
}

void softmaxParallel(const float *input_matrix, float *const output_matrix,
                     const std::size_t n) {
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    float sum_exp = 0.0f;

    for (std::size_t j = 0; j < n; ++j) {
      output_matrix[i * n + j] = std::exp(input_matrix[i * n + j]);
      sum_exp += output_matrix[i * n + j];
    }
    for (std::size_t j = 0; j < n; ++j) {
      output_matrix[i * n + j] /= sum_exp;
    }
  }
}

void softmaxSimd(const float *input_matrix, float *const output_matrix,
                 const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum_exp = 0.0f;

    for (std::size_t j = 0; j < n / 8; ++j) {
      __m256 elems = _mm256_load_ps(input_matrix + i * 8);
      __m256 exps = _mm256_exp_ps(elems);
      output_matrix[i * n + j] = std::exp(input_matrix[i * n + j]);
      sum_exp += output_matrix[i * n + j];
    }
    for (std::size_t j = 0; j < n; ++j) {
      output_matrix[i * n + j] /= sum_exp;
    }
  }
}

void softmaxParallelSimd(const float *input_matrix, float *const output_matrix,
                         const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum_exp = 0.0f;

    for (std::size_t j = 0; j < n; ++j) {
      output_matrix[i * n + j] = std::exp(input_matrix[i * n + j]);
      sum_exp += output_matrix[i * n + j];
    }
    for (std::size_t j = 0; j < n; ++j) {
      output_matrix[i * n + j] /= sum_exp;
    }
  }
}

void softmaxSimt(const float *input_matrix, float *const output_matrix,
                 const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum_exp = 0.0f;

    for (std::size_t j = 0; j < n; ++j) {
      output_matrix[i * n + j] = std::exp(input_matrix[i * n + j]);
      sum_exp += output_matrix[i * n + j];
    }
    for (std::size_t j = 0; j < n; ++j) {
      output_matrix[i * n + j] /= sum_exp;
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
    return 1;
  }
  const std::size_t n = std::stoul(argv[1]);
  auto *matrix = new float[n * n];
  fillRandomMatrix(matrix, n, 0.1f, 100.0f);

  auto *res_seq = new float[n * n];
  const auto start_seq = std::chrono::high_resolution_clock::now();
  softmaxSequence(matrix, res_seq, n);
  const std::chrono::duration<double> duration_seq =
      std::chrono::high_resolution_clock::now() - start_seq;
  std::cout << "Sequential time: " << duration_seq.count() << std::endl;
  delete[] res_seq;

  auto *res_par = new float[n * n];
  const double start_par = omp_get_wtime();
  softmaxParallel(matrix, res_par, n);
  const double end_par = omp_get_wtime();
  std::cout << "Parallel time: " << end_par - start_par << std::endl;
  delete[] res_par;

  delete[] matrix;
}