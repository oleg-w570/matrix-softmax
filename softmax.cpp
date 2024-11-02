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

void softmaxSequential(const float *input_matrix, float *const output_matrix,
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

    for (std::size_t j = 0; j < n / 16; ++j) {
      __m512 elems = _mm512_load_ps(input_matrix + i * 16);
      __m512 exps = _mm512_exp_ps(elems);
      sum_exp += _mm512_reduce_add_ps(exps);
      _mm512_store_ps(output_matrix + i * 16, exps);
    }

    __m512 sum_exp_vec = _mm512_set1_ps(sum_exp);

    for (std::size_t j = 0; j < n / 16; ++j) {
      __m512 elems = _mm512_load_ps(output_matrix + j * 16);
      __m512 divs = _mm512_div_ps(elems, sum_exp_vec);
      _mm512_store_ps(output_matrix + j * 16, divs);
    }
  }
}

void softmaxParallelSimd(const float *input_matrix, float *const output_matrix,
                         const std::size_t n) {
#pragma omp parallel for
  for (std::size_t i = 0; i < n; ++i) {
    float sum_exp = 0.0f;

    for (std::size_t j = 0; j < n / 16; ++j) {
      __m512 elems = _mm512_load_ps(input_matrix + i * 16);
      __m512 exps = _mm512_exp_ps(elems);
      sum_exp += _mm512_reduce_add_ps(exps);
      _mm512_store_ps(output_matrix + i * 16, exps);
    }

    __m512 sum_exp_vec = _mm512_set1_ps(sum_exp);

    for (std::size_t j = 0; j < n / 16; ++j) {
      __m512 elems = _mm512_load_ps(output_matrix + j * 16);
      __m512 divs = _mm512_div_ps(elems, sum_exp_vec);
      _mm512_store_ps(output_matrix + j * 16, divs);
    }
  }
}

// void softmaxSimt(const float *input_matrix, float *const output_matrix,
//                  const std::size_t n) {
//   for (std::size_t i = 0; i < n; ++i) {
//     float sum_exp = 0.0f;

//     for (std::size_t j = 0; j < n; ++j) {
//       output_matrix[i * n + j] = std::exp(input_matrix[i * n + j]);
//       sum_exp += output_matrix[i * n + j];
//     }
//     for (std::size_t j = 0; j < n; ++j) {
//       output_matrix[i * n + j] /= sum_exp;
//     }
//   }
// }

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
    return 1;
  }
  const std::size_t n = std::stoul(argv[1]);
  auto *matrix = new float[n * n];
  fillRandomMatrix(matrix, n, 0.1f, 100.0f);

  float *res_seq = new float[n * n];
  const auto start_seq = std::chrono::high_resolution_clock::now();
  softmaxSequential(matrix, res_seq, n);
  const std::chrono::duration<double> duration_seq =
      std::chrono::high_resolution_clock::now() - start_seq;
  std::cout << "Sequential time: " << duration_seq.count() << std::endl;

  float *res_par = new float[n * n];
  const double start_par = omp_get_wtime();
  softmaxParallel(matrix, res_par, n);
  const double end_par = omp_get_wtime();
  std::cout << "Parallel time: " << end_par - start_par << std::endl;
  delete[] res_par;

  float *res_simd = new float[n * n];
  const auto start_simd = std::chrono::high_resolution_clock::now();
  softmaxSimd(matrix, res_simd, n);
  const std::chrono::duration<double> duration_simd =
      std::chrono::high_resolution_clock::now() - start_simd;
  std::cout << "Simd time: " << duration_simd.count() << std::endl;
  delete[] res_simd;

  float *res_par_simd = new float[n * n];
  const double start_par_simd = omp_get_wtime();
  softmaxParallelSimd(matrix, res_par_simd, n);
  const double end_par_simd = omp_get_wtime();
  std::cout << "Parallel Simd time: " << end_par_simd - start_par_simd << std::endl;
  delete[] res_par_simd;

  delete[] res_seq;
  delete[] matrix;
}