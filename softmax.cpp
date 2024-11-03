#include <immintrin.h>
#include <omp.h>

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

constexpr std::size_t VECTOR_SIZE = 16;

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
    const float *row_in = input_matrix + i * n;
    float *const row_out = output_matrix + i * n;

    for (std::size_t j = 0; j < n; ++j) {
      row_out[j] = std::exp(row_in[j]);
      sum_exp += row_out[j];
    }

    const float sum_exp_inv = 1.0f / sum_exp;
    for (std::size_t j = 0; j < n; ++j) {
      row_out[j] *= sum_exp_inv;
    }
  }
}

void softmaxParallel(const float *input_matrix, float *const output_matrix,
                     const std::size_t n) {
#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < n; ++i) {
    float sum_exp = 0.0f;
    const float *row_in = input_matrix + i * n;
    float *const row_out = output_matrix + i * n;

    for (std::size_t j = 0; j < n; ++j) {
      row_out[j] = std::exp(row_in[j]);
      sum_exp += row_out[j];
    }

    const float sum_exp_inv = 1.0f / sum_exp;
    for (std::size_t j = 0; j < n; ++j) {
      row_out[j] *= sum_exp_inv;
    }
  }
}

void softmaxSimd(const float *input_matrix, float *const output_matrix,
                 const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    float sum_exp = 0.0f;
    const float *row_in = input_matrix + i * n;
    float *const row_out = output_matrix + i * n;

    const std::size_t size = n / VECTOR_SIZE * VECTOR_SIZE;
    for (std::size_t j = 0; j < size; j += VECTOR_SIZE) {
      __m512 elems = _mm512_load_ps(row_in + j);
      __m512 exps = _mm512_exp_ps(elems);
      sum_exp += _mm512_reduce_add_ps(exps);
      _mm512_store_ps(row_out + j, exps);
    }

    const __m512 sum_exp_vec = _mm512_set1_ps(sum_exp);

    for (std::size_t j = 0; j < size; j += VECTOR_SIZE) {
      __m512 elems = _mm512_load_ps(row_out + j);
      _mm512_store_ps(row_out + j, _mm512_div_ps(elems, sum_exp_vec));
    }
  }
}

void softmaxParallelSimd(const float *input_matrix, float *const output_matrix,
                         const std::size_t n) {
#pragma omp parallel for schedule(dynamic, 16)
  for (std::size_t i = 0; i < n; ++i) {
    float sum_exp = 0.0f;
    const float *row_in = input_matrix + i * n;
    float *const row_out = output_matrix + i * n;

    const std::size_t size = n / VECTOR_SIZE * VECTOR_SIZE;
    for (std::size_t j = 0; j < size; j += VECTOR_SIZE) {
      __m512 elems = _mm512_load_ps(row_in + j);
      __m512 exps = _mm512_exp_ps(elems);
      sum_exp += _mm512_reduce_add_ps(exps);
      _mm512_store_ps(row_out + j, exps);
    }

    const __m512 sum_exp_vec = _mm512_set1_ps(sum_exp);

    for (std::size_t j = 0; j < size; j += VECTOR_SIZE) {
      __m512 elems = _mm512_load_ps(row_out + j);
      _mm512_store_ps(row_out + j, _mm512_div_ps(elems, sum_exp_vec));
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

void printExecutionTime(const char *label,
                        void (*softmax_func)(const float *, float *const,
                                             const std::size_t),
                        const float *input_matrix, float *const output_matrix,
                        const std::size_t n) {
  const double start = omp_get_wtime();
  softmax_func(input_matrix, output_matrix, n);
  const double end = omp_get_wtime();
  std::cout << label << " time: " << end - start << " ";
}

void printMaxDifference(const float *a, const float *b, const std::size_t n) {
  float max_difference = 0.0f;

  for (std::size_t i = 0; i < n; ++i) {
    max_difference = std::max(max_difference, std::abs(a[i] - b[i]));
  }

  std::cout << "(diff: " << max_difference << ")" << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
    return 1;
  }
  const std::size_t n = std::stoul(argv[1]);
  float *input_matrix = new float[n * n];
  fillRandomMatrix(input_matrix, n, 0.1f, 100.0f);

  float *seq_output_matrix = new float[n * n];
  printExecutionTime("Sequential", softmaxSequential, input_matrix,
                     seq_output_matrix, n);
  printMaxDifference(seq_output_matrix, seq_output_matrix, n);

  float *par_output_matrix = new float[n * n];
  printExecutionTime("Parallel", softmaxParallel, input_matrix,
                     par_output_matrix, n);
  printMaxDifference(seq_output_matrix, par_output_matrix, n);
  delete[] par_output_matrix;

  float *simd_output_matrix = new float[n * n];
  printExecutionTime("Simd", softmaxSimd, input_matrix, simd_output_matrix, n);
  printMaxDifference(seq_output_matrix, simd_output_matrix, n);
  delete[] simd_output_matrix;

  float *par_simd_output_matrix = new float[n * n];
  printExecutionTime("Parallel Simd", softmaxParallelSimd, input_matrix,
                     par_simd_output_matrix, n);
  printMaxDifference(seq_output_matrix, par_simd_output_matrix, n);
  delete[] par_simd_output_matrix;

  // float *simt_output_matrix = new float[n * n];
  // printExecutionTime("Simt", softmaxSimt, input_matrix, output_matrix, n);
  // delete[] simt_output_matrix;

  delete[] seq_output_matrix;
  delete[] input_matrix;
}