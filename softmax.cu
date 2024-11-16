#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <immintrin.h>
#include <omp.h>

#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <string>

constexpr int VECTOR_SIZE = sizeof(__m256) / sizeof(float);

static void fillRandomMatrix(float* const matrix, const int n,
                             const float min_val, const float max_val) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min_val, max_val);

  for (int i = 0; i < n * n; ++i) {
    matrix[i] = dis(gen);
  }
}

static void softmaxSequential(const float* input_matrix,
                              float* const output_matrix, const int n) {
  for (int i = 0; i < n; ++i) {
    const float* row_in = input_matrix + i * n;
    float* const row_out = output_matrix + i * n;
    float sum_exp = 0.0f;

    for (int j = 0; j < n; ++j) {
      const float exp_val = std::exp(row_in[j]);
      row_out[j] = exp_val;
      sum_exp += exp_val;
    }

    const float sum_exp_inv = 1.0f / sum_exp;
    for (int j = 0; j < n; ++j) {
      row_out[j] *= sum_exp_inv;
    }
  }
}

static void softmaxParallel(const float* input_matrix,
                            float* const output_matrix, const int n) {
#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    const float* row_in = input_matrix + i * n;
    float* const row_out = output_matrix + i * n;
    float sum_exp = 0.0f;

    for (int j = 0; j < n; ++j) {
      const float exp_val = std::exp(row_in[j]);
      row_out[j] = exp_val;
      sum_exp += exp_val;
    }

    const float sum_exp_inv = 1.0f / sum_exp;
    for (int j = 0; j < n; ++j) {
      row_out[j] *= sum_exp_inv;
    }
  }
}

static void softmaxVectorized(const float* input_matrix,
                              float* const output_matrix, const int n) {
  const int n_vec = n / VECTOR_SIZE * VECTOR_SIZE;

  for (int i = 0; i < n; ++i) {
    const float* row_in = input_matrix + i * n;
    float* const row_out = output_matrix + i * n;
    float sum_exp = 0.0f;
    __m256 sum_exp_vec = _mm256_setzero_ps();

    for (int j = 0; j < n_vec; j += VECTOR_SIZE) {
      __m256 vals = _mm256_loadu_ps(row_in + j);
      __m256 exp_vals = _mm256_exp_ps(vals);
      sum_exp_vec = _mm256_add_ps(sum_exp_vec, exp_vals);
      _mm256_storeu_ps(row_out + j, exp_vals);
    }

    for (int j = n_vec; j < n; ++j) {
      row_out[j] = std::exp(row_in[j]);
      sum_exp += row_out[j];
    }

    float sum_array[VECTOR_SIZE];
    _mm256_storeu_ps(sum_array, sum_exp_vec);
    for (int k = 0; k < VECTOR_SIZE; ++k) {
      sum_exp += sum_array[k];
    }

    const float sum_exp_inv = 1.0f / sum_exp;
    const __m256 sum_exp_inv_vec = _mm256_set1_ps(sum_exp_inv);

    for (int j = 0; j < n_vec; j += VECTOR_SIZE) {
      __m256 vals = _mm256_loadu_ps(row_out + j);
      __m256 normalized_vals = _mm256_mul_ps(vals, sum_exp_inv_vec);
      _mm256_storeu_ps(row_out + j, normalized_vals);
    }
    for (int j = n_vec; j < n; ++j) {
      row_out[j] *= sum_exp_inv;
    }
  }
}

static void softmaxParallelVectorized(const float* input_matrix,
                                      float* const output_matrix, const int n) {
  const int n_vec = n / VECTOR_SIZE * VECTOR_SIZE;

#pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    const float* row_in = input_matrix + i * n;
    float* const row_out = output_matrix + i * n;
    float sum_exp = 0.0f;
    __m256 sum_exp_vec = _mm256_setzero_ps();

    for (int j = 0; j < n_vec; j += VECTOR_SIZE) {
      __m256 vals = _mm256_loadu_ps(row_in + j);
      __m256 exp_vals = _mm256_exp_ps(vals);
      sum_exp_vec = _mm256_add_ps(sum_exp_vec, exp_vals);
      _mm256_storeu_ps(row_out + j, exp_vals);
    }

    for (int j = n_vec; j < n; ++j) {
      row_out[j] = std::exp(row_in[j]);
      sum_exp += row_out[j];
    }

    float sum_array[VECTOR_SIZE];
    _mm256_storeu_ps(sum_array, sum_exp_vec);
    for (int k = 0; k < VECTOR_SIZE; ++k) {
      sum_exp += sum_array[k];
    }

    const float sum_exp_inv = 1.0f / sum_exp;
    const __m256 sum_exp_inv_vec = _mm256_set1_ps(sum_exp_inv);

    for (int j = 0; j < n_vec; j += VECTOR_SIZE) {
      __m256 vals = _mm256_loadu_ps(row_out + j);
      __m256 normalized_vals = _mm256_mul_ps(vals, sum_exp_inv_vec);
      _mm256_storeu_ps(row_out + j, normalized_vals);
    }
    for (int j = n_vec; j < n; ++j) {
      row_out[j] *= sum_exp_inv;
    }
  }
}

__global__ void softmaxKernel(const float* __restrict__ input, float* output,
                              int n) {
  extern __shared__ float shared_exp[];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int idx = row * n + tid;

  const float val = input[idx];
  const float exp_val = expf(val);
  shared_exp[tid] = exp_val;
  __syncthreads();

  for (int stride = n / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_exp[tid] += shared_exp[tid + stride];
    }
    __syncthreads();
  }

  const float sum_exp = shared_exp[0];
  output[idx] = exp_val / sum_exp;
}

static void softmaxCuda(const float* input_matrix, float* output_matrix,
                        const int n) {
  float* d_input = nullptr;
  float* d_output = nullptr;
  const auto matrix_size = n * n * sizeof(float);

  cudaMalloc((void**)&d_input, matrix_size);
  cudaMalloc((void**)&d_output, matrix_size);
  cudaMemcpy(d_input, input_matrix, matrix_size, cudaMemcpyHostToDevice);

  const int threads_per_block = n;
  const int blocks_per_grid = n;
  const auto shared_memory_size = n * sizeof(float);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  softmaxKernel<<<blocks_per_grid, threads_per_block, shared_memory_size>>>(
      d_input, d_output, n);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << "CUDA time: " << milliseconds / 1000.0 << " sec.";

  cudaMemcpy(output_matrix, d_output, matrix_size, cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_input);
  cudaFree(d_output);
}

static void printExecutionTime(const char* label,
                               void (*softmax_func)(const float*, float* const,
                                                    const int),
                               const float* input_matrix,
                               float* const output_matrix, const int n) {
  const double start = omp_get_wtime();
  softmax_func(input_matrix, output_matrix, n);
  const double end = omp_get_wtime();
  std::cout << label << " time: " << end - start << " sec. ";
}

static void printMaxDifference(const float* a, const float* b, const int n) {
  float max_difference = 0.0f;

  for (int i = 0; i < n * n; ++i) {
    max_difference = std::max(max_difference, std::abs(a[i] - b[i]));
  }

  std::cout << "(diff: " << max_difference << ")" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
    return 1;
  }
  const int n = std::stoi(argv[1]);
  float* input_matrix = new float[n * n];
  fillRandomMatrix(input_matrix, n, 0.0f, 1.0f);

  float* seq_output_matrix = new float[n * n];
  printExecutionTime("Sequential", softmaxSequential, input_matrix,
                     seq_output_matrix, n);
  printMaxDifference(seq_output_matrix, seq_output_matrix, n);

  float* par_output_matrix = new float[n * n];
  printExecutionTime("Parallel", softmaxParallel, input_matrix,
                     par_output_matrix, n);
  printMaxDifference(seq_output_matrix, par_output_matrix, n);
  delete[] par_output_matrix;

  float* vec_output_matrix = new float[n * n];
  printExecutionTime("Vectorized", softmaxVectorized, input_matrix,
                     vec_output_matrix, n);
  printMaxDifference(seq_output_matrix, vec_output_matrix, n);
  delete[] vec_output_matrix;

  float* par_vec_output_matrix = new float[n * n];
  printExecutionTime("Parallel Vectorized", softmaxParallelVectorized,
                     input_matrix, par_vec_output_matrix, n);
  printMaxDifference(seq_output_matrix, par_vec_output_matrix, n);
  delete[] par_vec_output_matrix;

  float* cuda_output_matrix = new float[n * n];
  softmaxCuda(input_matrix, cuda_output_matrix, n);
  printMaxDifference(seq_output_matrix, cuda_output_matrix, n);
  delete[] cuda_output_matrix;

  delete[] seq_output_matrix;
  delete[] input_matrix;

  return 0;
}
