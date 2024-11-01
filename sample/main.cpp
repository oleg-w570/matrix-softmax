#include <chrono>
#include <iostream>
#include <omp.h>

#include "softmax.hpp"
#include "utils.hpp"

int main(int argc, void *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <matrix_size>\n";
    return 1;
  }
  const std::size_t n = std::stoul(static_cast<char*>(argv[1]));
  const auto matrix = generate_random_matrix(n, n, 0.01f, 100.0f);

  const auto start_seq = std::chrono::high_resolution_clock::now();
  const auto res_seq = softmax_seq(matrix);
  const std::chrono::duration<double> duration_seq =
      std::chrono::high_resolution_clock::now() - start_seq;
  std::cout << "Sequential time: " << duration_seq.count() << std::endl;

  const double start_par = omp_get_wtime();
  const auto res_par = softmax_par(matrix);
  const double end_par = omp_get_wtime();
  std::cout << "Parallel time: " << end_par - start_par << std::endl;
}