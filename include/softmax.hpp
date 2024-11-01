#pragma once
#include "matrix.hpp"

Matrix softmax_seq(const Matrix &);
Matrix softmax_par(const Matrix &);
Matrix softmax_simd(const Matrix &);
Matrix softmax_par_simd(const Matrix &);
Matrix softmax_simt(const Matrix &);
