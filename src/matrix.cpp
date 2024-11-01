#include "matrix.hpp"

Matrix::Matrix(const std::size_t rows, const std::size_t cols)
    : data_(rows * cols), rows_(rows), cols_(cols) {}

float &Matrix::operator()(const std::size_t i, const std::size_t j) {
    return data_[i * cols_ + j];
}

const float &Matrix::operator()(const std::size_t i, const std::size_t j) const {
    return data_[i * cols_ + j];
}

float &Matrix::operator[](const std::size_t i) {
    return data_[i];
}

const float &Matrix::operator[](const std::size_t i) const {
    return data_[i];
}

std::size_t Matrix::rows() const {
    return rows_;
}

std::size_t Matrix::cols() const {
    return cols_;
}
