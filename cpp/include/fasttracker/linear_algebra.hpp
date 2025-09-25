#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fasttracker {

class Matrix {
public:
  Matrix() : rows_(0), cols_(0) {}
  Matrix(std::size_t rows, std::size_t cols)
      : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}

  Matrix(std::size_t rows, std::size_t cols, std::initializer_list<double> init)
      : rows_(rows), cols_(cols), data_(init) {
    if (init.size() != rows * cols) {
      throw std::invalid_argument("Initializer size does not match matrix shape");
    }
  }

  double &operator()(std::size_t r, std::size_t c) {
    return data_.at(r * cols_ + c);
  }

  double operator()(std::size_t r, std::size_t c) const {
    return data_.at(r * cols_ + c);
  }

  std::size_t rows() const { return rows_; }
  std::size_t cols() const { return cols_; }

  const std::vector<double> &data() const { return data_; }
  std::vector<double> &data() { return data_; }

  static Matrix Identity(std::size_t n) {
    Matrix m(n, n);
    for (std::size_t i = 0; i < n; ++i) {
      m(i, i) = 1.0;
    }
    return m;
  }

  static Matrix Zeros(std::size_t rows, std::size_t cols) {
    return Matrix(rows, cols);
  }

private:
  std::size_t rows_;
  std::size_t cols_;
  std::vector<double> data_;
};

inline Matrix Diagonal(const std::vector<double> &diag) {
  Matrix m(diag.size(), diag.size());
  for (std::size_t i = 0; i < diag.size(); ++i) {
    m(i, i) = diag[i];
  }
  return m;
}

inline Matrix Transpose(const Matrix &m) {
  Matrix result(m.cols(), m.rows());
  for (std::size_t r = 0; r < m.rows(); ++r) {
    for (std::size_t c = 0; c < m.cols(); ++c) {
      result(c, r) = m(r, c);
    }
  }
  return result;
}

inline Matrix MatMul(const Matrix &a, const Matrix &b) {
  if (a.cols() != b.rows()) {
    throw std::invalid_argument("Matrix dimension mismatch in MatMul");
  }
  Matrix result(a.rows(), b.cols());
  for (std::size_t i = 0; i < a.rows(); ++i) {
    for (std::size_t k = 0; k < a.cols(); ++k) {
      double aik = a(i, k);
      if (aik == 0.0) {
        continue;
      }
      for (std::size_t j = 0; j < b.cols(); ++j) {
        result(i, j) += aik * b(k, j);
      }
    }
  }
  return result;
}

inline std::vector<double> MatVec(const Matrix &m, const std::vector<double> &v) {
  if (m.cols() != v.size()) {
    throw std::invalid_argument("Matrix/vector dimension mismatch in MatVec");
  }
  std::vector<double> result(m.rows(), 0.0);
  for (std::size_t r = 0; r < m.rows(); ++r) {
    double sum = 0.0;
    for (std::size_t c = 0; c < m.cols(); ++c) {
      sum += m(r, c) * v[c];
    }
    result[r] = sum;
  }
  return result;
}

inline std::vector<double> VecMat(const std::vector<double> &v, const Matrix &m) {
  if (v.size() != m.rows()) {
    throw std::invalid_argument("Vector/matrix dimension mismatch in VecMat");
  }
  std::vector<double> result(m.cols(), 0.0);
  for (std::size_t c = 0; c < m.cols(); ++c) {
    double sum = 0.0;
    for (std::size_t r = 0; r < m.rows(); ++r) {
      sum += v[r] * m(r, c);
    }
    result[c] = sum;
  }
  return result;
}

inline Matrix Add(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::invalid_argument("Matrix dimension mismatch in Add");
  }
  Matrix result(a.rows(), a.cols());
  for (std::size_t i = 0; i < a.data().size(); ++i) {
    result.data()[i] = a.data()[i] + b.data()[i];
  }
  return result;
}

inline Matrix Subtract(const Matrix &a, const Matrix &b) {
  if (a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::invalid_argument("Matrix dimension mismatch in Subtract");
  }
  Matrix result(a.rows(), a.cols());
  for (std::size_t i = 0; i < a.data().size(); ++i) {
    result.data()[i] = a.data()[i] - b.data()[i];
  }
  return result;
}

inline Matrix Outer(const std::vector<double> &a, const std::vector<double> &b) {
  Matrix result(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t j = 0; j < b.size(); ++j) {
      result(i, j) = a[i] * b[j];
    }
  }
  return result;
}

inline Matrix Inverse(const Matrix &input) {
  if (input.rows() != input.cols()) {
    throw std::invalid_argument("Matrix inversion requires a square matrix");
  }
  std::size_t n = input.rows();
  Matrix aug(n, 2 * n);
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t c = 0; c < n; ++c) {
      aug(r, c) = input(r, c);
    }
    for (std::size_t c = 0; c < n; ++c) {
      aug(r, n + c) = (r == c) ? 1.0 : 0.0;
    }
  }

  for (std::size_t i = 0; i < n; ++i) {
    // Pivot selection
    std::size_t pivot = i;
    double max_val = std::abs(aug(i, i));
    for (std::size_t r = i + 1; r < n; ++r) {
      double val = std::abs(aug(r, i));
      if (val > max_val) {
        max_val = val;
        pivot = r;
      }
    }
    if (max_val < 1e-12) {
      throw std::runtime_error("Matrix is singular and cannot be inverted");
    }
    if (pivot != i) {
      for (std::size_t c = 0; c < 2 * n; ++c) {
        std::swap(aug(i, c), aug(pivot, c));
      }
    }

    double pivot_val = aug(i, i);
    for (std::size_t c = 0; c < 2 * n; ++c) {
      aug(i, c) /= pivot_val;
    }

    for (std::size_t r = 0; r < n; ++r) {
      if (r == i) {
        continue;
      }
      double factor = aug(r, i);
      if (factor == 0.0) {
        continue;
      }
      for (std::size_t c = 0; c < 2 * n; ++c) {
        aug(r, c) -= factor * aug(i, c);
      }
    }
  }

  Matrix result(n, n);
  for (std::size_t r = 0; r < n; ++r) {
    for (std::size_t c = 0; c < n; ++c) {
      result(r, c) = aug(r, n + c);
    }
  }
  return result;
}

inline double Dot(const std::vector<double> &a, const std::vector<double> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Vector dot product requires equal sizes");
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

inline std::vector<double> Subtract(const std::vector<double> &a,
                                    const std::vector<double> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Vector subtraction requires equal sizes");
  }
  std::vector<double> result(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] - b[i];
  }
  return result;
}

inline std::vector<double> Add(const std::vector<double> &a,
                               const std::vector<double> &b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("Vector addition requires equal sizes");
  }
  std::vector<double> result(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    result[i] = a[i] + b[i];
  }
  return result;
}

}  // namespace fasttracker

