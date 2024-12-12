// solver.hpp
#pragma once
#include <vector>
#include <complex>
#include <cmath>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <algorithm>

template<typename T>
class SparseMatrix {
public:
    std::vector<std::complex<T>> values;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    int rows, cols;

    SparseMatrix(int r, int c) : rows(r), cols(c) {
        row_ptr.resize(r + 1, 0);
    }

    void from_csr(const std::vector<std::complex<T>>& vals,  // 修改参数类型
                 const std::vector<int>& indices,
                 const std::vector<int>& indptr) {
        values = vals;
        col_idx = indices;
        row_ptr = indptr;
    }

    std::string validate() const {
        std::stringstream errors;

        if (rows < 0 || cols < 0) {
            errors << "Invalid matrix dimensions\n";
        }

        if (row_ptr.size() != static_cast<size_t>(rows + 1)) {
            errors << "Invalid row_ptr size\n";
        }

        for (size_t i = 1; i < row_ptr.size(); ++i) {
            if (row_ptr[i] < row_ptr[i-1]) {
                errors << "Non-monotonic row_ptr\n";
                break;
            }
        }

        for (const auto& col : col_idx) {
            if (col < 0 || col >= cols) {
                errors << "Invalid column index\n";
                break;
            }
        }

        if (values.size() != col_idx.size()) {
            errors << "Inconsistent values and col_idx sizes\n";
        }

        if (!row_ptr.empty() && row_ptr.back() != static_cast<int>(values.size())) {
            errors << "Inconsistent row_ptr with number of non-zero elements\n";
        }

        return errors.str();
    }

    std::vector<std::complex<T>> get_diagonal_preconditioner() const {  // 修改返回类型
        std::vector<std::complex<T>> M(rows);
        for (int i = 0; i < rows; i++) {
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                if (col_idx[j] == i) {
                    M[i] = (std::abs(values[j]) > 1e-14) ? std::complex<T>(1) / values[j] : std::complex<T>(1);
                    break;
                }
            }
            if (std::abs(M[i]) == 0) M[i] = std::complex<T>(1);
        }
        return M;
    }

    std::vector<std::complex<T>> multiply(const std::vector<std::complex<T>>& x) const {
        std::vector<std::complex<T>> result(rows);
        for (int i = 0; i < rows; i++) {
            std::complex<T> sum = 0;
            for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++) {
                sum += values[j] * x[col_idx[j]];
            }
            result[i] = sum;
        }
        return result;
    }
};

template<typename T>
class IterativeSolver {
private:
    const SparseMatrix<T>& A;
    const std::vector<std::complex<T>>& b;
    T tol;
    int max_iter;
    bool verbose;
    std::vector<std::complex<T>> current_solution;

    struct IterationStats {
        int iteration;
        T residual;
        T rho;
        T error_estimate;
    };

    std::vector<IterationStats> convergence_history;

    T vector_norm(const std::vector<std::complex<T>>& v) const {
        T max_val = 0;
        for (const auto& x : v) {
            max_val = std::max(max_val, std::abs(x));
        }
        if (max_val == 0) return 0;

        T sum = 0;
        for (const auto& x : v) {
            T normalized = std::abs(x) / max_val;
            sum += normalized * normalized;
        }
        return max_val * std::sqrt(sum);
    }

    std::complex<T> dot_product(const std::vector<std::complex<T>>& a,
                               const std::vector<std::complex<T>>& b) const {
        T max_val_a = 0, max_val_b = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            max_val_a = std::max(max_val_a, std::abs(a[i]));
            max_val_b = std::max(max_val_b, std::abs(b[i]));
        }
        if (max_val_a == 0 || max_val_b == 0) return 0;

        std::complex<T> sum = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            sum += std::conj(a[i] / max_val_a) * (b[i] / max_val_b);
        }
        return sum * max_val_a * max_val_b;
    }

    void modified_gram_schmidt(std::vector<std::vector<std::complex<T>>>& v,
                             std::vector<std::vector<std::complex<T>>>& h,
                             int j) {
        for (int i = 0; i <= j; ++i) {
            h[i][j] = dot_product(v[i], v[j+1]);
            for (size_t k = 0; k < v[j+1].size(); ++k) {
                v[j+1][k] -= h[i][j] * v[i][k];
            }
        }
        h[j+1][j] = vector_norm(v[j+1]);
        if (std::abs(h[j+1][j]) > tol) {
            for (auto& vj : v[j+1]) {
                vj /= h[j+1][j];
            }
        }
    }

    std::vector<std::complex<T>> update_solution(const std::vector<std::complex<T>>& x0,
                                               const std::vector<std::vector<std::complex<T>>>& v,
                                               const std::vector<std::vector<std::complex<T>>>& h,
                                               const std::vector<std::complex<T>>& s,
                                               int j,
                                               const std::vector<std::complex<T>>& M) {
        std::vector<std::complex<T>> y(j+1);
        for (int i = j; i >= 0; --i) {
            y[i] = s[i];
            for (int k = i+1; k <= j; ++k) {
                y[i] -= h[i][k] * y[k];
            }
            y[i] /= h[i][i];
        }

        auto result = x0;
        for (int i = 0; i <= j; ++i) {
            for (size_t k = 0; k < result.size(); ++k) {
                result[k] += y[i] * v[i][k] / M[k];
            }
        }
        return result;
    }

public:
    IterativeSolver(const SparseMatrix<T>& matrix,
                   const std::vector<std::complex<T>>& rhs,
                   T tolerance = 1e-10,
                   int max_iterations = 1000,
                   bool verbose_output = false)
        : A(matrix), b(rhs), tol(tolerance), max_iter(max_iterations),
          verbose(verbose_output), current_solution(rhs.size()) {
        if (A.rows != static_cast<int>(b.size())) {
            throw std::runtime_error("Matrix and vector dimensions do not match");
        }

        std::string matrix_errors = A.validate();
        if (!matrix_errors.empty()) {
            throw std::runtime_error("Invalid matrix: " + matrix_errors);
        }
    }

    const std::vector<std::complex<T>>& get_current_solution() const {
        return current_solution;
    }

    std::vector<std::complex<T>> solve(const std::vector<std::complex<T>>& x0) {
        const int n = A.rows;
        auto M = A.get_diagonal_preconditioner();

        current_solution = x0;
        auto r = b;
        auto Ax = A.multiply(x0);
        for (int i = 0; i < n; ++i) {
            r[i] = (b[i] - Ax[i]) * M[i];
        }

        T beta = vector_norm(r);
        if (beta < tol) {
            if (verbose) {
                std::cout << "Initial guess satisfies tolerance" << std::endl;
            }
            return x0;
        }

        std::vector<std::vector<std::complex<T>>> v(max_iter + 1, std::vector<std::complex<T>>(n));
        for (int i = 0; i < n; ++i) {
            v[0][i] = r[i] / beta;
        }

        std::vector<std::vector<std::complex<T>>> h(max_iter + 1, std::vector<std::complex<T>>(max_iter));
        std::vector<std::complex<T>> s(max_iter + 1);
        std::vector<std::complex<T>> cs(max_iter + 1);
        std::vector<std::complex<T>> sn(max_iter + 1);

        s[0] = beta;
        T initial_residual = beta;

        for (int j = 0; j < max_iter; ++j) {
            auto w = A.multiply(v[j]);
            for (int i = 0; i < n; ++i) {
                w[i] *= M[i];
            }
            v[j+1] = w;

            modified_gram_schmidt(v, h, j);

            for (int i = 0; i < j; ++i) {
                std::complex<T> tmp = cs[i] * h[i][j] + std::conj(sn[i]) * h[i+1][j];
                h[i+1][j] = -sn[i] * h[i][j] + cs[i] * h[i+1][j];
                h[i][j] = tmp;
            }

            T rho = std::sqrt(std::norm(h[j][j]) + std::norm(h[j+1][j]));
            if (std::abs(rho) < tol * 1e-6) {
                if (verbose) {
                    std::cout << "Lucky breakdown at iteration " << j << std::endl;
                }
                current_solution = update_solution(x0, v, h, s, j, M);
                return current_solution;
            }

            cs[j] = h[j][j] / rho;
            sn[j] = h[j+1][j] / rho;
            h[j][j] = rho;
            h[j+1][j] = std::complex<T>(0);

            s[j+1] = -sn[j] * s[j];
            s[j] = cs[j] * s[j];

            T residual = std::abs(s[j+1]);

            if (verbose && (j+1) % 10 == 0) {
                std::cout << "Iteration " << (j+1) << ": residual = " << residual
                         << ", relative error = " << (residual / initial_residual) << std::endl;
            }

            convergence_history.push_back({j+1, residual,
                                         residual / initial_residual,
                                         residual / initial_residual});

            current_solution = update_solution(x0, v, h, s, j, M);

            if (residual < tol) {
                if (verbose) {
                    std::cout << "Convergence achieved at iteration " << (j+1) << std::endl;
                }
                return current_solution;
            }
        }

        if (verbose) {
            std::cout << "Maximum iterations reached. Returning current solution." << std::endl;
        }
        return current_solution;
    }

    const std::vector<IterationStats>& get_convergence_history() const {
        return convergence_history;
    }
};