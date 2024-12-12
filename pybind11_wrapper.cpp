// pybind11_wrapper.cpp
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "solver.hpp"

namespace py = pybind11;

class PySparseMatrix {
private:
    SparseMatrix<double> mat;

public:
    PySparseMatrix(int rows, int cols) : mat(rows, cols) {}

    void from_csr(const py::array_t<std::complex<double>>& data,  // 修改为复数类型
                 const py::array_t<int>& indices,
                 const py::array_t<int>& indptr) {
        auto data_buf = data.request();
        auto indices_buf = indices.request();
        auto indptr_buf = indptr.request();

        mat.values = std::vector<std::complex<double>>(
            static_cast<std::complex<double>*>(data_buf.ptr),
            static_cast<std::complex<double>*>(data_buf.ptr) + data_buf.size
        );
        mat.col_idx = std::vector<int>(
            static_cast<int*>(indices_buf.ptr),
            static_cast<int*>(indices_buf.ptr) + indices_buf.size
        );
        mat.row_ptr = std::vector<int>(
            static_cast<int*>(indptr_buf.ptr),
            static_cast<int*>(indptr_buf.ptr) + indptr_buf.size
        );
    }

    std::string validate() const {
        return mat.validate();
    }

    py::array_t<std::complex<double>> multiply(const py::array_t<std::complex<double>>& x) {
        auto x_buf = x.request();
        std::vector<std::complex<double>> x_vec(
            static_cast<std::complex<double>*>(x_buf.ptr),
            static_cast<std::complex<double>*>(x_buf.ptr) + x_buf.size
        );

        auto result = mat.multiply(x_vec);

        return py::array_t<std::complex<double>>(
            {static_cast<py::ssize_t>(mat.rows)},
            result.data()
        );
    }

    const SparseMatrix<double>& get_internal_matrix() const { return mat; }
};

class PySolver {
private:
    std::unique_ptr<IterativeSolver<double>> solver;
    PySparseMatrix& matrix;
    std::vector<std::complex<double>> rhs;

public:
    PySolver(PySparseMatrix& mat,
             const py::array_t<std::complex<double>>& b,
             double tol = 1e-10,
             int max_iter = 1000,
             bool verbose = false)
        : matrix(mat) {
        auto b_buf = b.request();
        rhs = std::vector<std::complex<double>>(
            static_cast<std::complex<double>*>(b_buf.ptr),
            static_cast<std::complex<double>*>(b_buf.ptr) + b_buf.size
        );

        solver = std::make_unique<IterativeSolver<double>>(
            matrix.get_internal_matrix(), rhs, tol, max_iter, verbose
        );
    }

    py::array_t<std::complex<double>> solve(const py::array_t<std::complex<double>>& x0) {
        try {
            auto x0_buf = x0.request();
            std::vector<std::complex<double>> x0_vec(
                static_cast<std::complex<double>*>(x0_buf.ptr),
                static_cast<std::complex<double>*>(x0_buf.ptr) + x0_buf.size
            );

            auto result = solver->solve(x0_vec);

            return py::array_t<std::complex<double>>(
                {static_cast<py::ssize_t>(result.size())},
                result.data()
            );
        } catch (const std::exception&) {
            // 如果solve抛出异常，返回当前解
            return get_current_solution();
        }
    }

    py::array_t<std::complex<double>> get_current_solution() {
        const auto& result = solver->get_current_solution();
        return py::array_t<std::complex<double>>(
            {static_cast<py::ssize_t>(result.size())},
            result.data()
        );
    }

    py::dict get_convergence_history() {
        const auto& history = solver->get_convergence_history();

        std::vector<int> iterations;
        std::vector<double> residuals;
        std::vector<double> rhos;
        std::vector<double> error_estimates;

        for (const auto& stat : history) {
            iterations.push_back(stat.iteration);
            residuals.push_back(stat.residual);
            rhos.push_back(stat.rho);
            error_estimates.push_back(stat.error_estimate);
        }

        py::dict result;
        result["iterations"] = py::array_t<int>(iterations.size(), iterations.data());
        result["residuals"] = py::array_t<double>(residuals.size(), residuals.data());
        result["rhos"] = py::array_t<double>(rhos.size(), rhos.data());
        result["error_estimates"] = py::array_t<double>(error_estimates.size(), error_estimates.data());

        return result;
    }
};

PYBIND11_MODULE(solver, m) {
    py::class_<PySparseMatrix>(m, "SparseMatrix")
        .def(py::init<int, int>())
        .def("from_csr", &PySparseMatrix::from_csr)
        .def("validate", &PySparseMatrix::validate)
        .def("multiply", &PySparseMatrix::multiply);

    py::class_<PySolver>(m, "Solver")
        .def(py::init<PySparseMatrix&, const py::array_t<std::complex<double>>&,
                     double, int, bool>(),
             py::arg("matrix"),
             py::arg("rhs"),
             py::arg("tolerance") = 1e-10,
             py::arg("max_iterations") = 1000,
             py::arg("verbose") = false)
        .def("solve", &PySolver::solve)
        .def("get_current_solution", &PySolver::get_current_solution)
        .def("get_convergence_history", &PySolver::get_convergence_history);
}