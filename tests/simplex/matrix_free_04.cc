// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


// Solve Poisson problem and Helmholtz problem on a simplex mesh with
// continuous elements and compare results between matrix-free and matrix-based
// implementations.

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/grid_generator.h>
#include <deal.II/simplex/quadrature_lib.h>

#include "../tests.h"

using namespace dealii;


template <int dim>
class PoissonOperator
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<double>;
  using number     = double;

  PoissonOperator(const MatrixFree<dim, double> &matrix_free)
    : matrix_free(matrix_free)
  {}

  void
  initialize_dof_vector(VectorType &vec)
  {
    matrix_free.initialize_dof_vector(vec);
  }

  void
  rhs(VectorType &vec) const
  {
    const int dummy = 0;

    matrix_free.template cell_loop<VectorType, int>(
      [&](const auto &data, auto &dst, const auto &, const auto cells) {
        FEEvaluation<dim, -1, 0, 1, double> phi(data);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_value(1.0, q);

            phi.integrate_scatter(true, false, dst);
          }
      },
      vec,
      dummy,
      true);
  }

  const int fe_degree = 5; /*TODO*/


  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.template loop<VectorType, VectorType>(
      [&](const auto &data, auto &dst, const auto &src, const auto cell_range) {
        FEEvaluation<dim, -1, 0, 1, number> phi(data);

        for (unsigned int cell = cell_range.first; cell < cell_range.second;
             ++cell)
          {
            phi.reinit(cell);
            phi.read_dof_values(src);
            phi.evaluate(EvaluationFlags::gradients);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_gradient(phi.get_gradient(q), q);
            phi.integrate(EvaluationFlags::gradients);
            phi.set_dof_values(dst);
          }
      },
      [&](const auto &data, auto &dst, const auto &src, const auto face_range) {
        FEFaceEvaluation<dim, -1, 0, 1, number> fe_eval(data, true);
        FEFaceEvaluation<dim, -1, 0, 1, number> fe_eval_neighbor(data, false);

        for (unsigned int face = face_range.first; face < face_range.second;
             face++)
          {
            fe_eval.reinit(face);
            fe_eval_neighbor.reinit(face);

            fe_eval.read_dof_values(src);
            fe_eval.evaluate(EvaluationFlags::values |
                             EvaluationFlags::gradients);
            fe_eval_neighbor.read_dof_values(src);
            fe_eval_neighbor.evaluate(EvaluationFlags::values |
                                      EvaluationFlags::gradients);
            VectorizedArray<number> sigmaF = 10.0;
            //  (std::abs((fe_eval.get_normal_vector(0) *
            //             fe_eval.inverse_jacobian(0))[dim - 1]) +
            //   std::abs((fe_eval.get_normal_vector(0) *
            //             fe_eval_neighbor.inverse_jacobian(0))[dim - 1])) *
            //  (number)(std::max(fe_degree, 1) * (fe_degree + 1.0));

            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              {
                VectorizedArray<number> average_value =
                  (fe_eval.get_value(q) - fe_eval_neighbor.get_value(q)) * 0.5;
                VectorizedArray<number> average_valgrad =
                  fe_eval.get_normal_derivative(q) +
                  fe_eval_neighbor.get_normal_derivative(q);
                average_valgrad =
                  average_value * 2. * sigmaF - average_valgrad * 0.5;
                fe_eval.submit_normal_derivative(-average_value, q);
                fe_eval_neighbor.submit_normal_derivative(-average_value, q);
                fe_eval.submit_value(average_valgrad, q);
                fe_eval_neighbor.submit_value(-average_valgrad, q);
              }
            fe_eval.integrate(EvaluationFlags::values |
                              EvaluationFlags::gradients);
            fe_eval.distribute_local_to_global(dst);
            fe_eval_neighbor.integrate(EvaluationFlags::values |
                                       EvaluationFlags::gradients);
            fe_eval_neighbor.distribute_local_to_global(dst);
          }
      },
      [&](const auto &data, auto &dst, const auto &src, const auto face_range) {
        FEFaceEvaluation<dim, -1, 0, 1, number> fe_eval(data, true);
        for (unsigned int face = face_range.first; face < face_range.second;
             face++)
          {
            fe_eval.reinit(face);
            fe_eval.read_dof_values(src);
            fe_eval.evaluate(EvaluationFlags::values |
                             EvaluationFlags::gradients);
            VectorizedArray<number> sigmaF = 10.0;
            //  std::abs((fe_eval.get_normal_vector(0) *
            //            fe_eval.inverse_jacobian(0))[dim - 1]) *
            //  number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              {
                VectorizedArray<number> average_value = fe_eval.get_value(q);
                VectorizedArray<number> average_valgrad =
                  -fe_eval.get_normal_derivative(q);
                average_valgrad += average_value * sigmaF * 2.0;
                fe_eval.submit_normal_derivative(-average_value, q);
                fe_eval.submit_value(average_valgrad, q);
              }

            fe_eval.integrate(EvaluationFlags::values |
                              EvaluationFlags::gradients);
            fe_eval.distribute_local_to_global(dst);
          }
      },
      dst,
      src);
  }

private:
  const MatrixFree<dim, double> &matrix_free;
};

template <int dim>
void
test(const unsigned int degree)
{
  Triangulation<dim> tria;

#if true
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, dim == 2 ? 16 : 8);

  Simplex::FE_DGP<dim> fe(degree);
  Simplex::QGauss<dim> quad(degree + 1);
  MappingFE<dim>       mapping(Simplex::FE_P<dim>(1));
#else
  GridGenerator::subdivided_hyper_cube(tria, dim == 2 ? 16 : 8);

  FE_DGQ<dim>    fe(degree);
  QGauss<dim>    quad(degree + 1);
  MappingFE<dim> mapping(FE_Q<dim>(1));

#endif

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  constraints.close();

  const auto solve_and_postprocess =
    [&](const auto &poisson_operator,
        auto &      x,
        auto &      b) -> std::pair<unsigned int, double> {
    ReductionControl reduction_control(1000);
    SolverCG<typename std::remove_reference<decltype(x)>::type> solver(
      reduction_control);

    try
      {
        solver.solve(poisson_operator, x, b, PreconditionIdentity());
      }
    catch (const std::exception &e)
      {
        deallog << e.what() << std::endl;
      }

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      printf("Solved in %d iterations.\n", reduction_control.last_step());

    constraints.distribute(x);

#if 1
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    x.update_ghost_values();
    data_out.add_data_vector(dof_handler, x, "solution");
    data_out.build_patches(mapping, 2);
    data_out.write_vtu_with_pvtu_record("./", "result", 0, MPI_COMM_WORLD);
#endif

    return {reduction_control.last_step(), reduction_control.last_value()};
  };

  const auto mf_algo = [&]() {
    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags = update_gradients | update_values;
    additional_data.mapping_update_flags_inner_faces =
      update_gradients | update_values;
    additional_data.mapping_update_flags_boundary_faces =
      update_gradients | update_values;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, double>::AdditionalData::none;

    MatrixFree<dim, double> matrix_free;
    matrix_free.reinit(
      mapping, dof_handler, constraints, quad, additional_data);

    PoissonOperator<dim> poisson_operator(matrix_free);

    LinearAlgebra::distributed::Vector<double> x, b;
    poisson_operator.initialize_dof_vector(x);
    poisson_operator.initialize_dof_vector(b);

    poisson_operator.rhs(b);

    return solve_and_postprocess(poisson_operator, x, b);
  };

  mf_algo();
}


int
main(int argc, char **argv)
{
  initlog();

  // deallog.depth_file(1);

  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<2>(/*degree=*/2);
  test<3>(/*degree=*/2);
}
