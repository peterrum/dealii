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

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

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
      [&](const auto &, auto &dst, const auto &, const auto cells) {
        FEEvaluation<dim, -1, 0, 1, double> phi(matrix_free);
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


  void
  vmult(VectorType &dst, const VectorType &src) const
  {
    matrix_free.template cell_loop<VectorType, VectorType>(
      [&](const auto &, auto &dst, const auto &src, const auto cells) {
        FEEvaluation<dim, -1, 0, 1, double> phi(matrix_free);
        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, false, true);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_gradient(phi.get_gradient(q), q);

            phi.integrate_scatter(false, true, dst);
          }
      },
      dst,
      src,
      true);
  }

private:
  const MatrixFree<dim, double> &matrix_free;
};


int
main(int argc, char **argv)
{
  initlog();

  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim    = 2; // only 2D is working for now
  const unsigned int degree = 2;

  // create mesh, select relevant FEM ingredients, and set up DoFHandler
  Triangulation<dim> tria;

  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 16);

  Simplex::FE_P<dim>   fe(degree);
  Simplex::QGauss<dim> quad(degree + 1);
  MappingFE<dim>       mapping(Simplex::FE_P<dim>(1));

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Create constraint matrix
  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  // initialize MatrixFree
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags = update_gradients | update_values;

  MatrixFree<dim, double> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quad, additional_data);

  // create operator
  PoissonOperator<dim> poisson_operator(matrix_free);

  // initialize vectors
  LinearAlgebra::distributed::Vector<double> x, b;
  poisson_operator.initialize_dof_vector(x);
  poisson_operator.initialize_dof_vector(b);

  poisson_operator.rhs(b);

  // solve linear equation system
  ReductionControl                                     reduction_control;
  SolverCG<LinearAlgebra::distributed::Vector<double>> solver(
    reduction_control);
  solver.solve(poisson_operator, x, b, PreconditionIdentity());

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    printf("Solved in %d iterations.\n", reduction_control.last_step());

  constraints.distribute(x);

#if false
  // output results
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  x.update_ghost_values();
  data_out.add_data_vector(dof_handler, x, "solution");
  data_out.build_patches(mapping, 2);
  data_out.write_vtu_with_pvtu_record("./", "result", 0, MPI_COMM_WORLD);
#endif
}
