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
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/grid_generator.h>
#include <deal.II/simplex/quadrature_lib.h>

#include "../tests.h"

#include "./simplex_grids.h"

using namespace dealii;


template <int dim>
class PoissonOperator
{
public:
  using VectorType = Vector<double>;

  PoissonOperator(const MatrixFree<dim, double> &matrix_free,
                  const bool                     do_helmholtz)
    : matrix_free(matrix_free)
    , do_helmholtz(do_helmholtz)
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
            phi.gather_evaluate(src, do_helmholtz, true);

            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                if (do_helmholtz)
                  phi.submit_value(phi.get_value(q), q);

                phi.submit_gradient(phi.get_gradient(q), q);
              }

            phi.integrate_scatter(do_helmholtz, true, dst);
          }
      },
      dst,
      src,
      true);
  }

private:
  const MatrixFree<dim, double> &matrix_free;
  const bool                     do_helmholtz;
};

template <int dim>
void
test()
{
  const unsigned int degree = 1;

  Triangulation<dim> tria;

  std::shared_ptr<FiniteElement<dim>> fe;
  std::shared_ptr<Quadrature<dim>>    quad;
  std::shared_ptr<FiniteElement<dim>> fe_mapping;

  GridGenerator::subdivided_hyper_cube_with_wedges(tria, dim == 2 ? 16 : 8);
  fe         = std::make_shared<Simplex::FE_WedgeP<dim>>(degree);
  quad       = std::make_shared<Simplex::QGaussWedge<dim>>(degree + 1);
  fe_mapping = std::make_shared<Simplex::FE_WedgeP<dim>>(1);

  MappingFE<dim> mapping(*fe_mapping);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(*fe);

  AffineConstraints<double> constraints;


  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags = update_gradients | update_values;
  additional_data.mapping_update_flags_boundary_faces =
    update_gradients | update_values;

  MatrixFree<dim, double> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, *quad, additional_data);
}


int
main(int argc, char **argv)
{
  initlog();

  deallog.depth_file(2);

  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<3>();
}
