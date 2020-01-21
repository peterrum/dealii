// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
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

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/vector_access_internal.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

using namespace dealii;

template <int dim,
          int fe_degree,
          int n_points                 = fe_degree + 1,
          int n_components             = 1,
          typename Number              = double,
          typename VectorizedArrayType = VectorizedArray<Number>>
void
test()
{
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  GridGenerator::hyper_cube(tria, -numbers::PI / 2, numbers::PI / 2);

  tria.refine_global();
  for (auto &cell : tria.active_cell_iterators())
    if (cell->active() && cell->is_locally_owned() && cell->center()[0] < 0.0)
      cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  const FE_Q<dim>     fe_q(fe_degree);
  const FESystem<dim> fe(fe_q, n_components);

  // setup dof-handlers
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<Number> constraint;
  DoFTools::make_hanging_node_constraints(dof_handler, constraint);

  VectorTools::interpolate_boundary_values(
    dof_handler, 0, Functions::ZeroFunction<dim>(n_components), constraint);

  constraint.close();

  typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData
    additional_data;
  additional_data.mapping_update_flags = update_values | update_gradients;

  MappingQ<dim> mapping(1);
  QGauss<1>     quad(fe_degree + 1);

  MatrixFree<dim, Number, VectorizedArrayType> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraint, quad, additional_data);

  LinearAlgebra::distributed::Vector<Number> temp;
  matrix_free.initialize_dof_vector(temp);
  temp = 1.0;

  matrix_free.template cell_loop<LinearAlgebra::distributed::Vector<Number>,
                                 LinearAlgebra::distributed::Vector<Number>>(
    [](const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
       LinearAlgebra::distributed::Vector<Number> &,
       const LinearAlgebra::distributed::Vector<Number> &temp,
       const std::pair<unsigned int, unsigned int> &     range) mutable {
      FEEvaluation<dim,
                   fe_degree,
                   n_points,
                   n_components,
                   Number,
                   VectorizedArrayType>
        phi(matrix_free);
      for (unsigned int cell = range.first; cell < range.second; cell++)
        {
          phi.reinit(cell);

          phi.gather_evaluate_plain(temp, true, false);
          for (unsigned int q = 0; q < phi.n_q_points; q++)
            deallog << phi.get_value(q) << std::endl;

          phi.gather_evaluate(temp, true, false);
          for (unsigned int q = 0; q < phi.n_q_points; q++)
            deallog << phi.get_value(q) << std::endl;

          deallog << std::endl;
        }
    },
    temp,
    temp,
    false);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  test<2, 1, 2, 1>(); // scalar
}