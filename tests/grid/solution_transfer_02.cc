// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2021 by the deal.II authors
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



// Test distributed solution transfer with fullydistributed triangulations.

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/vector_tools.h>

#include "./tests.h"

using namespace dealii;

template <int dim>
class InterpolationFunction : public Function<dim>
{
public:
  InterpolationFunction()
    : Function<dim>(1)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const
  {
    return p[0];
  }
};

template <int dim, typename TriangulationType>
void
test(TriangulationType &triangulation)
{
  const FE_Q<dim> fe(2);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  using VectorType = Vector<double>;

  VectorType vector(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, InterpolationFunction<dim>(), vector);
  // vector = 1.0;

  parallel::distributed::SolutionTransfer<dim, VectorType> solution_transfer(
    dof_handler);

  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(vector);

  triangulation.refine_global(1);
  dof_handler.distribute_dofs(fe);

  vector.reinit(dof_handler.n_dofs());
  solution_transfer.interpolate(vector);

  vector.print(std::cout);

  VectorType error(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, InterpolationFunction<dim>(), error);

  error -= vector;

  deallog << (error.linfty_norm() < 1e-16 ? "PASSED" : "FAILED") << std::endl;
}


int
main(int argc, char **argv)
{
  initlog();

  deallog.push("2d");
  {
    constexpr int dim = 2;

    Triangulation<dim> triangulation;
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(3);

    test<dim>(triangulation);
  }
  deallog.pop();

  deallog.push("3d");
  {
    constexpr int dim = 3;

    Triangulation<dim> triangulation;
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(3);

    test<dim>(triangulation);
  }
  deallog.pop();
}
