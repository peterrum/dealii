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


// Test different triangulation iterators.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/tet/data_out.h>
#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/grid_generator.h>
#include <deal.II/tet/mapping_q.h>
#include <deal.II/tet/quadrature_lib.h>

#include "../tests.h"

using namespace dealii;

void
test_3(const std::vector<unsigned int> indices)
{
  const int dim      = 3;
  const int spacedim = 3;

  Triangulation<dim, spacedim> tria;
  Tet::GridGenerator::subdivided_hyper_rectangle(tria,
                                                 {1, 1, 1},
                                                 {0.0, 0.0, 0.0},
                                                 {1.0, 1.0, 1.0});

  DoFHandler<dim, spacedim> dof_handler(tria);

  Tet::FE_Q<dim> fe(2);
  dof_handler.distribute_dofs(fe);

  LinearAlgebra::distributed::Vector<double> vector(dof_handler.n_dofs());

  std::ofstream output("cell_tet.vtk");
  Tet::data_out(dof_handler, vector, "data", output);
}

int
main()
{
  initlog();

  test_3({0, 1, 2, 3});
  //  test_3({0, 1, 3, 2});
  //  test_3({0, 2, 1, 3});
  //  test_3({0, 2, 3, 1});
  //  test_3({0, 3, 1, 2});
  //  test_3({0, 3, 2, 1});
  //  deallog << std::endl;
  //
  //  test_3({1, 0, 2, 3});
  //  test_3({1, 0, 3, 2});
  //  test_3({1, 2, 0, 3});
  //  test_3({1, 2, 3, 0});
  //  test_3({1, 3, 0, 2});
  //  test_3({1, 3, 2, 0});
  //  deallog << std::endl;
  //
  //  test_3({2, 1, 0, 3});
  //  test_3({2, 1, 3, 0});
  //  test_3({2, 0, 1, 3});
  //  test_3({2, 0, 3, 1});
  //  test_3({2, 3, 1, 0});
  //  test_3({2, 3, 0, 1});
  //  deallog << std::endl;
  //
  //  test_3({3, 1, 2, 0});
  //  test_3({3, 1, 0, 2});
  //  test_3({3, 2, 1, 0});
  //  test_3({3, 2, 0, 1});
  //  test_3({3, 0, 1, 2});
  //  test_3({3, 0, 2, 1});
  //  deallog << std::endl;
}
