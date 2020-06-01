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

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/tet/grid_generator.h>

#include "../tests.h"

using namespace dealii;

void
test_2()
{
  const int dim = 2;

  Triangulation<dim> tria;
  Tet::GridGenerator::subdivided_hyper_rectangle(tria,
                                                 {1, 1},
                                                 {0.0, 0.0},
                                                 {1.0, 1.0});

  std::ofstream output_file("mesh2D.vtk");
  GridOut().write_vtk(tria, output_file);
}

void
test_3()
{
  const int dim = 3;

  Triangulation<dim> tria;
  Tet::GridGenerator::subdivided_hyper_rectangle(tria,
                                                 {1, 1, 1},
                                                 {0.0, 0.0, 0.0},
                                                 {1.0, 1.0, 1.0});

  std::ofstream output_file("mesh3D.vtk");
  GridOut().write_vtk(tria, output_file);
}

int
main()
{
  initlog();

  {
    deallog.push("2D");
    test_2();
    deallog.pop();
  }

  {
    deallog.push("3D");
    test_3();
    deallog.pop();
  }
}