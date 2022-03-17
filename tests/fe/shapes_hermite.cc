// ---------------------------------------------------------------------
//
// Copyright (C) 2013 - 2022 by the deal.II authors
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

#include <deal.II/fe/fe_hermite.h>
#include <deal.II/fe/mapping_cartesian.h>

#include <string>

#include "../tests.h"

#include "shapes.h"

#define PRECISION 8

/*
 * Test file to check that the shape values of an Hermite polynomial basis
 * of a given regularity are correct. Values of derivatives at the boundaries
 * are checked elsehwere in derivatives_hermite.cc.
 */

template <int dim>
void
plot_FE_Hermite_shape_functions()
{
  MappingCartesian<dim> m;

  FE_Hermite<dim> herm0(0);
  plot_shape_functions(m, herm0, "Hermite-0");
  plot_face_shape_functions(m, herm0, "Hermite-0");
  test_compute_functions(m, herm0, "Hermite-0");

  FE_Hermite<dim> herm1(1);
  plot_shape_functions(m, herm1, "Hermite-1");
  plot_face_shape_functions(m, herm1, "Hermite-1");
  test_compute_functions(m, herm1, "Hermite-1");

  // skip the following tests to
  // reduce run-time
  if (dim < 3)
    {
      FE_Hermite<dim> herm2(2);
      plot_shape_functions(m, herm2, "Hermite-2");
      plot_face_shape_functions(m, herm2, "Hermite-2");
      test_compute_functions(m, herm2, "Hermite-2");
    }

  if (dim == 1)
    {
      FE_Hermite<dim> herm3(3);
      plot_shape_functions(m, herm3, "Hermite-3");
      plot_face_shape_functions(m, herm3, "Hermite-3");
      test_compute_functions(m, herm3, "Hermite-3");

      FE_Hermite<dim> herm4(4);
      plot_shape_functions(m, herm4, "Hermite-4");
      plot_face_shape_functions(m, herm4, "Hermite-4");
      test_compute_functions(m, herm4, "Hermite-4");
    };
}



int
main()
{
  std::ofstream logfile("output");
  deallog << std::setprecision(PRECISION) << std::fixed;
  deallog.attach(logfile);

  plot_FE_Hermite_shape_functions<1>();
  plot_FE_Hermite_shape_functions<2>();
  plot_FE_Hermite_shape_functions<3>();

  return 0;
}
