// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2022 by the deal.II authors
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

#include "derivatives.h"

#define PRECISION 8

/*
 * Tests the values of derivatives of an Hermite basis at the boundaries
 * of an element in 1D. This is important to ensure that the basis is
 * behaving as expected, and will not accidentally enforce discontinuities
 * in derivatives across element boundaries.
 */


template <int dim>
void
print_hermite_endpoint_derivatives()
{
  MappingCartesian<dim> m;

  FE_Hermite<dim> herm0(0);
  plot_function_derivatives<dim>(m, herm0, "Hermite-0");

  FE_Hermite<dim> herm1(1);
  plot_function_derivatives<dim>(m, herm1, "Hermite-1");

  // Skip the following for dim 3 or greater
  if (dim < 3)
    {
      FE_Hermite<dim> herm2(2);
      plot_function_derivatives<dim>(m, herm2, "Hermite-2");
    }
  if (dim == 1)
    {
      FE_Hermite<dim> herm3(3);
      plot_function_derivatives<dim>(m, herm3, "Hermite-3");

      FE_Hermite<dim> herm4(4);
      plot_function_derivatives<dim>(m, herm4, "Hermite-4");
    }
}



int
main()
{
  std::ofstream logfile("output");

  deallog << std::setprecision(PRECISION) << std::fixed;
  deallog.attach(logfile);

  print_hermite_endpoint_derivatives<1>();
  print_hermite_endpoint_derivatives<2>();
  print_hermite_endpoint_derivatives<3>();

  return 0;
}
