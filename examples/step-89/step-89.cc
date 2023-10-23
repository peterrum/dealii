/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 *
 * Authors: Johannes Heinz, TU Wien, 2023
 *          Peter Munch, University of Augsburg, 2023
 */

// @sect3{Include files}
//
// The program starts with including all the relevant header files.
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>

// The following header file provides the class FERemoteEvaluation, which allows
// to access values and/or gradients at remote triangulations similar to FEEvaluation.
#include <deal.II/matrix_free/fe_remote_evaluation.h>

// We pack everything that is specific for this program into a namespace
// of its own.

namespace Step89
{
  using namespace dealii;

  // @sect3{Point-to-point interpolation}
  //
  // Description
  void point_to_point_interpolation()
  {
    constexpr unsigned int dim       = 2;
    constexpr unsigned int fe_degree = 3;

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);
  }


  // @sect3{Nitsche-type mortaring}
  //
  // Description
  void nitsche_type_mortaring()
  {

  }
  
} // namespace Step87


// @sect3{Driver}
//
// Finally, the driver executes the different versions of handling non-matching interfaces.

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  std::cout.precision(5);

  Step87::point_to_point_interpolation();
  Step87::nitsche_type_mortaring();
    
  return 0;
}
