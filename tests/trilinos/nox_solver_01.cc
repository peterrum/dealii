// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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



// Check TrilinosWrappers::NOXSolver.

#include <deal.II/base/mpi.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/trilinos/nox.h>

#include "../tests.h"

int
main(int argc, char **argv)
{
  initlog();

  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  using Number     = double;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  // set up solver control
  const unsigned int n_max_iterations = 100;
  const double       abs_tolerance    = 1e-9;
  const double       rel_tolerance    = 1e-5;

  TrilinosWrappers::AdditionalData statistics(n_max_iterations,
                                              abs_tolerance,
                                              rel_tolerance);

  // set up parameters
  Teuchos::RCP<Teuchos::ParameterList> non_linear_parameters =
    Teuchos::rcp(new Teuchos::ParameterList);

  non_linear_parameters->set("Nonlinear Solver", "Line Search Based");

  auto &printParams = non_linear_parameters->sublist("Printing");
  printParams.set("Output Information", 0);

  auto &dir_parameters = non_linear_parameters->sublist("Direction");
  dir_parameters.set("Method", "Newton");

  auto &search_parameters = non_linear_parameters->sublist("Line Search");
  search_parameters.set("Method", "Polynomial");

  // set up solver
  TrilinosWrappers::NOXSolver<VectorType> solver(statistics,
                                                 non_linear_parameters);

  // ... helper functions
  double J = 0.0;

  solver.residual = [](const auto &src, auto &dst) {
    // compute residual
    dst[0] = src[0] * src[0];
    return 0;
  };

  solver.setup_jacobian = [&](const auto &src) {
    // compute Jacobian
    J = 2.0 * src[0];
    return 0;
  };

  solver.solve_with_jacobian = [&](const auto &src, auto &dst) {
    // solve with Jacobian
    dst[0] = src[0] / J;
    return 0;
  };

  // initial guess
  VectorType solution(1);
  solution[0] = 2.0;

  // solve with the given initial guess
  solver.solve(solution);

  deallog << "The solution is: " << solution[0] << std::endl;
}
