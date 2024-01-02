// ---------------------------------------------------------------------
//
// Copyright (C) 1999 - 2022 by the deal.II authors
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

#ifndef dealii_solution_transfer_h
#define dealii_solution_transfer_h


/*----------------------------   solutiontransfer.h     ----------------------*/


#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN

template <int dim, typename VectorType = Vector<double>, int spacedim = dim>
using SolutionTransfer =
  parallel::distributed::SolutionTransfer<dim, VectorType, spacedim>;

DEAL_II_NAMESPACE_CLOSE

#endif
