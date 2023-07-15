// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2023 by the deal.II authors
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

#ifndef dealii_mg_transfer_matrix_free_h
#define dealii_mg_transfer_matrix_free_h

#include <deal.II/base/config.h>

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/mg_transfer_internal.h>


DEAL_II_NAMESPACE_OPEN


template <int dim, typename Number>
using MGTransferMatrixFree =
  MGTransferGlobalCoarsening<dim, LinearAlgebra::distributed::Vector<Number>>;



template <int dim, typename Number>
using MGTransferBlockMatrixFree =
  MGTransferBlockGlobalCoarsening<dim,
                                  LinearAlgebra::distributed::Vector<Number>>;


DEAL_II_NAMESPACE_CLOSE

#endif
