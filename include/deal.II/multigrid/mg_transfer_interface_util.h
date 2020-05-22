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

#ifndef dealii_mg_transfer_interface_util_h
#define dealii_mg_transfer_interface_util_h

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_transfer_interface.h>

DEAL_II_NAMESPACE_OPEN

namespace MGTransferUtil
{
  bool
  polynomial_transfer_supported(const unsigned int fe_degree_fine,
                                const unsigned int fe_degree_coarse);

  template <int dim, typename Number, typename MeshType>
  void
  setup_global_coarsening_transfer(
    const MeshType &                 dof_handler_fine,
    const MeshType &                 dof_handler_coarse,
    const AffineConstraints<Number> &constraint_fine,
    const AffineConstraints<Number> &constraint_coarse,
    Transfer<dim, Number> &          transfer);


  template <int dim, typename Number, typename MeshType>
  void
  setup_polynomial_transfer(const MeshType &                 dof_handler_fine,
                            const MeshType &                 dof_handler_coarse,
                            const AffineConstraints<Number> &constraint_fine,
                            const AffineConstraints<Number> &constraint_coarse,
                            Transfer<dim, Number> &          transfer);

  template <int dim, typename Number, typename MeshType>
  void
  setup_vector_repartitioner(const MeshType &dof_handler_fine,
                             const MeshType &dof_handler_coarse,
                             VectorRepartitioner<dim, Number> &transfer);

} // namespace MGTransferUtil

DEAL_II_NAMESPACE_CLOSE

#endif
