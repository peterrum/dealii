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

#ifndef dealii_mg_transfer_interface_h
#define dealii_mg_transfer_interface_h

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_base.h>

DEAL_II_NAMESPACE_OPEN

template <typename Number>
struct TransferScheme
{
  unsigned int n_cells_coarse;

  unsigned int n_cell_dofs_coarse;
  unsigned int n_cell_dofs_fine;

  // cached values
  unsigned int degree_coarse;
  unsigned int degree_fine;

  bool                fine_element_is_continuous;
  std::vector<Number> weights;

  AlignedVector<VectorizedArray<Number>> prolongation_matrix_1d;

  std::vector<unsigned int> level_dof_indices_coarse;
  std::vector<unsigned int> level_dof_indices_fine;

  void
  print(std::ostream &out) const;
};

template <int dim, typename Number>
class Transfer
  : public MGTransferBase<LinearAlgebra::distributed::Vector<Number>>
{
public:
  void
  print_internal(std::ostream &out) const;

  void
  prolongate(
    const unsigned int                                to_level,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const override;

  void
  restrict_and_add(
    const unsigned int                                from_level,
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const override;

  std::vector<TransferScheme<Number>> schemes;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine;
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_coarse;

  mutable LinearAlgebra::distributed::Vector<Number> vec_fine;
  mutable LinearAlgebra::distributed::Vector<Number> vec_coarse;

  AffineConstraints<Number> constraint_coarse;
};

template <int dim, typename Number>
class VectorRepartitioner
{
public:
  void
  update_forwards(LinearAlgebra::distributed::Vector<Number> &      dst,
                  const LinearAlgebra::distributed::Vector<Number> &src) const;

  void
  update_backwards(LinearAlgebra::distributed::Vector<Number> &      dst,
                   const LinearAlgebra::distributed::Vector<Number> &src) const;

  std::shared_ptr<const Utilities::MPI::Partitioner> extended_partitioner;
  std::vector<unsigned int>                          indices;
};

DEAL_II_NAMESPACE_CLOSE

#endif
