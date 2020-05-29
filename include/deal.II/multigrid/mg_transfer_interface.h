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

#ifndef dealii_mg_transfer_interface_h
#define dealii_mg_transfer_interface_h

#include <deal.II/base/mg_level_object.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_base.h>

DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
namespace MGTransferUtil
{
  class Implementation;
}
#endif

/**
 * A multigrid transfer scheme. A multrigrid transfer class can have different
 * transfer schemes to enable p-adaptivity (one transfer scheme per
 * polynomial degree pair) and to enable global coarsening (one transfer
 * scheme for transfer between children and parent cells, as well as, one
 * transfer scheme for cells that are not refined).
 *
 * @note Normally, this data structure does not to be filled by users, but, one
 *   can use the can use the utility functions provided in the namespace
 *   MGTransferUtil to setup the Transfer operators.
 */
template <typename Number>
struct TransferScheme
{
  /**
   * Number of coarse cells.
   */
  unsigned int n_cells_coarse;

  /**
   * Number of degrees of freedom of a coarse cell.
   */
  unsigned int n_cell_dofs_coarse;

  /**
   * Number of degrees of freedom of fine cell.
   */
  unsigned int n_cell_dofs_fine;

  /**
   * Polynomial degree of the finite element of the coarse cells.
   */
  unsigned int degree_coarse;

  /**
   * Polynomial degree of the finite element of the fine cells.
   */
  unsigned int degree_fine;

  /**
   * Flag if the finite element on the fine cells are contiguous. If yes,
   * the multiplicity of DoF shares and constraints have to be taken in account
   * via weights.
   */
  bool fine_element_is_continuous;

  /**
   * Weights for continuous elements.
   */
  std::vector<Number> weights;

  /**
   * 1D prolongation matrix.
   */
  AlignedVector<VectorizedArray<Number>> prolongation_matrix_1d;

  /**
   * DoF indices of the coarse cells.
   */
  std::vector<unsigned int> level_dof_indices_coarse;

  /**
   * DoF indices of the children of the coarse cells.
   */
  std::vector<unsigned int> level_dof_indices_fine;

  /**
   * Print internal data structures to stream @p out.
   */
  template <typename Stream>
  void
  print(Stream &out) const
  {
    out << "weights:" << std::endl;
    for (const auto w : weights)
      out << w << " ";
    out << std::endl;

    out << "level_dof_indices_fine:" << std::endl;
    for (const auto w : level_dof_indices_fine)
      out << w << " ";
    out << std::endl;

    out << "level_dof_indices_coarse:" << std::endl;
    for (const auto w : level_dof_indices_coarse)
      out << w << " ";
    out << std::endl;

    out << "prolongation_matrix_1d:" << std::endl;
    for (const auto w : prolongation_matrix_1d)
      out << w[0] << " ";
    out << std::endl;
  }
};



/**
 * Class for transfer between two multigrid levels.
 */
template <int dim, typename Number>
class Transfer
{
public:
  /**
   * Print internal data structures to stream @p out.
   */
  template <typename Stream>
  void
  print_internal(Stream &out) const
  {
    for (const auto &scheme : schemes)
      scheme.print(out);
  }

  /**
   * Perform prolongation.
   */
  void
  prolongate(const unsigned int,
             LinearAlgebra::distributed::Vector<Number> &      dst,
             const LinearAlgebra::distributed::Vector<Number> &src) const;

  /**
   * Perform restriction.
   */
  void
  restrict_and_add(const unsigned int,
                   LinearAlgebra::distributed::Vector<Number> &      dst,
                   const LinearAlgebra::distributed::Vector<Number> &src) const;

private:
  /**
   * Transfer schemes.
   */
  std::vector<TransferScheme<Number>> schemes;

  /**
   * Partitioner needed by the intermediate vector.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine;

  /**
   * TODO: needed?
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_coarse;

  /**
   * Internal vector needed for collecting all degrees of freedom of the
   * children cells.
   */
  mutable LinearAlgebra::distributed::Vector<Number> vec_fine;

  /**
   * TODO: needed?
   */
  mutable LinearAlgebra::distributed::Vector<Number> vec_coarse;

  /**
   * Constraint matrix on coarse level.
   */
  AffineConstraints<Number> constraint_coarse;

  friend class MGTransferUtil::Implementation;
};



/**
 * Implementation of the MGTransferBase interface for which the transfer
 * operations is implemented in a matrix-free way based on the interpolation
 * matrices of the underlying finite element. This requires considerably less
 * memory than MGTransferPrebuilt and can also be considerably faster than that
 * variant. In contrast to MGTransferMatrixFree, the user can provide separate
 * transfer operators between each level.
 *
 * This class currently only works for tensor-product finite elements based on
 * FE_Q and FE_DGQ elements. Systems involving multiple components of
 * one of these element, as well as, systems with different elements or other
 * elements are currently not implemented.
 */
template <typename MatrixType>
class MGTransferMatrixFreeNew
  : public dealii::MGTransferBase<dealii::LinearAlgebra::distributed::Vector<
      typename MatrixType::value_type>>
{
public:
  /**
   * Dimension.
   */
  static const int dim = MatrixType::dim;

  /**
   * Value type.
   */
  using Number = typename MatrixType::value_type;

  /**
   * Constructor.
   */
  MGTransferMatrixFreeNew(const MGLevelObject<MatrixType> &           matrices,
                          const MGLevelObject<Transfer<dim, Number>> &transfer)
    : matrices(matrices)
    , transfer(transfer)
  {}

  /**
   * Perform prolongation.
   */
  void
  prolongate(
    const unsigned int                                        to_level,
    dealii::LinearAlgebra::distributed::Vector<Number> &      dst,
    const dealii::LinearAlgebra::distributed::Vector<Number> &src) const
  {
    this->transfer[to_level].prolongate(0 /*dummy*/, dst, src);
  }

  /**
   * Perform restriction.
   */
  virtual void
  restrict_and_add(
    const unsigned int                                        from_level,
    dealii::LinearAlgebra::distributed::Vector<Number> &      dst,
    const dealii::LinearAlgebra::distributed::Vector<Number> &src) const
  {
    this->transfer[from_level].restrict_and_add(0 /*dummy*/, dst, src);
  }

  /**
   * Initialize internal vectors and copy @p src vector to the finest
   * multigrid level.
   *
   * @note DoFHandler is not needed here, but is required by the interface.
   */
  template <class InVector, int spacedim>
  void
  copy_to_mg(
    const DoFHandler<dim, spacedim> &dof_handler,
    MGLevelObject<dealii::LinearAlgebra::distributed::Vector<Number>> &dst,
    const InVector &src) const
  {
    (void)dof_handler;

    for (unsigned int level = dst.min_level(); level <= dst.max_level();
         ++level)
      matrices[level].initialize_dof_vector(dst[level]);

    dst[dst.max_level()].copy_locally_owned_data_from(src);
    dst[dst.max_level()].update_ghost_values();
  }

  /**
   * Initialize internal vectors and copy the values on the finest
   * multigrid level to @p dst vector.
   *
   * @note DoFHandler is not needed here, but is required by the interface.
   */
  template <class OutVector, int spacedim>
  void
  copy_from_mg(
    const DoFHandler<dim, spacedim> &dof_handler,
    OutVector &                      dst,
    const MGLevelObject<dealii::LinearAlgebra::distributed::Vector<Number>>
      &src) const
  {
    (void)dof_handler;

    dst.copy_locally_owned_data_from(src[src.max_level()]);
    dst.update_ghost_values();
  }

private:
  const MGLevelObject<MatrixType> &           matrices;
  const MGLevelObject<Transfer<dim, Number>> &transfer;
};



/**
 * Class for repartitioning data living on the same triangulation, which
 * has been partitioned differently.
 */
template <int dim, typename Number>
class VectorRepartitioner
{
public:
  /**
   * Transfer data.
   */
  void
  update_forwards(LinearAlgebra::distributed::Vector<Number> &      dst,
                  const LinearAlgebra::distributed::Vector<Number> &src) const;

  /**
   * Transfer data back.
   */
  void
  update_backwards(LinearAlgebra::distributed::Vector<Number> &      dst,
                   const LinearAlgebra::distributed::Vector<Number> &src) const;

private:
  /**
   * Partitioner needed by an intermediate vector, which is needed for
   * collecting all degrees of freedom of the children cells.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> extended_partitioner;

  /**
   * Indices for copying the data from/to the intermediate vector.
   */
  std::vector<unsigned int> indices;

  friend class MGTransferUtil::Implementation;
};

DEAL_II_NAMESPACE_CLOSE

#endif
