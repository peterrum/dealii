// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2023 by the deal.II authors
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

#ifndef dealii_mg_transfer_global_coarsening_h
#define dealii_mg_transfer_global_coarsening_h

#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/constraint_info.h>
#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <deal.II/non_matching/mapping_info.h>



DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
namespace internal
{
  class MGTwoLevelTransferImplementation;
}

namespace RepartitioningPolicyTools
{
  template <int dim, int spacedim>
  class Base;
}
#endif



/**
 * Global coarsening utility functions.
 */
namespace MGTransferGlobalCoarseningTools
{
  /**
   * Common polynomial coarsening sequences.
   *
   * @note These polynomial coarsening sequences up to a degree of 9 are
   *   precompiled in MGTwoLevelTransfer. See also:
   *   MGTwoLevelTransfer::fast_polynomial_transfer_supported()
   */
  enum class PolynomialCoarseningSequenceType
  {
    /**
     * Half polynomial degree by integer division. For example, for degree=7
     * the following sequence would be obtained:: 7 -> 3 -> 1
     */
    bisect,
    /**
     * Decrease the polynomial degree by one. E.g., for degree=7 following
     * sequence would result: 7 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1
     */
    decrease_by_one,
    /**
     * Decrease the polynomial degree to one. E.g., for degree=7 following
     * sequence would result: 7 -> 1
     */
    go_to_one
  };

  /**
   * For a given @p degree and polynomial coarsening sequence @p p_sequence,
   * determine the next coarser degree.
   */
  unsigned int
  create_next_polynomial_coarsening_degree(
    const unsigned int                      degree,
    const PolynomialCoarseningSequenceType &p_sequence);

  /**
   * For a given @p max_degree and polynomial coarsening sequence @p p_sequence,
   * determine the full sequence of polynomial degrees, sorted in ascending
   * order.
   */
  std::vector<unsigned int>
  create_polynomial_coarsening_sequence(
    const unsigned int                      max_degree,
    const PolynomialCoarseningSequenceType &p_sequence);

  /**
   * For a given triangulation @p tria, determine the geometric coarsening
   * sequence by repeated global coarsening of the provided triangulation.
   *
   * @note For convenience, a reference to the input triangulation is stored in
   *   the last entry of the return vector.
   * @note Currently, not implemented for parallel::fullydistributed::Triangulation.
   * @note The type of the returned triangulations is the same as of the input
   *   triangulation.
   */
  template <int dim, int spacedim>
  std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
  create_geometric_coarsening_sequence(
    const Triangulation<dim, spacedim> &tria);

  /**
   * Similar to the above function but also taking a @p policy for
   * repartitioning the triangulations on the coarser levels. If
   * @p preserve_fine_triangulation is set, the input triangulation is not
   * altered,
   * else the triangulation is coarsened. If @p repartition_fine_triangulation
   * is set, the triangulation on the finest level is repartitioned as well. If
   * the flags are set to true/false, the input triangulation is simply used as
   * the finest triangulation.
   *
   * @note For convenience, a reference to the input triangulation is stored in
   *   the last entry of the return vector.
   * @note The type of the returned triangulations is
   *   parallel::fullydistributed::Triangulation.
   * @note Currently, only implemented for parallel::distributed::Triangulation.
   */
  template <int dim, int spacedim>
  std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
  create_geometric_coarsening_sequence(
    Triangulation<dim, spacedim> &                        tria,
    const RepartitioningPolicyTools::Base<dim, spacedim> &policy,
    const bool preserve_fine_triangulation,
    const bool repartition_fine_triangulation);

  /**
   * Similar to the above function but taking in a constant version of
   * @p tria. As a consequence, it can not be used for coarsening directly,
   * so a temporary copy will be created internally.
   */
  template <int dim, int spacedim>
  std::vector<std::shared_ptr<const Triangulation<dim, spacedim>>>
  create_geometric_coarsening_sequence(
    const Triangulation<dim, spacedim> &                  tria,
    const RepartitioningPolicyTools::Base<dim, spacedim> &policy,
    const bool repartition_fine_triangulation = false);

} // namespace MGTransferGlobalCoarseningTools


/**
 * Abstract base class for transfer operators between two multigrid levels.
 */
template <typename VectorType>
class MGTwoLevelTransferBase : public Subscriptor
{
public:
  /**
   * Perform prolongation.
   */
  virtual void
  prolongate_and_add(VectorType &dst, const VectorType &src) const = 0;

  /**
   * Perform restriction.
   */
  virtual void
  restrict_and_add(VectorType &dst, const VectorType &src) const = 0;

  /**
   * Perform interpolation of a solution vector from the fine level to the
   * coarse level. This function is different from restriction, where a
   * weighted residual is transferred to a coarser level (transposition of
   * prolongation matrix).
   */
  virtual void
  interpolate(VectorType &dst, const VectorType &src) const = 0;

  /**
   * Enable inplace vector operations if external and internal vectors
   * are compatible.
   */
  virtual void
  enable_inplace_operations_if_possible(
    const std::shared_ptr<const Utilities::MPI::Partitioner>
      &partitioner_coarse,
    const std::shared_ptr<const Utilities::MPI::Partitioner>
      &partitioner_fine) = 0;

  /**
   * Return the memory consumption of the allocated memory in this class.
   */
  virtual std::size_t
  memory_consumption() const = 0;
};


/**
 * Base class for transfer operators between two multigrid levels.
 * Specialization for LinearAlgebra::distributed::Vector. The implementation of
 * restriction and prolongation between levels is delegated to derived classes,
 * which implement prolongate_and_add_internal() and restrict_and_add_internal()
 * accordingly.
 */
template <typename Number>
class MGTwoLevelTransferBase<LinearAlgebra::distributed::Vector<Number>>
  : public Subscriptor
{
public:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  /**
   * Perform prolongation.
   */
  virtual void
  prolongate_and_add(VectorType &dst, const VectorType &src) const;

  /**
   * Perform restriction.
   */
  virtual void
  restrict_and_add(VectorType &dst, const VectorType &src) const;

  /**
   * Perform interpolation of a solution vector from the fine level to the
   * coarse level.
   */
  virtual void
  interpolate(VectorType &dst, const VectorType &src) const = 0;

  /**
   * Enable inplace vector operations if external and internal vectors
   * are compatible.
   */
  virtual void
  enable_inplace_operations_if_possible(
    const std::shared_ptr<const Utilities::MPI::Partitioner>
      &partitioner_coarse,
    const std::shared_ptr<const Utilities::MPI::Partitioner>
      &partitioner_fine) = 0;

  /**
   * Return the memory consumption of the allocated memory in this class.
   */
  virtual std::size_t
  memory_consumption() const = 0;

protected:
  /**
   * Perform prolongation on vectors with correct ghosting.
   */
  virtual void
  prolongate_and_add_internal(
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const = 0;

  /**
   * Perform restriction on vectors with correct ghosting.
   */
  virtual void
  restrict_and_add_internal(
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const = 0;

  /**
   * A wrapper around update_ghost_values() optimized in case the
   * present vector has the same parallel layout of one of the external
   * partitioners.
   */
  void
  update_ghost_values(
    const LinearAlgebra::distributed::Vector<Number> &vec) const;

  /**
   * A wrapper around compress() optimized in case the
   * present vector has the same parallel layout of one of the external
   * partitioners.
   */
  void
  compress(LinearAlgebra::distributed::Vector<Number> &vec,
           const VectorOperation::values               op) const;

  /**
   * A wrapper around zero_out_ghost_values() optimized in case the
   * present vector has the same parallel layout of one of the external
   * partitioners.
   */
  void
  zero_out_ghost_values(
    const LinearAlgebra::distributed::Vector<Number> &vec) const;

  /**
   * Enable inplace vector operations if external and internal vectors
   * are compatible.
   */
  template <int dim, std::size_t width>
  void
  internal_enable_inplace_operations_if_possible(
    const std::shared_ptr<const Utilities::MPI::Partitioner>
      &partitioner_coarse,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner_fine,
    internal::MatrixFreeFunctions::ConstraintInfo<
      dim,
      VectorizedArray<Number, width>> &constraint_info_coarse,
    std::vector<unsigned int> &        dof_indices_fine);

  /**
   * Flag if the finite elements on the fine cells are continuous. If yes,
   * the multiplicity of DoF sharing a vertex/line as well as constraints have
   * to be taken into account via weights.
   */
  bool fine_element_is_continuous;

public:
  /**
   * Partitioner needed by the intermediate vector.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_coarse;

  /**
   * Partitioner needed by the intermediate vector.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine;

protected:
  /**
   * Internal vector needed for collecting all degrees of freedom of the fine
   * cells. It is only initialized if the fine-level DoF indices touch DoFs
   * other than the locally active ones (which we always assume can be
   * accessed by the given vectors in the prolongate/restrict functions),
   * otherwise it is left at size zero.
   */
  mutable LinearAlgebra::distributed::Vector<Number> vec_fine;

  /**
   * Internal vector on that the actual prolongation/restriction is performed.
   */
  mutable LinearAlgebra::distributed::Vector<Number> vec_coarse;

  /**
   * Embedded partitioner for efficient communication if locally relevant DoFs
   * are a subset of an external Partitioner object.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner>
    partitioner_coarse_embedded;

  /**
   * Embedded partitioner for efficient communication if locally relevant DoFs
   * are a subset of an external Partitioner object.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine_embedded;

  /**
   * Buffer for efficient communication if locally relevant DoFs
   * are a subset of an external Partitioner object.
   */
  mutable AlignedVector<Number> buffer_coarse_embedded;

  /**
   * Buffer for efficient communication if locally relevant DoFs
   * are a subset of an external Partitioner object.
   */
  mutable AlignedVector<Number> buffer_fine_embedded;
};



/**
 * Class for transfer between two multigrid levels for p- or global coarsening.
 *
 * The implementation of this class is explained in detail in @cite munch2022gc.
 */
template <int dim, typename VectorType>
class MGTwoLevelTransfer : public MGTwoLevelTransferBase<VectorType>
{
public:
  /**
   * Perform prolongation.
   */
  void
  prolongate_and_add(VectorType &dst, const VectorType &src) const override;

  /**
   * Perform restriction.
   */
  void
  restrict_and_add(VectorType &dst, const VectorType &src) const override;

  /**
   * Perform interpolation of a solution vector from the fine level to the
   * coarse level.
   */
  void
  interpolate(VectorType &dst, const VectorType &src) const override;

  /**
   * Enable inplace vector operations if external and internal vectors
   * are compatible.
   */
  void
  enable_inplace_operations_if_possible(
    const std::shared_ptr<const Utilities::MPI::Partitioner>
      &partitioner_coarse,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner_fine)
    override;

  /**
   * Return the memory consumption of the allocated memory in this class.
   */
  std::size_t
  memory_consumption() const override;
};



/**
 * Class for transfer between two multigrid levels for p- or global coarsening.
 * Specialization for LinearAlgebra::distributed::Vector.
 *
 * The implementation of this class is explained in detail in @cite munch2022gc.
 */
template <int dim, typename Number>
class MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<Number>>
  : public MGTwoLevelTransferBase<LinearAlgebra::distributed::Vector<Number>>
{
  using VectorizedArrayType = VectorizedArray<Number>;

public:
  /**
   * Set up global coarsening between the given DoFHandler objects (
   * @p dof_handler_fine and @p dof_handler_coarse). The transfer
   * can be only performed on active levels.
   */
  void
  reinit_geometric_transfer(
    const DoFHandler<dim> &          dof_handler_fine,
    const DoFHandler<dim> &          dof_handler_coarse,
    const AffineConstraints<Number> &constraint_fine =
      AffineConstraints<Number>(),
    const AffineConstraints<Number> &constraint_coarse =
      AffineConstraints<Number>(),
    const unsigned int mg_level_fine   = numbers::invalid_unsigned_int,
    const unsigned int mg_level_coarse = numbers::invalid_unsigned_int);

  /**
   * Set up polynomial coarsening between the given DoFHandler objects (
   * @p dof_handler_fine and @p dof_handler_coarse). Polynomial transfers
   * can be only performed on active levels (`numbers::invalid_unsigned_int`)
   * or on coarse-grid levels, i.e., levels without hanging nodes.
   *
   * @note The function polynomial_transfer_supported() can be used to
   *   check if the given polynomial coarsening strategy is supported.
   */
  void
  reinit_polynomial_transfer(
    const DoFHandler<dim> &          dof_handler_fine,
    const DoFHandler<dim> &          dof_handler_coarse,
    const AffineConstraints<Number> &constraint_fine =
      AffineConstraints<Number>(),
    const AffineConstraints<Number> &constraint_coarse =
      AffineConstraints<Number>(),
    const unsigned int mg_level_fine   = numbers::invalid_unsigned_int,
    const unsigned int mg_level_coarse = numbers::invalid_unsigned_int);

  /**
   * Set up transfer operator between the given DoFHandler objects (
   * @p dof_handler_fine and @p dof_handler_coarse). Depending on the
   * underlying Triangulation objects polynomial or geometrical global
   * coarsening is performed.
   *
   * @note While geometric transfer can be only performed on active levels
   *   (`numbers::invalid_unsigned_int`), polynomial transfers can also be
   *   performed on coarse-grid levels, i.e., levels without hanging nodes.
   *
   * @note The function polynomial_transfer_supported() can be used to
   *   check if the given polynomial coarsening strategy is supported.
   */
  void
  reinit(const DoFHandler<dim> &          dof_handler_fine,
         const DoFHandler<dim> &          dof_handler_coarse,
         const AffineConstraints<Number> &constraint_fine =
           AffineConstraints<Number>(),
         const AffineConstraints<Number> &constraint_coarse =
           AffineConstraints<Number>(),
         const unsigned int mg_level_fine   = numbers::invalid_unsigned_int,
         const unsigned int mg_level_coarse = numbers::invalid_unsigned_int);

  /**
   * Check if a fast templated version of the polynomial transfer between
   * @p fe_degree_fine and @p fe_degree_coarse is available.
   *
   * @note Currently, the polynomial coarsening strategies: 1) go-to-one,
   *   2) bisect, and 3) decrease-by-one are precompiled with templates for
   *   degrees up to 9.
   */
  static bool
  fast_polynomial_transfer_supported(const unsigned int fe_degree_fine,
                                     const unsigned int fe_degree_coarse);

  /**
   * Perform interpolation of a solution vector from the fine level to the
   * coarse level.
   */
  void
  interpolate(
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const override;

  /**
   * Enable inplace vector operations if external and internal vectors
   * are compatible.
   */
  void
  enable_inplace_operations_if_possible(
    const std::shared_ptr<const Utilities::MPI::Partitioner>
      &partitioner_coarse,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner_fine)
    override;

  /**
   * Return the memory consumption of the allocated memory in this class.
   */
  std::size_t
  memory_consumption() const override;

protected:
  void
  prolongate_and_add_internal(
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const override;

  void
  restrict_and_add_internal(
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const override;

private:
  /**
   * A multigrid transfer scheme. A multrigrid transfer class can have different
   * transfer schemes to enable p-adaptivity (one transfer scheme per
   * polynomial degree pair) and to enable global coarsening (one transfer
   * scheme for transfer between children and parent cells, as well as, one
   * transfer scheme for cells that are not refined).
   */
  struct MGTransferScheme
  {
    /**
     * Number of coarse cells.
     */
    unsigned int n_coarse_cells;

    /**
     * Number of degrees of freedom of a coarse cell.
     *
     * @note For tensor-product elements, the value equals
     *   `n_components * (degree_coarse + 1)^dim`.
     */
    unsigned int n_dofs_per_cell_coarse;

    /**
     * Number of degrees of freedom of fine cell.
     *
     * @note For tensor-product elements, the value equals
     *   `n_components * (n_dofs_per_cell_fine + 1)^dim`.
     */
    unsigned int n_dofs_per_cell_fine;

    /**
     * Polynomial degree of the finite element of a coarse cell.
     */
    unsigned int degree_coarse;

    /**
     * "Polynomial degree" of the finite element of the union of all children
     * of a coarse cell, i.e., actually `degree_fine * 2 + 1` if a cell is
     * refined.
     */
    unsigned int degree_fine;

    /**
     * Prolongation matrix for non-tensor-product elements.
     */
    AlignedVector<VectorizedArrayType> prolongation_matrix;

    /**
     * 1d prolongation matrix for tensor-product elements.
     */
    AlignedVector<VectorizedArrayType> prolongation_matrix_1d;

    /**
     * Restriction matrix for non-tensor-product elements.
     */
    AlignedVector<VectorizedArrayType> restriction_matrix;

    /**
     * 1d restriction matrix for tensor-product elements.
     */
    AlignedVector<VectorizedArrayType> restriction_matrix_1d;

    /**
     * ShapeInfo description of the coarse cell. Needed during the
     * fast application of hanging-node constraints.
     */
    internal::MatrixFreeFunctions::ShapeInfo<VectorizedArrayType>
      shape_info_coarse;
  };

  /**
   * Transfer schemes.
   */
  std::vector<MGTransferScheme> schemes;

  /**
   * Helper class for reading from and writing to global coarse vectors and for
   * applying constraints.
   */
  internal::MatrixFreeFunctions::ConstraintInfo<dim, VectorizedArrayType>
    constraint_info_coarse;

  /**
   * Helper class for reading from and writing to global fine vectors.
   */
  internal::MatrixFreeFunctions::ConstraintInfo<dim, VectorizedArrayType>
    constraint_info_fine;

  /**
   * Weights for continuous elements.
   */
  std::vector<Number> weights; // TODO: vectorize

  /**
   * Weights for continuous elements, compressed into 3^dim doubles per
   * cell if possible.
   */
  AlignedVector<VectorizedArrayType> weights_compressed;

  /**
   * Number of components.
   */
  unsigned int n_components;

  friend class internal::MGTwoLevelTransferImplementation;
};



/**
 * Class for transfer between two non-nested multigrid levels.
 *
 */
template <int dim, typename VectorType>
class MGTwoLevelTransferNonNested : public MGTwoLevelTransferBase<VectorType>
{
public:
  /**
   * Perform prolongation.
   */
  void
  prolongate_and_add(VectorType &dst, const VectorType &src) const override;

  /**
   * Perform restriction.
   */
  void
  restrict_and_add(VectorType &dst, const VectorType &src) const override;

  /**
   * Perform interpolation of a solution vector from the fine level to the
   * coarse level. This function is different from restriction, where a
   * weighted residual is transferred to a coarser level (transposition of
   * prolongation matrix).
   */
  void
  interpolate(VectorType &dst, const VectorType &src) const override;

  /**
   * Return the memory consumption of the allocated memory in this class.
   */
  std::size_t
  memory_consumption() const override;
};



/**
 * Class for transfer between two non-nested multigrid levels.
 *
 * Specialization for LinearAlgebra::distributed::Vector.
 *
 */
template <int dim, typename Number>
class MGTwoLevelTransferNonNested<dim,
                                  LinearAlgebra::distributed::Vector<Number>>
  : public MGTwoLevelTransferBase<LinearAlgebra::distributed::Vector<Number>>
{
private:
  using VectorizedArrayType = VectorizedArray<Number, 1>;

public:
  /**
   * Set up transfer operator between the given DoFHandler objects (
   * @p dof_handler_fine and @p dof_handler_coarse).
   */
  void
  reinit(const DoFHandler<dim> &          dof_handler_fine,
         const DoFHandler<dim> &          dof_handler_coarse,
         const Mapping<dim> &             mapping_fine,
         const Mapping<dim> &             mapping_coarse,
         const AffineConstraints<Number> &constraint_fine =
           AffineConstraints<Number>(),
         const AffineConstraints<Number> &constraint_coarse =
           AffineConstraints<Number>());

  /**
   * Perform interpolation of a solution vector from the fine level to the
   * coarse level. This function is different from restriction, where a
   * weighted residual is transferred to a coarser level (transposition of
   * prolongation matrix).
   */
  void
  interpolate(
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const override;

  /**
   * Enable inplace vector operations if external and internal vectors
   * are compatible.
   */
  void
  enable_inplace_operations_if_possible(
    const std::shared_ptr<const Utilities::MPI::Partitioner>
      &partitioner_coarse,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner_fine)
    override;

  /**
   * Return the memory consumption of the allocated memory in this class.
   */
  std::size_t
  memory_consumption() const override;

protected:
  /**
   * Perform prolongation.
   */
  void
  prolongate_and_add_internal(
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const override;

  /**
   * Perform restriction.
   */
  void
  restrict_and_add_internal(
    LinearAlgebra::distributed::Vector<Number> &      dst,
    const LinearAlgebra::distributed::Vector<Number> &src) const override;

private:
  /**
   * Object to evaluate shape functions on one mesh on visited support points of
   * the other mesh.
   */
  Utilities::MPI::RemotePointEvaluation<dim> rpe;

  /**
   * MappingInfo object needed as Mapping argument by FEPointEvaluation.
   */
  std::shared_ptr<NonMatching::MappingInfo<dim, dim, Number>> mapping_info;

  /**
   * Helper class for reading from and writing to global vectors and for
   * applying constraints.
   */
  internal::MatrixFreeFunctions::ConstraintInfo<dim, VectorizedArrayType>
    constraint_info;

  /**
   * Finite element of the coarse DoFHandler passed to reinit().
   */
  std::unique_ptr<FiniteElement<dim>> fe_coarse;

  /**
   * DoF indices of the fine cells, expressed in indices local to the MPI
   * rank.
   */
  std::vector<unsigned int> level_dof_indices_fine;

  /**
   * CRS like structure which points to DoFs associated with the same support
   * point. The vector stays empty if only one DoF corresponds to one support
   * point.
   */
  std::vector<unsigned int> level_dof_indices_fine_ptrs;
};



/**
 * Implementation of the MGTransferBase. In contrast to
 * other multigrid transfer operators, the user can provide separate
 * transfer operators of type MGTwoLevelTransfer between each level.
 *
 * This class currently only works for the tensor-product finite elements
 * FE_Q and FE_DGQ and simplex elements FE_SimplexP and FE_SimplexDGP as well as
 * for systems involving multiple components of one of these elements. Other
 * elements are currently not implemented.
 *
 * The implementation of this class is explained in detail in @cite munch2022gc.
 */
template <int dim, typename VectorType>
class MGTransferGlobalCoarsening : public dealii::MGTransferBase<VectorType>
{
public:
  /**
   * Value type.
   */
  using Number = typename VectorType::value_type;

  /**
   * Constructor taking a collection of transfer operators (with the coarsest
   * level kept empty in @p transfer) and an optional function that initializes the
   * internal level vectors within the function call copy_to_mg() if used in the
   * context of PreconditionMG. The template parameter @p MGTwoLevelTransferObject should derive from
   * MGTwoLevelTransferBase and implement the transfer operation (see for
   * instance MGTwoLevelTransfer). It can also be a std::shared_ptr or
   * std::unique_ptr to the actual transfer operator.
   */
  template <typename MGTwoLevelTransferObject>
  MGTransferGlobalCoarsening(
    const MGLevelObject<MGTwoLevelTransferObject> &transfer,
    const std::function<void(const unsigned int, VectorType &)>
      &initialize_dof_vector = {});

  /**
   * Similar function to MGTransferMatrixFree::build() with the difference that
   * the information for the prolongation for each level has already been built.
   * So this function only tries to optimize the data structures of the
   * two-level transfer operators, e.g., by enabling inplace vector operations,
   * by checking if @p external_partitioners and the internal ones are
   * compatible.
   */
  void
  build(const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
          &external_partitioners = {});

  /**
   * Same as above but taking a lambda for initializing vector instead of
   * partitioners.
   */
  void
  build(const std::function<void(const unsigned int, VectorType &)>
          &initialize_dof_vector);

  /**
   * Default constructor.
   *
   * @note See also MGTransferMatrixFree.
   */
  MGTransferGlobalCoarsening() = default;

  /**
   * Constructor with constraints. Equivalent to the default constructor
   * followed by initialize_constraints().
   *
   * @note See also MGTransferMatrixFree.
   */
  MGTransferGlobalCoarsening(const MGConstrainedDoFs &mg_constrained_dofs);

  /**
   * Initialize the constraints to be used in build().
   *
   * @note See also MGTransferMatrixFree.
   */
  void
  initialize_constraints(const MGConstrainedDoFs &mg_constrained_dofs);

  /**
   * Actually build the information for the prolongation for each level.
   *
   * @note See also MGTransferMatrixFree.
   */
  void
  build(const DoFHandler<dim> &dof_handler,
        const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
          &external_partitioners = {});

  /**
   * Same as above but taking a lambda for initializing vector instead of
   * partitioners.
   *
   * @note See also MGTransferMatrixFree.
   */
  void
  build(const DoFHandler<dim> &dof_handler,
        const std::function<void(const unsigned int, VectorType &)>
          &initialize_dof_vector);

  /**
   * Perform prolongation.
   */
  void
  prolongate(const unsigned int to_level,
             VectorType &       dst,
             const VectorType & src) const override;

  /**
   * Perform prolongation.
   */
  void
  prolongate_and_add(const unsigned int to_level,
                     VectorType &       dst,
                     const VectorType & src) const override;

  /**
   * Perform restriction.
   */
  virtual void
  restrict_and_add(const unsigned int from_level,
                   VectorType &       dst,
                   const VectorType & src) const override;

  /**
   * Initialize internal vectors and copy @p src vector to the finest
   * multigrid level.
   *
   * @note DoFHandler is not needed here, but is required by the interface.
   */
  template <class InVector>
  void
  copy_to_mg(const DoFHandler<dim> &    dof_handler,
             MGLevelObject<VectorType> &dst,
             const InVector &           src) const;

  /**
   * Initialize internal vectors and copy the values on the finest
   * multigrid level to @p dst vector.
   *
   * @note DoFHandler is not needed here, but is required by the interface.
   */
  template <class OutVector>
  void
  copy_from_mg(const DoFHandler<dim> &          dof_handler,
               OutVector &                      dst,
               const MGLevelObject<VectorType> &src) const;

  /**
   * Interpolate fine-mesh field @p src to each multigrid level in
   * @p dof_handler and store the result in @p dst. This function is different
   * from restriction, where a weighted residual is
   * transferred to a coarser level (transposition of prolongation matrix).
   *
   * The argument @p dst has to be initialized with the correct size according
   * to the number of levels of the triangulation.
   *
   * If an inner vector of @p dst is empty or has incorrect locally owned size,
   * it will be resized to locally relevant degrees of freedom on each level.
   */
  template <class InVector>
  void
  interpolate_to_mg(MGLevelObject<VectorType> &dst, const InVector &src) const;

  /**
   * Like the above function but with a user-provided DoFHandler as
   * additional argument. However, this DoFHandler is not used internally, but
   * is required to be able to use MGTransferGlobalCoarsening and
   * MGTransferMatrixFree as template argument.
   */
  template <class InVector>
  void
  interpolate_to_mg(const DoFHandler<dim> &    dof_handler,
                    MGLevelObject<VectorType> &dst,
                    const InVector &           src) const;

  /**
   * Return the memory consumption of the allocated memory in this class.
   *
   * @note Counts also the memory consumption of the underlying two-level
   *   transfer operators.
   */
  std::size_t
  memory_consumption() const;

  /**
   * Minimum level.
   */
  unsigned int
  min_level() const;

  /**
   * Maximum level.
   */
  unsigned int
  max_level() const;

  /**
   * TODO.
   */
  void
  print_indices(std::ostream &os) const
  {
    (void)os; // TODO
  }

  /**
   * Clear all data fields and brings the class into a condition similar
   * to after having called the default constructor.
   */
  void
  clear()
  {
    mg_constrained_dofs = nullptr;
    internal_transfer.clear();
    transfer.clear();
    external_partitioners.clear();
    solution_ghosted_global_vector.reinit(0);
    ghosted_global_vector.reinit(0);
    ghosted_level_vector.clear();
    solution_copy_indices.clear();
    copy_indices.clear();
    solution_copy_indices_level_mine.clear();
    copy_indices_level_mine.clear();
    copy_indices_global_mine.clear();
  }

private:
  /**
   * Initial internal transfer operator.
   *
   * @note See also MGTransferMatrixFree.
   */
  void
  intitialize_internal_transfer(
    const DoFHandler<dim> &                      dof_handler,
    const SmartPointer<const MGConstrainedDoFs> &mg_constrained_dofs);

  /**
   * Set references to two-level transfer operators to be used.
   */
  template <typename MGTwoLevelTransferObject>
  void
  intitialize_transfer_references(
    const MGLevelObject<MGTwoLevelTransferObject> &transfer);

  /**
   * Function to initialize internal level vectors.
   */
  template <class InVector>
  void
  initialize_dof_vector(const unsigned int level,
                        VectorType &       vector,
                        const InVector &   vector_reference) const;

  /**
   * TODO
   */
  void
  fill_and_communicate_copy_indices(const DoFHandler<dim> &dof_handler);

  /**
   * MGConstrainedDoFs passed during build().
   *
   * @note See also MGTransferMatrixFree.
   */
  SmartPointer<const MGConstrainedDoFs> mg_constrained_dofs;

  /**
   * Internal transfer operator.
   *
   * @note See also MGTransferMatrixFree.
   */
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> internal_transfer;

  /**
   * Collection of the two-level transfer operators.
   */
  MGLevelObject<SmartPointer<MGTwoLevelTransferBase<VectorType>>> transfer;

  /**
   * External partitioners used during initialize_dof_vector().
   */
  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
    external_partitioners;

  /**
   * TODO
   */
  bool perform_plain_copy;

  /**
   * TODO
   */
  bool perform_renumbered_plain_copy;

  /**
   * TODO
   */
  mutable VectorType solution_ghosted_global_vector;

  /**
   * TODO
   */
  mutable VectorType ghosted_global_vector;

  /**
   * TODO
   */
  mutable MGLevelObject<VectorType> ghosted_level_vector;

  /**
   * TODO
   */
  std::vector<Table<2, unsigned int>> solution_copy_indices;

  /**
   * TODO
   */
  std::vector<Table<2, unsigned int>> copy_indices;

  /**
   * TODO
   */
  std::vector<Table<2, unsigned int>> solution_copy_indices_level_mine;

  /**
   * TODO
   */
  std::vector<Table<2, unsigned int>> copy_indices_level_mine;

  /**
   * TODO
   */
  std::vector<Table<2, unsigned int>> copy_indices_global_mine;
};



/**
 * This class works with LinearAlgebra::distributed::BlockVector and
 * performs exactly the same transfer operations for each block as
 * MGTransferGlobalCoarsening.
 */
template <int dim, typename VectorType>
class MGTransferBlockGlobalCoarsening
  : public MGTransferBlockMatrixFreeBase<
      dim,
      typename VectorType::value_type,
      MGTransferGlobalCoarsening<dim, VectorType>>
{
public:
  /**
   * Constructor.
   */
  MGTransferBlockGlobalCoarsening(
    const MGTransferGlobalCoarsening<dim, VectorType> &transfer_operator);

  /**
   * Constructor.
   *
   * @note See also MGTransferBlockMatrixFree.
   */
  MGTransferBlockGlobalCoarsening() = default;

  /**
   * Constructor.
   *
   * @note See also MGTransferBlockMatrixFree.
   */
  MGTransferBlockGlobalCoarsening(const MGConstrainedDoFs &mg_constrained_dofs);

  /**
   * Constructor.
   *
   * @note See also MGTransferBlockMatrixFree.
   */
  MGTransferBlockGlobalCoarsening(
    const std::vector<MGConstrainedDoFs> &mg_constrained_dofs);

  /**
   * Initialize the constraints to be used in build().
   *
   * @note See also MGTransferBlockMatrixFree.
   */
  void
  initialize_constraints(const MGConstrainedDoFs &mg_constrained_dofs);

  /**
   * Same as above for the case that each block has its own DoFHandler.
   *
   * @note See also MGTransferBlockMatrixFree.
   */
  void
  initialize_constraints(
    const std::vector<MGConstrainedDoFs> &mg_constrained_dofs);

  /**
   * Actually build the information for the prolongation for each level.
   *
   * @note See also MGTransferBlockMatrixFree.
   */
  void
  build(const DoFHandler<dim> &dof_handler);

  /**
   * Same as above for the case that each block has its own DoFHandler.
   *
   * @note See also MGTransferBlockMatrixFree.
   */
  void
  build(const std::vector<const DoFHandler<dim> *> &dof_handler);

protected:
  const MGTransferGlobalCoarsening<dim, VectorType> &
  get_matrix_free_transfer(const unsigned int b) const override;

private:
  /**
   * Non-block version of transfer operation.
   */
  std::vector<SmartPointer<const MGTransferGlobalCoarsening<dim, VectorType>>>
    transfer_operators;

  /**
   * Internal non-block version of transfer operation.
   */
  std::vector<MGTransferGlobalCoarsening<dim, VectorType>>
    transfer_operators_internal;
};



#ifndef DOXYGEN

/* ----------------------- Inline functions --------------------------------- */



template <int dim, typename VectorType>
template <typename MGTwoLevelTransferObject>
MGTransferGlobalCoarsening<dim, VectorType>::MGTransferGlobalCoarsening(
  const MGLevelObject<MGTwoLevelTransferObject> &transfer,
  const std::function<void(const unsigned int, VectorType &)>
    &initialize_dof_vector)
{
  this->intitialize_transfer_references(transfer);
  this->build(initialize_dof_vector);
}



template <int dim, typename VectorType>
MGTransferGlobalCoarsening<dim, VectorType>::MGTransferGlobalCoarsening(
  const MGConstrainedDoFs &mg_constrained_dofs)
{
  this->initialize_constraints(mg_constrained_dofs);
}



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::initialize_constraints(
  const MGConstrainedDoFs &mg_constrained_dofs)
{
  this->mg_constrained_dofs = &mg_constrained_dofs;
}



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::intitialize_internal_transfer(
  const DoFHandler<dim> &                      dof_handler,
  const SmartPointer<const MGConstrainedDoFs> &mg_constrained_dofs)
{
  const unsigned int min_level = 0;
  const unsigned int max_level =
    dof_handler.get_triangulation().n_global_levels() - 1;

  MGLevelObject<AffineConstraints<typename VectorType::value_type>> constraints(
    min_level, max_level);

  if (mg_constrained_dofs)
    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        // TODO: set IndexSet

        // Dirichlet boundary conditions
        if (mg_constrained_dofs->have_boundary_indices())
          constraints[l].add_lines(
            mg_constrained_dofs->get_boundary_indices(l));

        // periodic-bounary conditions
        constraints[l].merge(
          mg_constrained_dofs->get_level_constraints(l),
          AffineConstraints<typename VectorType::value_type>::left_object_wins,
          true);

        // user constraints
        constraints[l].merge(
          mg_constrained_dofs->get_user_constraint_matrix(l),
          AffineConstraints<typename VectorType::value_type>::left_object_wins,
          true);

        constraints[l].close();
      }

  this->internal_transfer.resize(min_level, max_level);

  for (unsigned int l = min_level; l < max_level; ++l)
    internal_transfer[l + 1].reinit_geometric_transfer(
      dof_handler, dof_handler, constraints[l + 1], constraints[l], l + 1, l);
}



template <int dim, typename VectorType>
template <typename MGTwoLevelTransferObject>
void
MGTransferGlobalCoarsening<dim, VectorType>::intitialize_transfer_references(
  const MGLevelObject<MGTwoLevelTransferObject> &transfer)
{
  const unsigned int min_level = transfer.min_level();
  const unsigned int max_level = transfer.max_level();

  this->transfer.resize(min_level, max_level);

  for (unsigned int l = min_level; l <= max_level; ++l)
    this->transfer[l] = &const_cast<MGTwoLevelTransferBase<VectorType> &>(
      static_cast<const MGTwoLevelTransferBase<VectorType> &>(
        Utilities::get_underlying_value(transfer[l])));
}



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::build(
  const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
    &external_partitioners)
{
  this->external_partitioners = external_partitioners;

  if (this->external_partitioners.size() > 0)
    {
      const unsigned int min_level = transfer.min_level();
      const unsigned int max_level = transfer.max_level();

      AssertDimension(this->external_partitioners.size(), transfer.n_levels());

      for (unsigned int l = min_level + 1; l <= max_level; ++l)
        transfer[l]->enable_inplace_operations_if_possible(
          this->external_partitioners[l - 1 - min_level],
          this->external_partitioners[l - min_level]);
    }
  else
    {
      const unsigned int min_level = transfer.min_level();
      const unsigned int max_level = transfer.max_level();

      for (unsigned int l = min_level + 1; l <= max_level; ++l)
        {
          if (l == min_level + 1)
            this->external_partitioners.push_back(
              transfer[l]->partitioner_coarse);

          this->external_partitioners.push_back(transfer[l]->partitioner_fine);
        }
    }

  perform_plain_copy            = true;
  perform_renumbered_plain_copy = false;
}



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::build(
  const std::function<void(const unsigned int, VectorType &)>
    &initialize_dof_vector)
{
  if (initialize_dof_vector)
    {
      const unsigned int min_level = transfer.min_level();
      const unsigned int max_level = transfer.max_level();
      const unsigned int n_levels  = transfer.n_levels();

      std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
        external_partitioners(n_levels);

      for (unsigned int l = min_level; l <= max_level; ++l)
        {
          LinearAlgebra::distributed::Vector<typename VectorType::value_type>
            vector;
          initialize_dof_vector(l, vector);
          external_partitioners[l - min_level] = vector.get_partitioner();
        }

      this->build(external_partitioners);
    }
  else
    {
      this->build();
    }
}



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::build(
  const DoFHandler<dim> &dof_handler,
  const std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
    &external_partitioners)
{
  this->intitialize_internal_transfer(dof_handler, mg_constrained_dofs);
  this->intitialize_transfer_references(internal_transfer);
  this->build(external_partitioners);
  this->fill_and_communicate_copy_indices(dof_handler);
}



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::build(
  const DoFHandler<dim> &dof_handler,
  const std::function<void(const unsigned int, VectorType &)>
    &initialize_dof_vector)
{
  this->intitialize_internal_transfer(dof_handler, mg_constrained_dofs);
  this->intitialize_transfer_references(internal_transfer);
  this->build(initialize_dof_vector);
  this->fill_and_communicate_copy_indices(dof_handler);
}



namespace internal
{
  namespace MGTransfer
  {
    // Internal data structure that is used in the MPI communication in
    // fill_copy_indices().  It represents an entry in the copy_indices* map,
    // that associates a level dof index with a global dof index.
    struct DoFPair
    {
      unsigned int            level;
      types::global_dof_index global_dof_index;
      types::global_dof_index level_dof_index;

      DoFPair(const unsigned int            level,
              const types::global_dof_index global_dof_index,
              const types::global_dof_index level_dof_index)
        : level(level)
        , global_dof_index(global_dof_index)
        , level_dof_index(level_dof_index)
      {}

      DoFPair()
        : level(numbers::invalid_unsigned_int)
        , global_dof_index(numbers::invalid_dof_index)
        , level_dof_index(numbers::invalid_dof_index)
      {}
    };

    template <int dim, int spacedim>
    void
    fill_copy_indices(
      const DoFHandler<dim, spacedim> &dof_handler,
      const MGConstrainedDoFs *        mg_constrained_dofs,
      std::vector<std::vector<
        std::pair<types::global_dof_index, types::global_dof_index>>>
        &copy_indices,
      std::vector<std::vector<
        std::pair<types::global_dof_index, types::global_dof_index>>>
        &copy_indices_global_mine,
      std::vector<std::vector<
        std::pair<types::global_dof_index, types::global_dof_index>>>
        &        copy_indices_level_mine,
      const bool skip_interface_dofs)
    {
      // Now we are filling the variables copy_indices*, which are essentially
      // maps from global to mgdof for each level stored as a std::vector of
      // pairs. We need to split this map on each level depending on the
      // ownership of the global and mgdof, so that we later do not access
      // non-local elements in copy_to/from_mg.
      // We keep track in the bitfield dof_touched which global dof has been
      // processed already (on the current level). This is the same as the
      // multigrid running in serial.

      // map cpu_index -> vector of data
      // that will be copied into copy_indices_level_mine
      std::vector<DoFPair> send_data_temp;

      const unsigned int n_levels =
        dof_handler.get_triangulation().n_global_levels();
      copy_indices.resize(n_levels);
      copy_indices_global_mine.resize(n_levels);
      copy_indices_level_mine.resize(n_levels);
      const IndexSet &owned_dofs = dof_handler.locally_owned_dofs();

      const unsigned int dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();
      std::vector<types::global_dof_index> global_dof_indices(dofs_per_cell);
      std::vector<types::global_dof_index> level_dof_indices(dofs_per_cell);

      for (unsigned int level = 0; level < n_levels; ++level)
        {
          std::vector<bool> dof_touched(owned_dofs.n_elements(), false);
          const IndexSet &  owned_level_dofs =
            dof_handler.locally_owned_mg_dofs(level);

          // for the most common case where copy_indices are locally owned
          // both globally and on the level, we want to skip collecting pairs
          // and later sorting them. instead, we insert these indices into a
          // plain vector
          std::vector<types::global_dof_index> unrolled_copy_indices;

          copy_indices_level_mine[level].clear();
          copy_indices_global_mine[level].clear();

          for (const auto &level_cell :
               dof_handler.active_cell_iterators_on_level(level))
            {
              if (dof_handler.get_triangulation().locally_owned_subdomain() !=
                    numbers::invalid_subdomain_id &&
                  (level_cell->level_subdomain_id() ==
                     numbers::artificial_subdomain_id ||
                   level_cell->subdomain_id() ==
                     numbers::artificial_subdomain_id))
                continue;

              unrolled_copy_indices.resize(owned_dofs.n_elements(),
                                           numbers::invalid_dof_index);

              // get the dof numbers of this cell for the global and the
              // level-wise numbering
              level_cell->get_dof_indices(global_dof_indices);
              level_cell->get_mg_dof_indices(level_dof_indices);

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  // we need to ignore if the DoF is on a refinement edge
                  // (hanging node)
                  if (skip_interface_dofs && mg_constrained_dofs != nullptr &&
                      mg_constrained_dofs->at_refinement_edge(
                        level, level_dof_indices[i]))
                    continue;

                  // First check whether we own any of the active dof index
                  // and the level one. This check involves locally owned
                  // indices which often consist only of a single range, so
                  // they are cheap to look up.
                  const types::global_dof_index global_index_in_set =
                    owned_dofs.index_within_set(global_dof_indices[i]);
                  bool global_mine =
                    global_index_in_set != numbers::invalid_dof_index;
                  bool level_mine =
                    owned_level_dofs.is_element(level_dof_indices[i]);

                  if (global_mine && level_mine)
                    {
                      // we own both the active dof index and the level one ->
                      // set them into the vector, indexed by the local index
                      // range of the active dof
                      unrolled_copy_indices[global_index_in_set] =
                        level_dof_indices[i];
                    }
                  else if (global_mine &&
                           dof_touched[global_index_in_set] == false)
                    {
                      copy_indices_global_mine[level].emplace_back(
                        global_dof_indices[i], level_dof_indices[i]);

                      // send this to the owner of the level_dof:
                      send_data_temp.emplace_back(level,
                                                  global_dof_indices[i],
                                                  level_dof_indices[i]);
                      dof_touched[global_index_in_set] = true;
                    }
                  else
                    {
                      // somebody will send those to me
                    }
                }
            }

          // we now translate the plain vector for the copy_indices field into
          // the expected format of a pair of indices
          if (!unrolled_copy_indices.empty())
            {
              copy_indices[level].clear();

              // reserve the full length in case we did not hit global-mine
              // indices, so we expect all indices to come into copy_indices
              if (copy_indices_global_mine[level].empty())
                copy_indices[level].reserve(unrolled_copy_indices.size());

              // owned_dofs.nth_index_in_set(i) in this query is
              // usually cheap to look up as there are few ranges in
              // the locally owned part
              for (unsigned int i = 0; i < unrolled_copy_indices.size(); ++i)
                if (unrolled_copy_indices[i] != numbers::invalid_dof_index)
                  copy_indices[level].emplace_back(
                    owned_dofs.nth_index_in_set(i), unrolled_copy_indices[i]);
            }
        }

      const dealii::parallel::TriangulationBase<dim, spacedim> *tria =
        (dynamic_cast<const dealii::parallel::TriangulationBase<dim, spacedim>
                        *>(&dof_handler.get_triangulation()));
      AssertThrow(
        send_data_temp.empty() || tria != nullptr,
        ExcMessage(
          "We should only be sending information with a parallel Triangulation!"));

#  ifdef DEAL_II_WITH_MPI
      if (tria && Utilities::MPI::sum(send_data_temp.size(),
                                      tria->get_communicator()) > 0)
        {
          const std::set<types::subdomain_id> &neighbors =
            tria->level_ghost_owners();
          std::map<int, std::vector<DoFPair>> send_data;

          std::sort(send_data_temp.begin(),
                    send_data_temp.end(),
                    [](const DoFPair &lhs, const DoFPair &rhs) {
                      if (lhs.level < rhs.level)
                        return true;
                      if (lhs.level > rhs.level)
                        return false;

                      if (lhs.level_dof_index < rhs.level_dof_index)
                        return true;
                      if (lhs.level_dof_index > rhs.level_dof_index)
                        return false;

                      if (lhs.global_dof_index < rhs.global_dof_index)
                        return true;
                      else
                        return false;
                    });
          send_data_temp.erase(
            std::unique(send_data_temp.begin(),
                        send_data_temp.end(),
                        [](const DoFPair &lhs, const DoFPair &rhs) {
                          return (lhs.level == rhs.level) &&
                                 (lhs.level_dof_index == rhs.level_dof_index) &&
                                 (lhs.global_dof_index == rhs.global_dof_index);
                        }),
            send_data_temp.end());

          for (unsigned int level = 0; level < n_levels; ++level)
            {
              const IndexSet &owned_level_dofs =
                dof_handler.locally_owned_mg_dofs(level);

              std::vector<types::global_dof_index> level_dof_indices;
              std::vector<types::global_dof_index> global_dof_indices;
              for (const auto &dofpair : send_data_temp)
                if (dofpair.level == level)
                  {
                    level_dof_indices.push_back(dofpair.level_dof_index);
                    global_dof_indices.push_back(dofpair.global_dof_index);
                  }

              IndexSet is_ghost(owned_level_dofs.size());
              is_ghost.add_indices(level_dof_indices.begin(),
                                   level_dof_indices.end());

              AssertThrow(level_dof_indices.size() == is_ghost.n_elements(),
                          ExcMessage("Size does not match!"));

              const auto index_owner =
                Utilities::MPI::compute_index_owner(owned_level_dofs,
                                                    is_ghost,
                                                    tria->get_communicator());

              AssertThrow(level_dof_indices.size() == index_owner.size(),
                          ExcMessage("Size does not match!"));

              for (unsigned int i = 0; i < index_owner.size(); ++i)
                send_data[index_owner[i]].emplace_back(level,
                                                       global_dof_indices[i],
                                                       level_dof_indices[i]);
            }


          // Protect the send/recv logic with a mutex:
          static Utilities::MPI::CollectiveMutex      mutex;
          Utilities::MPI::CollectiveMutex::ScopedLock lock(
            mutex, tria->get_communicator());

          const int mpi_tag =
            Utilities::MPI::internal::Tags::mg_transfer_fill_copy_indices;

          // * send
          std::vector<MPI_Request> requests;
          {
            for (const auto dest : neighbors)
              {
                requests.push_back(MPI_Request());
                std::vector<DoFPair> &data = send_data[dest];

                const int ierr =
                  MPI_Isend(data.data(),
                            data.size() * sizeof(decltype(*data.data())),
                            MPI_BYTE,
                            dest,
                            mpi_tag,
                            tria->get_communicator(),
                            &*requests.rbegin());
                AssertThrowMPI(ierr);
              }
          }

          // * receive
          {
            // We should get one message from each of our neighbors
            std::vector<DoFPair> receive_buffer;
            for (unsigned int counter = 0; counter < neighbors.size();
                 ++counter)
              {
                MPI_Status status;
                int        ierr = MPI_Probe(MPI_ANY_SOURCE,
                                     mpi_tag,
                                     tria->get_communicator(),
                                     &status);
                AssertThrowMPI(ierr);
                int len;
                ierr = MPI_Get_count(&status, MPI_BYTE, &len);
                AssertThrowMPI(ierr);

                if (len == 0)
                  {
                    ierr = MPI_Recv(nullptr,
                                    0,
                                    MPI_BYTE,
                                    status.MPI_SOURCE,
                                    status.MPI_TAG,
                                    tria->get_communicator(),
                                    &status);
                    AssertThrowMPI(ierr);
                    continue;
                  }

                int count = len / sizeof(DoFPair);
                Assert(static_cast<int>(count * sizeof(DoFPair)) == len,
                       ExcInternalError());
                receive_buffer.resize(count);

                void *ptr = receive_buffer.data();
                ierr      = MPI_Recv(ptr,
                                len,
                                MPI_BYTE,
                                status.MPI_SOURCE,
                                status.MPI_TAG,
                                tria->get_communicator(),
                                &status);
                AssertThrowMPI(ierr);

                for (const auto &dof_pair : receive_buffer)
                  {
                    copy_indices_level_mine[dof_pair.level].emplace_back(
                      dof_pair.global_dof_index, dof_pair.level_dof_index);
                  }
              }
          }

          // * wait for all MPI_Isend to complete
          if (requests.size() > 0)
            {
              const int ierr = MPI_Waitall(requests.size(),
                                           requests.data(),
                                           MPI_STATUSES_IGNORE);
              AssertThrowMPI(ierr);
              requests.clear();
            }
#    ifdef DEBUG
          // Make sure in debug mode, that everybody sent/received all packages
          // on this level. If a deadlock occurs here, the list of expected
          // senders is not computed correctly.
          const int ierr = MPI_Barrier(tria->get_communicator());
          AssertThrowMPI(ierr);
#    endif
        }
#  endif

      // Sort the indices, except the copy_indices which already are
      // sorted. This will produce more reliable debug output for regression
      // tests and won't hurt performance even in release mode because the
      // non-owned indices are a small subset of all unknowns.
      std::less<std::pair<types::global_dof_index, types::global_dof_index>>
        compare;
      for (auto &level_indices : copy_indices_level_mine)
        std::sort(level_indices.begin(), level_indices.end(), compare);
      for (auto &level_indices : copy_indices_global_mine)
        std::sort(level_indices.begin(), level_indices.end(), compare);
    }
  } // namespace MGTransfer
} // namespace internal



namespace
{
  template <int dim, int spacedim, typename Number>
  void
  fill_internal(
    const DoFHandler<dim, spacedim> &           mg_dof,
    SmartPointer<const MGConstrainedDoFs>       mg_constrained_dofs,
    const MPI_Comm                              mpi_communicator,
    const bool                                  transfer_solution_vectors,
    std::vector<Table<2, unsigned int>> &       copy_indices,
    std::vector<Table<2, unsigned int>> &       copy_indices_global_mine,
    std::vector<Table<2, unsigned int>> &       copy_indices_level_mine,
    LinearAlgebra::distributed::Vector<Number> &ghosted_global_vector,
    MGLevelObject<LinearAlgebra::distributed::Vector<Number>>
      &ghosted_level_vector)
  {
    // first go to the usual routine...
    std::vector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>
      my_copy_indices;
    std::vector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>
      my_copy_indices_global_mine;
    std::vector<
      std::vector<std::pair<types::global_dof_index, types::global_dof_index>>>
      my_copy_indices_level_mine;

    internal::MGTransfer::fill_copy_indices(mg_dof,
                                            mg_constrained_dofs,
                                            my_copy_indices,
                                            my_copy_indices_global_mine,
                                            my_copy_indices_level_mine,
                                            !transfer_solution_vectors);

    // get all degrees of freedom that we need read access to in copy_to_mg
    // and copy_from_mg, respectively. We fill an IndexSet once on each level
    // (for the global_mine indices accessing remote level indices) and once
    // globally (for the level_mine indices accessing remote global indices).

    // the variables index_set and level_index_set are going to define the
    // ghost indices of the respective vectors (due to construction, these are
    // precisely the indices that we need)

    IndexSet index_set(mg_dof.locally_owned_dofs().size());
    std::vector<types::global_dof_index> accessed_indices;
    ghosted_level_vector.resize(0,
                                mg_dof.get_triangulation().n_global_levels() -
                                  1);
    std::vector<IndexSet> level_index_set(
      mg_dof.get_triangulation().n_global_levels());
    for (unsigned int l = 0; l < mg_dof.get_triangulation().n_global_levels();
         ++l)
      {
        for (const auto &indices : my_copy_indices_level_mine[l])
          accessed_indices.push_back(indices.first);
        std::vector<types::global_dof_index> accessed_level_indices;
        for (const auto &indices : my_copy_indices_global_mine[l])
          accessed_level_indices.push_back(indices.second);
        std::sort(accessed_level_indices.begin(), accessed_level_indices.end());
        level_index_set[l].set_size(mg_dof.locally_owned_mg_dofs(l).size());
        level_index_set[l].add_indices(accessed_level_indices.begin(),
                                       accessed_level_indices.end());
        level_index_set[l].compress();
        ghosted_level_vector[l].reinit(mg_dof.locally_owned_mg_dofs(l),
                                       level_index_set[l],
                                       mpi_communicator);
      }
    std::sort(accessed_indices.begin(), accessed_indices.end());
    index_set.add_indices(accessed_indices.begin(), accessed_indices.end());
    index_set.compress();
    ghosted_global_vector.reinit(mg_dof.locally_owned_dofs(),
                                 index_set,
                                 mpi_communicator);

    // localize the copy indices for faster access. Since all access will be
    // through the ghosted vector in 'data', we can use this (much faster)
    // option
    copy_indices.resize(mg_dof.get_triangulation().n_global_levels());
    copy_indices_level_mine.resize(
      mg_dof.get_triangulation().n_global_levels());
    copy_indices_global_mine.resize(
      mg_dof.get_triangulation().n_global_levels());
    for (unsigned int level = 0;
         level < mg_dof.get_triangulation().n_global_levels();
         ++level)
      {
        const Utilities::MPI::Partitioner &global_partitioner =
          *ghosted_global_vector.get_partitioner();
        const Utilities::MPI::Partitioner &level_partitioner =
          *ghosted_level_vector[level].get_partitioner();

        auto translate_indices =
          [&](const std::vector<
                std::pair<types::global_dof_index, types::global_dof_index>>
                &                     global_copy_indices,
              Table<2, unsigned int> &local_copy_indices) {
            local_copy_indices.reinit(2, global_copy_indices.size());
            for (unsigned int i = 0; i < global_copy_indices.size(); ++i)
              {
                local_copy_indices(0, i) = global_partitioner.global_to_local(
                  global_copy_indices[i].first);
                local_copy_indices(1, i) = level_partitioner.global_to_local(
                  global_copy_indices[i].second);
              }
          };

        // owned-owned case
        translate_indices(my_copy_indices[level], copy_indices[level]);

        // remote-owned case
        translate_indices(my_copy_indices_level_mine[level],
                          copy_indices_level_mine[level]);

        // owned-remote case
        translate_indices(my_copy_indices_global_mine[level],
                          copy_indices_global_mine[level]);
      }
  }
} // namespace



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::fill_and_communicate_copy_indices(
  const DoFHandler<dim> &mg_dof)
{
  const MPI_Comm mpi_communicator = mg_dof.get_communicator();

  fill_internal(mg_dof,
                mg_constrained_dofs,
                mpi_communicator,
                false,
                this->copy_indices,
                this->copy_indices_global_mine,
                this->copy_indices_level_mine,
                ghosted_global_vector,
                ghosted_level_vector);

  // in case we have hanging nodes which imply different ownership of
  // global and level dof indices, we must also fill the solution indices
  // which contain additional indices, otherwise they can re-use the
  // indices of the standard transfer.
  int have_refinement_edge_dofs = 0;
  if (mg_constrained_dofs != nullptr)
    for (unsigned int level = 0;
         level < mg_dof.get_triangulation().n_global_levels();
         ++level)
      if (mg_constrained_dofs->get_refinement_edge_indices(level).n_elements() >
          0)
        {
          have_refinement_edge_dofs = 1;
          break;
        }
  if (Utilities::MPI::max(have_refinement_edge_dofs, mpi_communicator) == 1)
    {
      // note: variables not needed
      std::vector<Table<2, unsigned int>> solution_copy_indices_global_mine;
      MGLevelObject<VectorType>           solution_ghosted_level_vector;

      fill_internal(mg_dof,
                    mg_constrained_dofs,
                    mpi_communicator,
                    true,
                    this->solution_copy_indices,
                    solution_copy_indices_global_mine,
                    this->solution_copy_indices_level_mine,
                    solution_ghosted_global_vector,
                    solution_ghosted_level_vector);
    }
  else
    {
      this->solution_copy_indices            = this->copy_indices;
      this->solution_copy_indices_level_mine = this->copy_indices_level_mine;
      solution_ghosted_global_vector         = ghosted_global_vector;
    }

  // Check if we can perform a cheaper "plain copy" (with or without
  // renumbering) instead of having to translate individual entries
  // using copy_indices*. This only works if a) we don't have to send
  // or receive any DoFs and we have all locally owned DoFs in our
  // copy_indices (so no adaptive refinement) and b) all processors
  // agree on the choice (see below).
  const bool my_perform_renumbered_plain_copy =
    (this->copy_indices.back().n_cols() ==
     mg_dof.locally_owned_dofs().n_elements()) &&
    (this->copy_indices_global_mine.back().n_rows() == 0) &&
    (this->copy_indices_level_mine.back().n_rows() == 0);

  bool my_perform_plain_copy = false;
  if (my_perform_renumbered_plain_copy)
    {
      my_perform_plain_copy = true;
      // check whether there is a renumbering of degrees of freedom on
      // either the finest level or the global dofs, which means that we
      // cannot apply a plain copy
      for (unsigned int i = 0; i < this->copy_indices.back().n_cols(); ++i)
        if (this->copy_indices.back()(0, i) != this->copy_indices.back()(1, i))
          {
            my_perform_plain_copy = false;
            break;
          }
    }

  // now do a global reduction over all processors to see what operation
  // they can agree upon
  perform_plain_copy =
    Utilities::MPI::min(static_cast<int>(my_perform_plain_copy),
                        mpi_communicator);
  perform_renumbered_plain_copy =
    Utilities::MPI::min(static_cast<int>(my_perform_renumbered_plain_copy),
                        mpi_communicator);

  // if we do a plain copy, no need to hold additional ghosted vectors
  if (perform_renumbered_plain_copy)
    {
      for (unsigned int i = 0; i < this->copy_indices.back().n_cols(); ++i)
        AssertDimension(this->copy_indices.back()(0, i), i);

      ghosted_global_vector.reinit(0);
      ghosted_level_vector.resize(0, 0);
      solution_ghosted_global_vector.reinit(0);
    }
}



template <int dim, typename VectorType>
template <class InVector>
void
MGTransferGlobalCoarsening<dim, VectorType>::initialize_dof_vector(
  const unsigned int level,
  VectorType &       vec,
  const InVector &   vec_reference) const
{
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;

  if (external_partitioners.empty())
    {
      vec.reinit(vec_reference);
    }
  else
    {
      Assert(transfer.min_level() <= level && level <= transfer.max_level(),
             ExcInternalError());

      partitioner = external_partitioners[level - transfer.min_level()];
    }

  // check if vectors are already correctly initalized

  // yes: same partitioners are used
  if (vec.get_partitioner().get() == partitioner.get())
    return; // nothing to do

  // yes: vectors are compatible
  if (vec.size() == partitioner->size() &&
      vec.locally_owned_size() == partitioner->locally_owned_size())
    return; // nothing to do

  // no
  vec.reinit(partitioner);
}



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::prolongate(
  const unsigned int to_level,
  VectorType &       dst,
  const VectorType & src) const
{
  dst = 0;
  prolongate_and_add(to_level, dst, src);
}



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::prolongate_and_add(
  const unsigned int to_level,
  VectorType &       dst,
  const VectorType & src) const
{
  this->transfer[to_level]->prolongate_and_add(dst, src);
}



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::restrict_and_add(
  const unsigned int from_level,
  VectorType &       dst,
  const VectorType & src) const
{
  this->transfer[from_level]->restrict_and_add(dst, src);
}



template <int dim, typename VectorType>
template <class InVector>
void
MGTransferGlobalCoarsening<dim, VectorType>::copy_to_mg(
  const DoFHandler<dim> &    dof_handler,
  MGLevelObject<VectorType> &dst,
  const InVector &           src) const
{
  (void)dof_handler;

  for (unsigned int level = dst.min_level(); level <= dst.max_level(); ++level)
    {
      initialize_dof_vector(level, dst[level], src);

      dst[level] = 0.0; // TODO
    }

  if (perform_plain_copy)
    {
      dst[dst.max_level()].copy_locally_owned_data_from(src);
    }
  else if (perform_renumbered_plain_copy)
    {
      auto &dst_level = dst[dst.max_level()];

      for (unsigned int i = 0; i < copy_indices.back().n_cols(); ++i)
        dst_level.local_element(copy_indices.back()(1, i)) =
          src.local_element(i);
    }
  else
    {
      ghosted_global_vector = src;
      ghosted_global_vector.update_ghost_values();

      for (unsigned int l = dst.max_level() + 1; l != dst.min_level();)
        {
          --l;

          auto &dst_level = dst[l];

          const auto copy_unknowns = [&](const auto &indices) {
            for (unsigned int i = 0; i < indices.n_cols(); ++i)
              dst_level.local_element(indices(1, i)) =
                ghosted_global_vector.local_element(indices(0, i));
          };

          copy_unknowns(copy_indices[l]);
          copy_unknowns(copy_indices_level_mine[l]);

          dst_level.compress(VectorOperation::insert);
        }
    }
}



template <int dim, typename VectorType>
template <class OutVector>
void
MGTransferGlobalCoarsening<dim, VectorType>::copy_from_mg(
  const DoFHandler<dim> &          dof_handler,
  OutVector &                      dst,
  const MGLevelObject<VectorType> &src) const
{
  (void)dof_handler;

  if (perform_plain_copy)
    {
      dst.zero_out_ghost_values();
      dst.copy_locally_owned_data_from(src[src.max_level()]);
    }
  else if (perform_renumbered_plain_copy)
    {
      const auto &src_level = src[src.max_level()];
      dst.zero_out_ghost_values();
      for (unsigned int i = 0; i < copy_indices.back().n_cols(); ++i)
        dst.local_element(i) =
          src_level.local_element(copy_indices.back()(1, i));
    }
  else
    {
      dst = 0;
      for (unsigned int l = src.min_level(); l <= src.max_level(); ++l)
        {
          auto &ghosted_vector = ghosted_level_vector[l];

          if (ghosted_level_vector[l].size() > 0)
            ghosted_vector = src[l];

          const auto *const ghosted_vector_ptr =
            (ghosted_level_vector[l].size() > 0) ? &ghosted_vector : &src[l];

          ghosted_vector_ptr->update_ghost_values();

          const auto copy_unknowns = [&](const auto &indices) {
            for (unsigned int i = 0; i < indices.n_cols(); ++i)
              dst.local_element(indices(0, i)) =
                ghosted_vector_ptr->local_element(indices(1, i));
          };

          copy_unknowns(copy_indices[l]);
          copy_unknowns(copy_indices_global_mine[l]);
        }
      dst.compress(VectorOperation::insert);
    }
}



template <int dim, typename VectorType>
template <class InVector>
void
MGTransferGlobalCoarsening<dim, VectorType>::interpolate_to_mg(
  MGLevelObject<VectorType> &dst,
  const InVector &           src) const
{
  const unsigned int min_level = transfer.min_level();
  const unsigned int max_level = transfer.max_level();

  AssertDimension(min_level, dst.min_level());
  AssertDimension(max_level, dst.max_level());

  for (unsigned int level = min_level; level <= max_level; ++level)
    initialize_dof_vector(level, dst[level], src);

  if (perform_plain_copy)
    {
      dst[max_level].copy_locally_owned_data_from(src);

      for (unsigned int l = max_level; l > min_level; --l)
        this->transfer[l]->interpolate(dst[l - 1], dst[l]);
    }
  else if (perform_renumbered_plain_copy)
    {
      auto &dst_level = dst[max_level];

      for (unsigned int i = 0; i < solution_copy_indices.back().n_cols(); ++i)
        dst_level.local_element(solution_copy_indices.back()(1, i)) =
          src.local_element(i);

      for (unsigned int l = max_level; l > min_level; --l)
        this->transfer[l]->interpolate(dst[l - 1], dst[l]);
    }
  else
    {
      solution_ghosted_global_vector = src;
      solution_ghosted_global_vector.update_ghost_values();

      for (unsigned int l = max_level + 1; l != min_level;)
        {
          --l;

          auto &dst_level = dst[l];

          const auto copy_unknowns = [&](const auto &indices) {
            for (unsigned int i = 0; i < indices.n_cols(); ++i)
              dst_level.local_element(indices(1, i)) =
                solution_ghosted_global_vector.local_element(indices(0, i));
          };

          copy_unknowns(solution_copy_indices[l]);
          copy_unknowns(solution_copy_indices_level_mine[l]);

          dst_level.compress(VectorOperation::insert);

          if (l != min_level)
            this->transfer[l]->interpolate(dst[l - 1], dst[l]);
        }
    }
}



template <int dim, typename VectorType>
template <class InVector>
void
MGTransferGlobalCoarsening<dim, VectorType>::interpolate_to_mg(
  const DoFHandler<dim> &    dof_handler,
  MGLevelObject<VectorType> &dst,
  const InVector &           src) const
{
  (void)dof_handler;

  this->interpolate_to_mg(dst, src);
}



template <int dim, typename VectorType>
std::size_t
MGTransferGlobalCoarsening<dim, VectorType>::memory_consumption() const
{
  std::size_t size = 0;

  const unsigned int min_level = transfer.min_level();
  const unsigned int max_level = transfer.max_level();

  for (unsigned int l = min_level + 1; l <= max_level; ++l)
    size += this->transfer[l]->memory_consumption();

  return size;
}



template <int dim, typename VectorType>
inline unsigned int
MGTransferGlobalCoarsening<dim, VectorType>::min_level() const
{
  return transfer.min_level();
}



template <int dim, typename VectorType>
inline unsigned int
MGTransferGlobalCoarsening<dim, VectorType>::max_level() const
{
  return transfer.max_level();
}

#endif

DEAL_II_NAMESPACE_CLOSE

#endif
