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
#include <deal.II/multigrid/mg_constrained_dofs.h>

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

  /**
   * Partitioner needed by the intermediate vector.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_coarse;

  /**
   * Partitioner needed by the intermediate vector.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine;

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
  MGTransferGlobalCoarsening();

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
  template <class InVector, int spacedim>
  void
  copy_to_mg(const DoFHandler<dim, spacedim> &dof_handler,
             MGLevelObject<VectorType> &      dst,
             const InVector &                 src) const;

  /**
   * Initialize internal vectors and copy the values on the finest
   * multigrid level to @p dst vector.
   *
   * @note DoFHandler is not needed here, but is required by the interface.
   */
  template <class OutVector, int spacedim>
  void
  copy_from_mg(const DoFHandler<dim, spacedim> &dof_handler,
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
  template <class InVector, int spacedim>
  void
  interpolate_to_mg(const DoFHandler<dim, spacedim> &dof_handler,
                    MGLevelObject<VectorType> &      dst,
                    const InVector &                 src) const;

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
  void
  initialize_dof_vector(const unsigned int level, VectorType &vector) const;

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
};



/**
 * Base class of MGTransferBlockMatrixFree. While MGTransferBlockMatrixFree
 * contains all the setup routines of the transfer operators for the blocks,
 * this class simply applies them, e.g., for restricting and prolongating.
 */
template <int dim, typename Number, typename TransferType>
class MGTransferBlockMatrixFreeBase
  : public MGTransferBase<LinearAlgebra::distributed::BlockVector<Number>>
{
public:
  MGTransferBlockMatrixFreeBase(const bool same_for_all)
    : same_for_all(same_for_all)
  {}

  /**
   * Prolongate a vector from level <tt>to_level-1</tt> to level
   * <tt>to_level</tt> using the embedding matrices of the underlying finite
   * element. The previous content of <tt>dst</tt> is overwritten.
   *
   * @param to_level The index of the level to prolongate to, which is the
   * level of @p dst.
   *
   * @param src is a vector with as many elements as there are degrees of
   * freedom on the coarser level involved.
   *
   * @param dst has as many elements as there are degrees of freedom on the
   * finer level.
   */
  virtual void
  prolongate(
    const unsigned int                                     to_level,
    LinearAlgebra::distributed::BlockVector<Number> &      dst,
    const LinearAlgebra::distributed::BlockVector<Number> &src) const override;

  virtual void
  prolongate_and_add(
    const unsigned int                                     to_level,
    LinearAlgebra::distributed::BlockVector<Number> &      dst,
    const LinearAlgebra::distributed::BlockVector<Number> &src) const override;

  /**
   * Restrict a vector from level <tt>from_level</tt> to level
   * <tt>from_level-1</tt> using the transpose operation of the prolongate()
   * method. If the region covered by cells on level <tt>from_level</tt> is
   * smaller than that of level <tt>from_level-1</tt> (local refinement), then
   * some degrees of freedom in <tt>dst</tt> are active and will not be
   * altered. For the other degrees of freedom, the result of the restriction
   * is added.
   *
   * @param from_level The index of the level to restrict from, which is the
   * level of @p src.
   *
   * @param src is a vector with as many elements as there are degrees of
   * freedom on the finer level involved.
   *
   * @param dst has as many elements as there are degrees of freedom on the
   * coarser level.
   */
  virtual void
  restrict_and_add(
    const unsigned int                                     from_level,
    LinearAlgebra::distributed::BlockVector<Number> &      dst,
    const LinearAlgebra::distributed::BlockVector<Number> &src) const override;

  /**
   * Transfer from a block-vector on the global grid to block-vectors defined
   * on each of the levels separately for active degrees of freedom.
   * In particular, for a globally refined mesh only the finest level in @p dst
   * is filled as a plain copy of @p src. All the other level objects are left
   * untouched.
   *
   * This function will initialize @p dst accordingly if needed as required by
   * the Multigrid class.
   */
  template <typename BlockVectorType2, int spacedim>
  void
  copy_to_mg(
    const DoFHandler<dim, spacedim> &                               dof_handler,
    MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>> &dst,
    const BlockVectorType2 &                                        src) const;

  /**
   * Same as above for the case that each block has its own DoFHandler.
   */
  template <typename BlockVectorType2, int spacedim>
  void
  copy_to_mg(
    const std::vector<const DoFHandler<dim, spacedim> *> &          dof_handler,
    MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>> &dst,
    const BlockVectorType2 &                                        src) const;

  /**
   * Transfer from multi-level block-vector to normal vector.
   */
  template <typename BlockVectorType2, int spacedim>
  void
  copy_from_mg(
    const DoFHandler<dim, spacedim> &dof_handler,
    BlockVectorType2 &               dst,
    const MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>> &src)
    const;

  /**
   * Same as above for the case that each block has its own DoFHandler.
   */
  template <typename BlockVectorType2, int spacedim>
  void
  copy_from_mg(
    const std::vector<const DoFHandler<dim, spacedim> *> &dof_handler,
    BlockVectorType2 &                                    dst,
    const MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>> &src)
    const;

  /**
   * This class can both be used with a single DoFHandler
   * or a separate DoFHandler for each block.
   */
  static const bool supports_dof_handler_vector = true;

protected:
  /**
   * Return the right non-block transfer operator. Has to be implemented by
   * the derived class.
   */
  virtual const TransferType &
  get_matrix_free_transfer(const unsigned int b) const = 0;

  /**
   * A flag to indicate whether the same DoFHandler is used for all
   * the components or if each block has its own DoFHandler.
   */
  const bool same_for_all;
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

protected:
  const MGTransferGlobalCoarsening<dim, VectorType> &
  get_matrix_free_transfer(const unsigned int b) const override;

private:
  /**
   * Non-block version of transfer operation.
   */
  const MGTransferGlobalCoarsening<dim, VectorType> &transfer_operator;
};



#ifndef DOXYGEN

/* ----------------------- Inline functions --------------------------------- */



template <int dim, typename Number, typename TransferType>
template <typename BlockVectorType2, int spacedim>
void
MGTransferBlockMatrixFreeBase<dim, Number, TransferType>::copy_to_mg(
  const DoFHandler<dim, spacedim> &                               dof_handler,
  MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>> &dst,
  const BlockVectorType2 &                                        src) const
{
  Assert(same_for_all,
         ExcMessage(
           "This object was initialized with support for usage with one "
           "DoFHandler for each block, but this method assumes that "
           "the same DoFHandler is used for all the blocks!"));
  const std::vector<const DoFHandler<dim, spacedim> *> mg_dofs(src.n_blocks(),
                                                               &dof_handler);

  copy_to_mg(mg_dofs, dst, src);
}



template <int dim, typename Number, typename TransferType>
template <typename BlockVectorType2, int spacedim>
void
MGTransferBlockMatrixFreeBase<dim, Number, TransferType>::copy_to_mg(
  const std::vector<const DoFHandler<dim, spacedim> *> &          dof_handler,
  MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>> &dst,
  const BlockVectorType2 &                                        src) const
{
  const unsigned int n_blocks = src.n_blocks();
  AssertDimension(dof_handler.size(), n_blocks);

  if (n_blocks == 0)
    return;

  const unsigned int min_level = dst.min_level();
  const unsigned int max_level = dst.max_level();

  for (unsigned int level = min_level; level <= max_level; ++level)
    if (dst[level].n_blocks() != n_blocks)
      dst[level].reinit(n_blocks);

  // FIXME: this a quite ugly as we need a temporary object:
  MGLevelObject<LinearAlgebra::distributed::Vector<Number>> dst_non_block(
    min_level, max_level);

  for (unsigned int b = 0; b < n_blocks; ++b)
    {
      const unsigned int data_block = same_for_all ? 0 : b;
      get_matrix_free_transfer(data_block)
        .copy_to_mg(*dof_handler[b], dst_non_block, src.block(b));

      for (unsigned int l = min_level; l <= max_level; ++l)
        dst[l].block(b) = dst_non_block[l];
    }

  for (unsigned int level = min_level; level <= max_level; ++level)
    dst[level].collect_sizes();
}

template <int dim, typename Number, typename TransferType>
template <typename BlockVectorType2, int spacedim>
void
MGTransferBlockMatrixFreeBase<dim, Number, TransferType>::copy_from_mg(
  const DoFHandler<dim, spacedim> &dof_handler,
  BlockVectorType2 &               dst,
  const MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>> &src)
  const
{
  Assert(same_for_all,
         ExcMessage(
           "This object was initialized with support for usage with one "
           "DoFHandler for each block, but this method assumes that "
           "the same DoFHandler is used for all the blocks!"));
  const std::vector<const DoFHandler<dim, spacedim> *> mg_dofs(dst.n_blocks(),
                                                               &dof_handler);

  copy_from_mg(mg_dofs, dst, src);
}

template <int dim, typename Number, typename TransferType>
template <typename BlockVectorType2, int spacedim>
void
MGTransferBlockMatrixFreeBase<dim, Number, TransferType>::copy_from_mg(
  const std::vector<const DoFHandler<dim, spacedim> *> &dof_handler,
  BlockVectorType2 &                                    dst,
  const MGLevelObject<LinearAlgebra::distributed::BlockVector<Number>> &src)
  const
{
  const unsigned int n_blocks = dst.n_blocks();
  AssertDimension(dof_handler.size(), n_blocks);

  if (n_blocks == 0)
    return;

  const unsigned int min_level = src.min_level();
  const unsigned int max_level = src.max_level();

  for (unsigned int l = min_level; l <= max_level; ++l)
    AssertDimension(src[l].n_blocks(), dst.n_blocks());

  // FIXME: this a quite ugly as we need a temporary object:
  MGLevelObject<LinearAlgebra::distributed::Vector<Number>> src_non_block(
    min_level, max_level);

  for (unsigned int b = 0; b < n_blocks; ++b)
    {
      for (unsigned int l = min_level; l <= max_level; ++l)
        {
          src_non_block[l].reinit(src[l].block(b));
          src_non_block[l] = src[l].block(b);
        }
      const unsigned int data_block = same_for_all ? 0 : b;
      get_matrix_free_transfer(data_block)
        .copy_from_mg(*dof_handler[b], dst.block(b), src_non_block);
    }
}



template <int dim, typename VectorType>
MGTransferGlobalCoarsening<dim, VectorType>::MGTransferGlobalCoarsening()
{}



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
        constraints[l].add_lines(mg_constrained_dofs->get_boundary_indices(l));

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
}



template <int dim, typename VectorType>
void
MGTransferGlobalCoarsening<dim, VectorType>::initialize_dof_vector(
  const unsigned int level,
  VectorType &       vec) const
{
  AssertDimension(transfer.n_levels(), external_partitioners.size());
  AssertIndexRange(level, transfer.max_level() + 1);

  const auto &external_partitioner =
    external_partitioners[level - transfer.min_level()];

  if (vec.get_partitioner().get() != external_partitioner.get())
    vec.reinit(external_partitioner);
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
template <class InVector, int spacedim>
void
MGTransferGlobalCoarsening<dim, VectorType>::copy_to_mg(
  const DoFHandler<dim, spacedim> &dof_handler,
  MGLevelObject<VectorType> &      dst,
  const InVector &                 src) const
{
  (void)dof_handler;

  for (unsigned int level = dst.min_level(); level <= dst.max_level(); ++level)
    {
      initialize_dof_vector(level, dst[level]);

      if (level == dst.max_level())
        dst[level].copy_locally_owned_data_from(src);
      else
        dst[level] = 0.0;
    }
}



template <int dim, typename VectorType>
template <class OutVector, int spacedim>
void
MGTransferGlobalCoarsening<dim, VectorType>::copy_from_mg(
  const DoFHandler<dim, spacedim> &dof_handler,
  OutVector &                      dst,
  const MGLevelObject<VectorType> &src) const
{
  (void)dof_handler;

  dst.copy_locally_owned_data_from(src[src.max_level()]);
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
    initialize_dof_vector(level, dst[level]);

  dst[transfer.max_level()].copy_locally_owned_data_from(src);

  for (unsigned int l = max_level; l > min_level; --l)
    this->transfer[l]->interpolate(dst[l - 1], dst[l]);
}



template <int dim, typename VectorType>
template <class InVector, int spacedim>
void
MGTransferGlobalCoarsening<dim, VectorType>::interpolate_to_mg(
  const DoFHandler<dim, spacedim> &dof_handler,
  MGLevelObject<VectorType> &      dst,
  const InVector &                 src) const
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
