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

#ifndef dealii_mg_transfer_global_coarsening_h
#define dealii_mg_transfer_global_coarsening_h

#include <deal.II/base/mg_level_object.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_base.h>

DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
namespace internal
{
  class MGTwoLevelTransferImplementation;
}
#endif


/**
 * Global coarsening utility functions.
 */
namespace MGTransferGlobalCoarseningTools
{
  enum class PolynomialSequenceType
  {
    bisect,
    decrease_by_one,
    go_to_one
  };

  unsigned int
  generate_level_degree(const unsigned int            previous_fe_degree,
                        const PolynomialSequenceType &p_sequence)
  {
    switch (p_sequence)
      {
        case PolynomialSequenceType::bisect:
          return std::max(previous_fe_degree / 2, 1u);
        case PolynomialSequenceType::decrease_by_one:
          return std::max(previous_fe_degree - 1, 1u);
        case PolynomialSequenceType::go_to_one:
          return 1u;
        default:
          Assert(false, StandardExceptions::ExcNotImplemented());
          return 1u;
      }
  }

  std::vector<unsigned int>
  create_p_sequence(const unsigned int            degree,
                    const PolynomialSequenceType &p_sequence)
  {
    std::vector<unsigned int> degrees;
    degrees.push_back(degree);

    unsigned int previous_fe_degree = degree;
    while (previous_fe_degree > 1)
      {
        const unsigned int level_degree =
          generate_level_degree(previous_fe_degree, p_sequence);

        degrees.push_back(level_degree);
        previous_fe_degree = level_degree;
      }

    std::reverse(degrees.begin(), degrees.end());

    return degrees;
  }

  template <int dim, int spacedim>
  void
  create_global_coarsening_sequence(
    MGLevelObject<std::shared_ptr<Triangulation<dim, spacedim>>>
      &                                 coarse_grid_triangulations,
    const Triangulation<dim, spacedim> &fine_triangulation_in)
  {
    const auto fine_triangulation =
      dynamic_cast<const parallel::distributed::Triangulation<dim, spacedim> *>(
        &fine_triangulation_in);

    Assert(fine_triangulation, ExcNotImplemented());

    const unsigned int min_level = 0;
    const unsigned int max_level = fine_triangulation->n_global_levels() - 1;

    const auto coarse_mesh_description =
      GridTools::get_coarse_mesh_description(*fine_triangulation);

    coarse_grid_triangulations.resize(
      min_level, max_level); // note: maxlevel is not constructed!

    // create coarse meshes
    for (unsigned int l = max_level - 1; l != numbers::invalid_unsigned_int;
         --l)
      {
        // store finer mesh to file
        if (l + 1 == max_level)
          fine_triangulation->save("mesh");
        else
          dynamic_cast<
            const parallel::distributed::Triangulation<dim, spacedim> *>(
            coarse_grid_triangulations[l + 1].get())
            ->save("mesh");
        MPI_Barrier(fine_triangulation->get_communicator());

        // create empty triangulation
        auto new_tria =
          std::make_shared<parallel::distributed::Triangulation<dim, spacedim>>(
            fine_triangulation->get_communicator(),
            dealii::Triangulation<dim>::limit_level_difference_at_vertices);

        // create coarse grid
        new_tria->create_triangulation(std::get<0>(coarse_mesh_description),
                                       std::get<1>(coarse_mesh_description),
                                       std::get<2>(coarse_mesh_description));

        for (const auto i : fine_triangulation->get_manifold_ids())
          if (i != numbers::flat_manifold_id)
            new_tria->set_manifold(i, fine_triangulation->get_manifold(i));

        // create refinement hierarchy (by loading stored mesh)
        new_tria->load("mesh", false);
        MPI_Barrier(fine_triangulation->get_communicator());

        // coarsen mesh
        new_tria->coarsen_global();

        // save mesh
        coarse_grid_triangulations[l] = new_tria;
      }
  }
} // namespace MGTransferGlobalCoarseningTools


/**
 * Class for transfer between two multigrid levels for p- or global coarsening.
 */
template <int dim, typename VectorType>
class MGTwoLevelTransfer
{
public:
  /**
   * Perform prolongation.
   */
  void
  prolongate(VectorType &dst, const VectorType &src) const;

  /**
   * Perform restriction.
   */
  void
  restrict_and_add(VectorType &dst, const VectorType &src) const;
};



/**
 * Class for transfer between two multigrid levels for p- or global coarsening.
 * Specialization for LinearAlgebra::distributed::Vector.
 */
template <int dim, typename Number>
class MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<Number>>
{
public:
  /**
   * Set up global coarsening transfer @p transfer between the given
   * dof-handlers.
   */
  void
  reinit_geometric_transfer(const DoFHandler<dim> &          dof_handler_fine,
                            const DoFHandler<dim> &          dof_handler_coarse,
                            const AffineConstraints<Number> &constraint_fine,
                            const AffineConstraints<Number> &constraint_coarse);

  /**
   * Set up polynomial coarsening transfer @p transfer between the given
   * dof-handlers.
   *
   * @note The function polynomial_transfer_supported() can be used to
   *   check if the given polynomial coarsening strategy is supported.
   */
  void
  reinit_polynomial_transfer(
    const DoFHandler<dim> &          dof_handler_fine,
    const DoFHandler<dim> &          dof_handler_coarse,
    const AffineConstraints<Number> &constraint_fine,
    const AffineConstraints<Number> &constraint_coarse);

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
   * Perform prolongation.
   */
  void
  prolongate(LinearAlgebra::distributed::Vector<Number> &      dst,
             const LinearAlgebra::distributed::Vector<Number> &src) const;

  /**
   * Perform restriction.
   */
  void
  restrict_and_add(LinearAlgebra::distributed::Vector<Number> &      dst,
                   const LinearAlgebra::distributed::Vector<Number> &src) const;

  /**
   * Print internal data structures to stream @p out.
   */
  template <typename Stream>
  void
  print_internal(Stream &out) const;

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
     */
    unsigned int dofs_per_cell_coarse;

    /**
     * Number of degrees of freedom of fine cell.
     */
    unsigned int dofs_per_cell_fine;

    /**
     * Polynomial degree of the finite element of the coarse cells.
     */
    unsigned int degree_coarse;

    /**
     * Polynomial degree of the finite element of the fine cells.
     */
    unsigned int degree_fine;

    /**
     * Flag if the finite element on the fine cells are continuous. If yes,
     * the multiplicity of DoF sharing a vertex/line as well as constraints have
     * to be taken in account via weights.
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
     * DoF indices of the coarse cells, expressed in indices local to the MPI
     * rank.
     */
    std::vector<unsigned int> level_dof_indices_coarse;

    /**
     * DoF indices of the fine cells, expressed in indices local to the MPI
     * rank.
     */
    std::vector<unsigned int> level_dof_indices_fine;

    /**
     * Print internal data structures to stream @p out.
     */
    template <typename Stream>
    void
    print(Stream &out) const;
  };

  /**
   * Transfer schemes.
   */
  std::vector<MGTransferScheme> schemes;

  /**
   * Partitioner needed by the intermediate vector.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine;

  /**
   * Partitioner needed by the intermediate vector.
   */
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_coarse;

  /**
   * Internal vector needed for collecting all degrees of freedom of the
   * fine cells.
   */
  mutable LinearAlgebra::distributed::Vector<Number> vec_fine;

  /**
   * Internal vector on that the actual prolongation/restriction is performed.
   */
  mutable LinearAlgebra::distributed::Vector<Number> vec_coarse;

  /**
   * Constraint matrix on coarse level.
   */
  AffineConstraints<Number> constraint_coarse;

  /**
   * Number of components.
   */
  unsigned int n_components;

  friend class internal::MGTwoLevelTransferImplementation;
};



/**
 * Implementation of the MGTransferBase. In contrast to
 * other multigrid transfer operators, the user can provide separate
 * transfer operators of type MGTwoLevelTransfer between each level.
 *
 * This class currently only works for tensor-product finite elements based on
 * FE_Q and FE_DGQ elements. Systems involving multiple components of
 * one of these element, as well as, systems with different elements or other
 * elements are currently not implemented.
 */
template <typename MatrixType, typename VectorType>
class MGTransferGlobalCoarsening : public dealii::MGTransferBase<VectorType>
{
public:
  static_assert(std::is_same<typename MatrixType::value_type,
                             typename VectorType::value_type>::value,
                "Types do not match.");

  /**
   * Dimension.
   */
  static const int dim = MatrixType::dim;

  /**
   * Value type.
   */
  using Number = typename MatrixType::value_type;

  /**
   * Constructor taking an operator for each level (minimum requirement is
   * that the operator provides the function initialize_dof_vector()) and
   * transfer operators (with the coarsest level kept empty in @p transfer).
   */
  MGTransferGlobalCoarsening(
    const MGLevelObject<MatrixType> &                         matrices,
    const MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> &transfer);

  /**
   * Perform prolongation.
   */
  void
  prolongate(const unsigned int to_level,
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

private:
  const MGLevelObject<MatrixType> &                         matrices;
  const MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> &transfer;
};



#ifndef DOXYGEN

/* ----------------------- Inline functions --------------------------------- */



template <int dim, typename Number>
template <typename Stream>
void
MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<Number>>::
  MGTransferScheme::print(Stream &out) const
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



template <int dim, typename Number>
template <typename Stream>
void
MGTwoLevelTransfer<dim, LinearAlgebra::distributed::Vector<Number>>::
  print_internal(Stream &out) const
{
  for (const auto &scheme : schemes)
    scheme.print(out);
}



template <typename MatrixType, typename VectorType>
MGTransferGlobalCoarsening<MatrixType, VectorType>::MGTransferGlobalCoarsening(
  const MGLevelObject<MatrixType> &                         matrices,
  const MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> &transfer)
  : matrices(matrices)
  , transfer(transfer)
{
  AssertDimension(matrices.max_level() - matrices.min_level(),
                  transfer.max_level() - transfer.min_level());
}



template <typename MatrixType, typename VectorType>
void
MGTransferGlobalCoarsening<MatrixType, VectorType>::prolongate(
  const unsigned int to_level,
  VectorType &       dst,
  const VectorType & src) const
{
  this->transfer[to_level].prolongate(dst, src);
}



template <typename MatrixType, typename VectorType>
void
MGTransferGlobalCoarsening<MatrixType, VectorType>::restrict_and_add(
  const unsigned int from_level,
  VectorType &       dst,
  const VectorType & src) const
{
  this->transfer[from_level].restrict_and_add(dst, src);
}



template <typename MatrixType, typename VectorType>
template <class InVector, int spacedim>
void
MGTransferGlobalCoarsening<MatrixType, VectorType>::copy_to_mg(
  const DoFHandler<dim, spacedim> &dof_handler,
  MGLevelObject<VectorType> &      dst,
  const InVector &                 src) const
{
  (void)dof_handler;

  for (unsigned int level = dst.min_level(); level <= dst.max_level(); ++level)
    matrices[level].initialize_dof_vector(dst[level]);

  dst[dst.max_level()].copy_locally_owned_data_from(src);
}



template <typename MatrixType, typename VectorType>
template <class OutVector, int spacedim>
void
MGTransferGlobalCoarsening<MatrixType, VectorType>::copy_from_mg(
  const DoFHandler<dim, spacedim> &dof_handler,
  OutVector &                      dst,
  const MGLevelObject<VectorType> &src) const
{
  (void)dof_handler;

  dst.copy_locally_owned_data_from(src[src.max_level()]);
}

#endif

DEAL_II_NAMESPACE_CLOSE

#endif
