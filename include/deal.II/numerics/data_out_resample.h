// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
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

#ifndef dealii_data_out_resample_h
#define dealii_data_out_resample_h



#include <deal.II/base/config.h>

#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

#include <memory>

DEAL_II_NAMESPACE_OPEN

/**
 * A DataOut class that does not output the result on a numerical solution on
 * the cells of the original triangulation but interpolates the result onto a
 * second triangulation (that can be completely unrelated).
 * By using this class, one can output the result obtained on an unstructured
 * mesh onto a structured one or one can create a slice in 3D.
 *
 * @note While the dimension of the two triangulations might differ, their
 *   space dimension need to coincide.
 */
template <int dim, int patch_dim, int spacedim>
class DataOutResample
  : public DataOut_DoFData<dim, patch_dim, spacedim, spacedim>
{
public:
  /**
   * Constructor taking the triangulation and mapping for which the patches
   * should be generated.
   */
  DataOutResample(const Triangulation<patch_dim, spacedim> &patch_tria,
                  const Mapping<patch_dim, spacedim> &      patch_mapping);

  /**
   * Update the @p mapping of original triangulation. One needs to call this
   * function if the mapping has changed. Just like in the DataOut context,
   * @p n_subdivisions determines how many "patches" this function will build
   * out of every cell.
   *
   * @note If you use the version of build_patches() that does not take a
   *   mapping, this function has to be called before its first usage.
   */
  void
  update_mapping(const Mapping<dim, spacedim> &mapping,
                 const unsigned int            n_subdivisions = 0);

  /**
   * This is the central function of this class since it builds the list of
   * patches to be written by the low-level functions of the base class. A
   * patch is, in essence, some intermediate representation of the data on
   * each cell of a triangulation and DoFHandler object that can then be used
   * to write files in some format that is readable by visualization programs.
   *
   * Just like in the DataOut context,  @p n_subdivisions determines how many
   * "patches" this function will build out of every cell.
   */
  void
  build_patches(
    const Mapping<dim, spacedim> &mapping,
    const unsigned int            n_subdivisions = 0,
    const typename DataOut<patch_dim, spacedim>::CurvedCellRegion
      curved_region =
        DataOut<patch_dim, spacedim>::CurvedCellRegion::curved_boundary);

  /**
   * Just like the above function, this function builds a list of
   * patches to be written by the low-level functions of the base class.
   * However it skips the update of the mapping and reuses the one registered
   * via update_mapping(). This allows to skip the expensive setup of the
   * internal communication routines.
   *
   * @note This function can be only used if a mapping has been registered via
   *   update_mapping() or the other build_patches() function.
   */
  void
  build_patches(
    const typename DataOut<patch_dim, spacedim>::CurvedCellRegion
      curved_region =
        DataOut<patch_dim, spacedim>::CurvedCellRegion::curved_boundary);

protected:
  virtual const std::vector<typename DataOutBase::Patch<patch_dim, spacedim>> &
  get_patches() const override;

private:
  /**
   * Intermediate DoFHandler
   */
  DoFHandler<patch_dim, spacedim> patch_dof_handler;

  /**
   * Mapping used in connection with patch_tria.
   */
  const SmartPointer<const Mapping<patch_dim, spacedim>> patch_mapping;

  /**
   * DataOut object that does the actual building of the patches.
   */
  DataOut<patch_dim, spacedim> patch_data_out;

  /**
   * Object to evaluate
   */
  Utilities::MPI::RemotePointEvaluation<dim, spacedim> rpe;

  /**
   * Partitioner to create internally distributed vectors.
   */
  std::shared_ptr<Utilities::MPI::Partitioner> partitioner;

  /**
   * Process local indices to access efficiently internal distributed vectors.
   */
  std::vector<types::global_dof_index> point_to_local_vector_indices;

  /**
   * Mapping of the original triangulation provided in update_mapping().
   */
  SmartPointer<const Mapping<dim, spacedim>> mapping;
};

DEAL_II_NAMESPACE_CLOSE

#endif
