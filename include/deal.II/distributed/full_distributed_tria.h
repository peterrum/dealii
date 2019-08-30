// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2019 by the deal.II authors
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

#ifndef dealii_full_distributed_tria_h
#define dealii_full_distributed_tria_h


#include <deal.II/base/config.h>

#include <deal.II/distributed/tria_base.h>

#include <vector>

#ifdef DEAL_II_WITH_MPI
#  include <mpi.h>
#endif

DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
namespace internal
{
  namespace DoFHandlerImplementation
  {
    namespace Policy
    {
      template <typename>
      class ParallelDistributed;
    } // namespace Policy
  }   // namespace DoFHandlerImplementation
} // namespace internal

namespace FETools
{
  namespace internal
  {
    template <int, int, class>
    class ExtrapolateImplementation;
  }
} // namespace FETools

// forward declaration of the data type for periodic face pairs
namespace GridTools
{
  template <typename CellIterator>
  struct PeriodicFacePair;
}
#endif

namespace parallel
{
  /**
   * A namespace for the fully distributed Triangulation.
   *
   * @ingroup parallel
   */
  namespace fullydistributed
  {
    /**
     * Information needed for each locally relevant cell, stored in
     * ConstructionData and used during construction of a
     * parallel::fullydistributed::Triangulation. This struct stores
     * the cell id, the subdomain_id and the level_subdomain_id as well as
     * information related to manifold_id and boundary_id.
     *
     * @author Peter Munch, 2019
     */
    template <int dim>
    struct CellInfo
    {
      /**
       * Constructor.
       */
      CellInfo() = default;

      /**
       * Unique CellID of the cell.
       */
      CellId::binary_type id;

      /**
       * subdomain_id of the cell.
       */
      types::subdomain_id subdomain_id;

      /**
       * level_subdomain_id of the cell.
       */
      types::subdomain_id level_subdomain_id;

      /**
       * Manifold id of the cell.
       */
      types::material_id manifold_id;

      /**
       * Manifold id of all lines of the cell.
       *
       * @note Only used for 2D and 3D.
       */
      std::array<types::material_id, GeometryInfo<dim>::lines_per_cell>
        manifold_line_ids;

      /**
       * Manifold id of all faces of the cell.
       *
       * @note Only used for 3D.
       */
      std::array<types::material_id, GeometryInfo<dim>::quads_per_cell>
        manifold_quad_ids;

      /**
       * List of face number and boundary id of all non-internal faces of the
       * cell.
       */
      std::vector<std::pair<unsigned int, types::boundary_id>> boundary_ids;
    };



    /**
     * Data used to construct a parallel::fullydistributed::Triangulation.
     *
     * @author Peter Munch, 2019
     */
    template <int dim, int spacedim>
    struct ConstructionData
    {
      /**
       * Cells of the locally-relevant coarse-grid triangulation.
       */
      std::vector<CellData<dim>> coarse_cells;

      /**
       * Vertices of the locally-relevant coarse-grid triangulation.
       */
      std::vector<Point<spacedim>> coarse_cell_vertices;

      /**
       * Mapping from coarse-grid index to coarse-grid id.
       */
      std::vector<types::coarse_cell_id> coarse_cell_index_to_coarse_cell_id;

      /**
       * CellInfo for each locally relevant cell on each level.
       */
      std::vector<std::vector<CellInfo<dim>>> cell_infos;
    };


    /**
     * A distributed triangulation with a distributed coarse grid.
     *
     * The motivation for parallel::fullydistributed::Triangulation has its
     * origins in the following observations about complex geometries and/or
     * about given meshes created by an external mesh generator. We regard
     * complex geometries as geometries that can be meshed only with a
     * non-negligible number of coarse cells (>10,000):
     * - storing the coarse-grid information on every process is too expensive
     *   from a memory point of view (as done by
     *   parallel::distributed::Triangulation). Normally, a process only needs a
     *   small section of the global triangulation, i.e., a small section of the
     *   coarse grid such that a partitioning of the coarse grid is indeed
     *   essential. We call cells needed by a process locally relevant cells.
     * - the distribution of the active cells - on the finest level - among all
     *   processes by simply partitioning a space-filling curve might not lead
     *   to an optimal result for triangulations that originate from large
     *   coarse grids: e.g. partitions that belong to the same process might
     *   be discontinuous, leading to increased communication amount (within a
     *   node and beyond). Graph-based partitioning algorithms might be a sound
     *   alternative.
     *
     * To be able to construct a fully partitioned triangulation that
     * distributes the coarse grid and gives flexibility regarding partitioning,
     *  the following ingredients are required:
     * - a locally relevant coarse-grid triangulation
     *   (vertices, cell definition; including a layer of ghost cells)
     * - a mapping of the locally relevant coarse-grid triangulation into the
     *   global coarse-grid triangulation
     * - information about which cell should be refined as well as information
     *   regarding the subdomain_id, the level_subdomain_id, manifold_id,
     *   and boundary_id of each cell.
     *
     * The ingredients listed above are bundled in the struct
     * parallel::fullydistributed::ConstructionData. The user has to fill this
     * data structure - in a pre-processing step - before actually creating the
     * triangulation. Predefined functions to create ConstructionData
     * can be found in the namespace dealii::fullydistributed::Util.
     *
     * Once the ConstructionData `construction_data` has been constructed, the
     * triangulation `tria` can be created by calling
     * `tria.reinit(construction_data);`.
     *
     * @note This triangulation supports: 1D/2D/3D, hanging nodes,
     *       geometric multigrid, and periodicity.
     *
     * @note Currently no modifications of the triangulation is supported after
     *       it has been created, i.e., this type of mesh does not support
     *       any form of adaptivity, not even simple global refinements and
     *       coarsenings.
     *
     * @note Currently only simple periodicity conditions are supported.
     *
     * @author Peter Munch, 2019
     */
    template <int dim, int spacedim = dim>
    class Triangulation
      : public parallel::DistributedTriangulationBase<dim, spacedim>
    {
    public:
      typedef typename dealii::Triangulation<dim, spacedim>::cell_iterator
        cell_iterator;

      typedef
        typename dealii::Triangulation<dim, spacedim>::active_cell_iterator
          active_cell_iterator;

      typedef
        typename dealii::Triangulation<dim, spacedim>::CellStatus CellStatus;

      /**
       * Configuration flags for fully distributed Triangulations to be set in
       * the constructor. Settings can be combined using bitwise OR.
       */
      enum Settings
      {
        /**
         * Default settings, other options are disabled.
         */
        default_setting = 0x0,
        /**
         * This flags needs to be set to use the geometric multigrid
         * functionality. This option requires additional computation and
         * communication.
         */
        construct_multigrid_hierarchy = 0x1
      };


      /**
       * Constructor.
       *
       * @param mpi_communicator denotes the MPI communicator to be used for the
       *                         triangulation.
       * @param settings See the description of the Settings enumerator.
       */
      explicit Triangulation(MPI_Comm       mpi_communicator,
                             const Settings settings = default_setting);

      /**
       * Destructor.
       */
      virtual ~Triangulation() = default;

      /**
       * Create a triangulation from the provided ConstructionData.
       *
       * @note The namespace dealii::fullydistributed::Util contains functions
       *       to create ConstructionData.
       *
       * @note This is the function to be used instead of create_triangulation.
       *
       * @param construction_data
       */
      void
      reinit(const ConstructionData<dim, spacedim> &construction_data);

      /**
       * @note This function is not implemented for this class. Instead, use
       *       the function reinit() to create the triangulation.
       */
      virtual void
      create_triangulation(const std::vector<Point<spacedim>> &vertices,
                           const std::vector<CellData<dim>> &  cells,
                           const SubCellData &subcelldata) override;

      /**
       * Coarsen and refine the mesh according to refinement and coarsening
       * flags set.
       *
       * @note Not implemented yet.
       */
      virtual void
      execute_coarsening_and_refinement() override;

      /**
       * Override the implementation of prepare_coarsening_and_refinement from
       * the base class.
       *
       * @note Not implemented yet.
       */
      virtual bool
      prepare_coarsening_and_refinement() override;

      /**
       * Return true if the triangulation has hanging nodes.
       *
       * @note Not implemented yet.
       */
      virtual bool
      has_hanging_nodes() const override;

      /**
       * Return the local memory consumption in bytes.
       */
      virtual std::size_t
      memory_consumption() const override;

      /**
       * Override the function to update the number cache so we can fill data
       * like level_ghost_owners.
       */
      virtual void
      update_number_cache() override;

      /**
       * This function determines the neighboring subdomains that are adjacent
       *  to each vertex.
       */
      virtual std::map<unsigned int, std::set<dealii::types::subdomain_id>>
      compute_vertices_with_ghost_neighbors() const override;

      /**
       * @copydoc dealii::parallel::DistributedTriangulationBase::is_multilevel_hierarchy_constructed
       */
      virtual bool
      is_multilevel_hierarchy_constructed() const override;

      /**
       * @copydoc dealii::coarse_cell_id_to_coarse_cell_index
       */
      virtual unsigned int
      coarse_cell_id_to_coarse_cell_index(
        const types::coarse_cell_id coarse_cell_id) const override;

      /**
       * @copydoc dealii::coarse_cell_index_to_coarse_cell_id
       */
      virtual types::coarse_cell_id
      coarse_cell_index_to_coarse_cell_id(
        const unsigned int coarse_cell_index) const override;

    private:
      /**
       * store the Settings.
       */
      const Settings settings;

      /**
       * Sorted list of pairs of coarse-cell ids and their indices.
       */
      std::vector<std::pair<types::coarse_cell_id, unsigned int>>
        coarse_cell_id_to_coarse_cell_index_vector;

      /**
       * List of coarse-cell ids for each coarse cell (stored at cell->index()).
       */
      std::vector<types::coarse_cell_id>
        coarse_cell_index_to_coarse_cell_id_vector;

      bool create_triangulation_for_internal_usage;
      bool prepare_coarsening_and_refinement_for_internal_usage;


      template <typename>
      friend class dealii::internal::DoFHandlerImplementation::Policy::
        ParallelDistributed;

      template <int, int, class>
      friend class dealii::FETools::internal::ExtrapolateImplementation;
    };

  } // namespace fullydistributed
} // namespace parallel


DEAL_II_NAMESPACE_CLOSE

#endif
