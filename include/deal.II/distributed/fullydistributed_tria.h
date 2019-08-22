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

#ifndef dealii_fullydistributed_tria_h
#define dealii_fullydistributed_tria_h


#include <deal.II/base/config.h>

#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/p4est_wrappers.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/grid/tria.h>

#include <boost/range/iterator_range.hpp>

#include <functional>
#include <list>
#include <set>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef DEAL_II_WITH_MPI
#  include <mpi.h>
#endif

#ifdef DEAL_II_WITH_P4EST
#  include <p4est.h>
#  include <p4est_connectivity.h>
#  include <p4est_ghost.h>
#  include <p8est.h>
#  include <p8est_connectivity.h>
#  include <p8est_ghost.h>
#endif

DEAL_II_NAMESPACE_OPEN

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

struct Part_
{
  Part_()
  {}

  Part_(CellId::binary_type index,
        unsigned int        subdomain_id,
        unsigned int        level_subdomain_id)
    : index(index)
    , subdomain_id(subdomain_id)
    , level_subdomain_id(level_subdomain_id){};

  CellId::binary_type index;
  unsigned int        subdomain_id;
  unsigned int        level_subdomain_id;
};

class Part
{
public:
  std::vector<Part_> cells;
};

namespace parallel
{
  namespace fullydistributed
  {
    template <int dim, int spacedim>
    struct ConstructionData
    {
      // information describing the local part of the coarse grid
      std::vector<CellData<dim>>      cells;
      std::vector<Point<spacedim>>    vertices;
      std::vector<types::boundary_id> boundary_ids;

      // information
      std::map<int, int> coarse_lid_to_gid;

      // information describing how to constuct the levels
      std::vector<Part> parts;
    };

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

      void
      reinit(ConstructionData<dim, spacedim> &construction_data);

      virtual void
      create_triangulation(const std::vector<Point<spacedim>> &vertices,
                           const std::vector<CellData<dim>> &  cells,
                           const SubCellData &subcelldata) override;


      enum Settings
      {
        /**
         * Default settings, other options are disabled.
         */
        default_setting = 0x0,
        /**
         * This flags needs to be set to use the geometric multigrid
         * functionality. This option requires additional computation and
         * communication. Note: geometric multigrid is still a work in
         * progress.
         */
        construct_multigrid_hierarchy = 0x1
      };


      Triangulation(MPI_Comm mpi_communicator,
                    Settings settings_ = default_setting);

      Triangulation(MPI_Comm mpi_communicator,
                    MPI_Comm mpi_communicator_coarse,
                    Settings settings_ = default_setting);

      virtual ~Triangulation();

      virtual void
      clear() override;

      void
      copy_local_forest_to_triangulation();

      virtual void
      execute_coarsening_and_refinement() override;

      virtual bool
      prepare_coarsening_and_refinement() override;

      virtual bool
      has_hanging_nodes() const override;

      virtual std::size_t
      memory_consumption() const override;

      virtual void
      add_periodicity(
        const std::vector<GridTools::PeriodicFacePair<cell_iterator>> &)
        override;

      virtual void
      update_number_cache() override;

      virtual std::map<unsigned int, std::set<dealii::types::subdomain_id>>
      compute_vertices_with_ghost_neighbors() const override;

      bool
      is_multilevel_hierarchy_constructed() const override;

      MPI_Comm
      get_coarse_communicator() const;

      virtual unsigned int
      coarse_cell_id_to_coarse_cell_index(
        const types::coarse_cell_id coarse_cell_id) const override;

      virtual types::coarse_cell_id
      coarse_cell_index_to_coarse_cell_id(
        const unsigned int coarse_cell_index) const override;

    private:
      /**
       * store the Settings.
       */
      Settings settings;

      template <typename>
      friend class dealii::internal::DoFHandlerImplementation::Policy::
        ParallelDistributed;

      template <int, int, class>
      friend class dealii::FETools::internal::ExtrapolateImplementation;

      std::vector<std::pair<types::coarse_cell_id, unsigned int>>
        coarse_gid_to_lid;
      std::vector<std::pair<unsigned int, types::coarse_cell_id>>
        coarse_lid_to_gid;

      MPI_Comm mpi_communicator_coarse;
    };

  } // namespace fullydistributed
} // namespace parallel


DEAL_II_NAMESPACE_CLOSE

#endif
