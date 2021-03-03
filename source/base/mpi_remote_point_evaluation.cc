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

#include <deal.II/base/config.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/mpi_consensus_algorithms.templates.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

DEAL_II_NAMESPACE_OPEN


namespace Utilities
{
  namespace MPI
  {
    template <int dim, int spacedim>
    RemotePointEvaluation<dim, spacedim>::RemotePointEvaluation(
      const double tolerance)
      : tolerance(tolerance)
    {}

    template <int dim, int spacedim>
    void
    RemotePointEvaluation<dim, spacedim>::reinit(
      const std::vector<Point<spacedim>> &points,
      const Triangulation<dim, spacedim> &tria,
      const Mapping<dim, spacedim> &      mapping)
    {
#ifndef DEAL_II_WITH_MPI
      Assert(false, ExcNeedsMPI());
      (void)points;
      (void)tria;
      (void)mapping;
#else
      this->tria    = &tria;
      this->mapping = &mapping;

      comm = internal::get_mpi_comm(tria);

      std::vector<BoundingBox<spacedim>> local_boxes;
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          local_boxes.push_back(mapping.get_bounding_box(cell));

      // create r-tree of bounding boxes
      const auto local_tree = pack_rtree(local_boxes);

      // compress r-tree to a minimal set of bounding boxes
      const auto local_reduced_box = extract_rtree_level(local_tree, 0);

      // gather bounding boxes of other processes
      const auto global_bboxes =
        Utilities::MPI::all_gather(comm, local_reduced_box);

      const GridTools::Cache<dim, spacedim> cache(tria, mapping);

      const auto data = distributed_compute_point_locations_internal(
        cache, points, global_bboxes, tolerance, true);

      this->recv_ranks  = data.recv_ranks;
      this->indices_ptr = data.recv_ptrs;

      this->send_ranks = data.send_ranks;
      this->send_ptr   = data.send_ptrs;

      this->indices = {};
      this->indices.resize(data.recv_components.size());
      this->quadrature_points_ptr.assign(points.size() + 1, 0);
      for (unsigned int i = 0; i < data.recv_components.size(); ++i)
        {
          this->indices[std::get<2>(data.recv_components[i])] = i;
          this
            ->quadrature_points_ptr[std::get<2>(data.recv_components[i]) + 1]++;
        }

      unique_mapping = true;
      for (unsigned int i = 0; i < points.size(); ++i)
        {
          unique_mapping &= this->quadrature_points_ptr[i + 1] == 1;
          this->quadrature_points_ptr[i + 1] += this->quadrature_points_ptr[i];
        }

      relevant_remote_points_per_process = {};

      std::pair<int, int> dummy{-1, -1};
      for (const auto &i : data.send_components)
        {
          if (dummy != std::get<0>(i))
            {
              dummy = std::get<0>(i);
              std::get<0>(this->relevant_remote_points_per_process)
                .emplace_back(dummy, 0);
            }

          std::get<0>(this->relevant_remote_points_per_process).back().second++;
          std::get<1>(this->relevant_remote_points_per_process)
            .emplace_back(std::get<3>(i));
          std::get<2>(this->relevant_remote_points_per_process)
            .emplace_back(std::get<5>(i));
        }

      /*
      std::get<2>(this->relevant_remote_points_per_process).resize(data.send_components.size());
      for(unsigned int i = 0; i < data.send_components.size(); ++i)
        std::get<2>(this->relevant_remote_points_per_process)[std::get<5>(data.send_components[i])]
      = i;
      */


      std::cout << "----------------------------------------------------------"
                << std::endl;

      for (const auto i : quadrature_points_ptr)
        std::cout << i << " ";
      std::cout << std::endl;

      for (const auto i : indices)
        std::cout << i << " ";
      std::cout << std::endl;

      for (const auto i : indices_ptr)
        std::cout << i << " ";
      std::cout << std::endl;

      for (const auto i : recv_ranks)
        std::cout << i << " ";
      std::cout << std::endl;

      for (const auto i : send_ranks)
        std::cout << i << " ";
      std::cout << std::endl;

      for (const auto i : send_ptr)
        std::cout << i << " ";
      std::cout << std::endl;

      for (const auto i : std::get<0>(relevant_remote_points_per_process))
        std::cout << "(" << i.first.first << ", " << i.first.second << ", "
                  << i.second << "), ";
      std::cout << std::endl;

      for (const auto i : std::get<1>(relevant_remote_points_per_process))
        std::cout << i << " ";
      std::cout << std::endl;

      for (const auto i : std::get<2>(relevant_remote_points_per_process))
        std::cout << i << " ";
      std::cout << std::endl;
#endif
    }


    template <int dim, int spacedim>
    const std::vector<unsigned int> &
    RemotePointEvaluation<dim, spacedim>::get_point_ptrs() const
    {
      return quadrature_points_ptr;
    }

    template <int dim, int spacedim>
    bool
    RemotePointEvaluation<dim, spacedim>::is_map_unique() const
    {
      return unique_mapping;
    }

    template <int dim, int spacedim>
    const Triangulation<dim, spacedim> &
    RemotePointEvaluation<dim, spacedim>::get_triangulation() const
    {
      return *tria;
    }

    template <int dim, int spacedim>
    const Mapping<dim, spacedim> &
    RemotePointEvaluation<dim, spacedim>::get_mapping() const
    {
      return *mapping;
    }

    template <int dim, int spacedim>
    bool
    RemotePointEvaluation<dim, spacedim>::is_ready() const
    {
      return true; // TODO
    }

  } // end of namespace MPI
} // end of namespace Utilities

#include "mpi_remote_point_evaluation.inst"

DEAL_II_NAMESPACE_CLOSE
