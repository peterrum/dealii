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
    namespace
    {
      template <typename MeshType>
      MPI_Comm
      get_mpi_comm(const MeshType &mesh)
      {
        const auto *tria_parallel = dynamic_cast<
          const parallel::TriangulationBase<MeshType::dimension,
                                            MeshType::space_dimension> *>(
          &(mesh.get_triangulation()));

        return tria_parallel != nullptr ? tria_parallel->get_communicator() :
                                          MPI_COMM_SELF;
      }
    } // namespace

    template <int dim, int spacedim>
    RemotePointEvaluation<dim, spacedim>::RemotePointEvaluation(
      const double tolerance)
      : tolerance(tolerance)
    {}

    template <int dim, int spacedim>
    void
    RemotePointEvaluation<dim, spacedim>::reinit(
      const std::vector<Point<spacedim>> &quadrature_points,
      const Triangulation<dim, spacedim> &tria,
      const Mapping<dim, spacedim> &      mapping)
    {
#ifndef DEAL_II_WITH_MPI
      Assert(false, ExcNeedsMPI());
      (void)quadrature_points;
      (void)tria;
      (void)mapping;
#else
      this->tria    = &tria;
      this->mapping = &mapping;

      comm = get_mpi_comm(tria);

      const GridTools::Cache<dim, spacedim> cache(tria, mapping);

      // create bounding boxed of local active cells
      std::vector<BoundingBox<spacedim>> local_boxes;
      for (const auto &cell : tria.active_cell_iterators())
        if (cell->is_locally_owned())
          local_boxes.push_back(mapping.get_bounding_box(cell));

      // create r-tree of bounding boxes
      const auto local_tree = pack_rtree(local_boxes);

      // compress r-tree to a minimal set of bounding boxes
      const auto local_reduced_box = extract_rtree_level(local_tree, 0);

      // gather bounding boxes of other processes
      const auto global_bounding_boxes =
        Utilities::MPI::all_gather(comm, local_reduced_box);

      const auto temp =
        dealii::GridTools::distributed_compute_point_locations<dim, spacedim>(
          cache, quadrature_points, global_bounding_boxes);

      const auto &cells         = std::get<0>(temp);
      const auto &local_points  = std::get<1>(temp);
      const auto &local_indices = std::get<2>(temp);
      const auto &from_cpu      = std::get<4>(temp);

      for (unsigned int i = 0; i < cells.size(); ++i)
        {
          std::get<0>(relevant_remote_points_per_process)
            .emplace_back(std::pair<int, int>(cells[i]->level(),
                                              cells[i]->index()),
                          local_points[i].size());
          std::get<1>(relevant_remote_points_per_process)
            .insert(std::get<1>(relevant_remote_points_per_process).end(),
                    local_points[i].begin(),
                    local_points[i].end());
        }

      std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>
        send_indices_sort;

      for (unsigned int i = 0, c = 0; i < local_points.size(); ++i)
        for (unsigned int j = 0; j < local_points[i].size(); ++j, ++c)
          send_indices_sort.emplace_back(from_cpu[i][j],
                                         local_indices[i][j],
                                         c);

      std::sort(send_indices_sort.begin(), send_indices_sort.end());

      std::map<unsigned int, std::vector<unsigned int>> send_map;

      for (const auto &i : send_indices_sort)
        {
          std::get<2>(relevant_remote_points_per_process)
            .push_back(std::get<2>(i));
          send_map[std::get<0>(i)].push_back(std::get<1>(i));
        }

      send_ptr = {0};

      for (const auto &i : send_map)
        {
          send_ranks.push_back(i.first);
          send_ptr.push_back(i.second.size());
        }

      std::map<unsigned int, std::vector<unsigned int>> recv_map;

      Utilities::MPI::ConsensusAlgorithms::AnonymousProcess<unsigned int,
                                                            unsigned int>
        process([&]() { return send_ranks; },
                [&](const unsigned int         other_rank,
                    std::vector<unsigned int> &send_buffer) {
                  send_buffer = send_map[other_rank];
                },
                [&](const unsigned int &             other_rank,
                    const std::vector<unsigned int> &recv_buffer,
                    std::vector<unsigned int> &) {
                  recv_map[other_rank] = recv_buffer;
                });

      Utilities::MPI::ConsensusAlgorithms::Selector<unsigned int, unsigned int>(
        process, comm)
        .run();

      indices_ptr = {0};

      std::vector<std::pair<unsigned int, unsigned int>> recv_indices_sort;

      for (const auto &i : recv_map)
        {
          recv_ranks.push_back(i.first);
          indices_ptr.push_back(i.second.size());

          for (const auto &j : i.second)
            recv_indices_sort.emplace_back(j, recv_indices_sort.size());
        }

      std::sort(recv_indices_sort.begin(), recv_indices_sort.end());

      quadrature_points_ptr.assign(quadrature_points.size() + 1, 0);

      for (unsigned int i = 0; i < recv_indices_sort.size(); ++i)
        {
          indices.push_back(recv_indices_sort[i].second);
          quadrature_points_ptr[recv_indices_sort[i].first]++;
        }

      unique_mapping = true;

      for (unsigned int i = 0; i < quadrature_points.size(); ++i)
        {
          unique_mapping &= (quadrature_points_ptr[i + 1] == 1);
          quadrature_points_ptr[i + 1] += quadrature_points_ptr[i];
        }
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
