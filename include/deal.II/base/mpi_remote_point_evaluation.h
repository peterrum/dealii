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

#ifndef dealii_mpi_mpi_remote_point_evaluation_h
#define dealii_mpi_mpi_remote_point_evaluation_h

#include <deal.II/base/config.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/mpi_consensus_algorithms.templates.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

DEAL_II_NAMESPACE_OPEN


namespace Utilities
{
  namespace MPI
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

    /**
     * TODO
     */
    template <int dim, int spacedim = dim>
    class RemotePointEvaluation
    {
    public:
      RemotePointEvaluation(const double tolerance = 1e-6)
        : tolerance(tolerance)
      {}

      void
      reinit(const std::vector<Point<spacedim>>  quadrature_points,
             const Triangulation<dim, spacedim> &tria,
             const Mapping<dim, spacedim> &      mapping)
      {
        this->tria    = &tria;
        this->mapping = &mapping;

        comm = get_mpi_comm(tria);

        // create bounding boxed of local active cells
        std::vector<BoundingBox<spacedim>> local_boxes;
        for (const auto cell : tria.active_cell_iterators())
          if (cell->is_locally_owned())
            local_boxes.push_back(mapping.get_bounding_box(cell));

        // create r-tree of bounding boxes
        const auto local_tree = pack_rtree(local_boxes);

        // compress r-tree to a minimal set of bounding boxes
        const auto local_reduced_box = extract_rtree_level(local_tree, 0);

        // gather bounding boxes of other processes
        const auto global_bounding_boxes =
          Utilities::MPI::all_gather(comm, local_reduced_box);

        // determine ranks which might poses quadrature point
        auto potentially_relevant_points_per_process =
          std::vector<std::vector<Point<spacedim>>>(
            global_bounding_boxes.size());

        auto potentially_relevant_points_per_process_id =
          std::vector<std::vector<unsigned int>>(global_bounding_boxes.size());

        for (unsigned int i = 0; i < quadrature_points.size(); ++i)
          {
            const auto &point = quadrature_points[i];
            for (unsigned rank = 0; rank < global_bounding_boxes.size(); ++rank)
              for (const auto &box : global_bounding_boxes[rank])
                if (box.point_inside(point))
                  {
                    potentially_relevant_points_per_process[rank].emplace_back(
                      point);
                    potentially_relevant_points_per_process_id[rank]
                      .emplace_back(i);
                    break;
                  }
          }

        // TODO
        std::map<unsigned int,
                 std::vector<std::pair<std::pair<int, int>, Point<spacedim>>>>
          relevant_remote_points_per_process;

        // only communicate with processes that might have a quadrature point
        std::vector<unsigned int> targets;

        for (unsigned int i = 0;
             i < potentially_relevant_points_per_process.size();
             ++i)
          if (potentially_relevant_points_per_process[i].size() > 0 &&
              i != Utilities::MPI::this_mpi_process(comm))
            targets.emplace_back(i);


        std::map<unsigned int, std::vector<unsigned int>>
          relevant_points_per_process_offset;
        std::map<unsigned int, std::vector<unsigned int>>
          relevant_points_per_process_count;



        const std::vector<bool>          marked_vertices;
        const GridTools::Cache<dim, dim> cache(tria, mapping);
        auto                             cell_hint = tria.begin_active();

        const auto find_all_locally_owned_active_cells_around_point =
          [&](const Point<dim> &point) {
            std::vector<
              std::pair<typename Triangulation<dim>::active_cell_iterator,
                        Point<dim>>>
              locally_owned_active_cells_around_point;

            try
              {
                const auto first_cell =
                  GridTools::find_active_cell_around_point(
                    cache, point, cell_hint, marked_vertices, tolerance);

                cell_hint = first_cell.first;

                const auto active_cells_around_point =
                  GridTools::find_all_active_cells_around_point(
                    mapping, tria, point, tolerance, first_cell);

                locally_owned_active_cells_around_point.reserve(
                  active_cells_around_point.size());

                for (const auto &cell : active_cells_around_point)
                  if (cell.first->is_locally_owned())
                    locally_owned_active_cells_around_point.push_back(cell);
              }
            catch (...)
              {}

            return locally_owned_active_cells_around_point;
          };


        // for local quadrature points no communication is needed...
        const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

        {
          const auto &potentially_relevant_points =
            potentially_relevant_points_per_process[my_rank];
          const auto &potentially_relevant_points_offset =
            potentially_relevant_points_per_process_id[my_rank];

          std::vector<std::pair<std::pair<int, int>, Point<spacedim>>> points;
          std::vector<unsigned int> point_ids;
          std::vector<unsigned int> point_counts;

          for (unsigned int j = 0; j < potentially_relevant_points.size(); ++j)
            {
              const auto adjacent_cells =
                find_all_locally_owned_active_cells_around_point(
                  potentially_relevant_points[j]);

              if (adjacent_cells.size() > 0)
                {
                  for (auto cell : adjacent_cells)
                    points.emplace_back(std::make_pair(cell.first->level(),
                                                       cell.first->index()),
                                        cell.second);

                  point_ids.push_back(potentially_relevant_points_offset[j]);
                  point_counts.push_back(adjacent_cells.size());
                }
            }


          if (points.size() > 0)
            {
              // to send
              relevant_remote_points_per_process[my_rank] = points;

              // to recv
              relevant_points_per_process_offset[my_rank] = point_ids;
              relevant_points_per_process_count[my_rank]  = point_counts;
            }
        }

        // send to remote ranks the requested quadrature points and eliminate
        // not needed ones (note: currently, we cannot communicate points ->
        // switch to doubles here)
        Utilities::MPI::ConsensusAlgorithms::AnonymousProcess<double,
                                                              unsigned int>
          process(
            [&]() { return targets; },
            [&](const unsigned int   other_rank,
                std::vector<double> &send_buffer) {
              // send requested points
              for (auto point :
                   potentially_relevant_points_per_process[other_rank])
                for (unsigned int i = 0; i < spacedim; ++i)
                  send_buffer.emplace_back(point[i]);
            },
            [&](const unsigned int &       other_rank,
                const std::vector<double> &recv_buffer,
                std::vector<unsigned int> &request_buffer) {
              // received points, determine if point is actually possessed,
              // and send the result back

              std::vector<std::pair<std::pair<int, int>, Point<spacedim>>>
                relevant_remote_points;

              request_buffer.clear();
              request_buffer.resize(recv_buffer.size() / spacedim);

              for (unsigned int i = 0, j = 0; i < recv_buffer.size();
                   i += spacedim, ++j)
                {
                  Point<spacedim> point;
                  for (unsigned int j = 0; j < spacedim; ++j)
                    point[j] = recv_buffer[i + j];

                  const auto adjacent_cells =
                    find_all_locally_owned_active_cells_around_point(point);

                  request_buffer[j] = adjacent_cells.size();

                  if (adjacent_cells.size() > 0)
                    {
                      for (auto cell : adjacent_cells)
                        relevant_remote_points.emplace_back(
                          std::make_pair(cell.first->level(),
                                         cell.first->index()),
                          cell.second);
                    }
                }

              if (relevant_remote_points.size() > 0)
                relevant_remote_points_per_process[other_rank] =
                  relevant_remote_points;
            },
            [&](const unsigned int         other_rank,
                std::vector<unsigned int> &recv_buffer) {
              // prepare buffer
              recv_buffer.resize(
                potentially_relevant_points_per_process[other_rank].size());
            },
            [&](const unsigned int               other_rank,
                const std::vector<unsigned int> &recv_buffer) {
              // store recv_buffer -> make the algorithm deterministic

              const auto &potentially_relevant_points =
                potentially_relevant_points_per_process[other_rank];
              const auto &potentially_relevant_points_offset =
                potentially_relevant_points_per_process_id[other_rank];

              AssertDimension(potentially_relevant_points.size(),
                              recv_buffer.size());
              AssertDimension(potentially_relevant_points_offset.size(),
                              recv_buffer.size());

              std::vector<unsigned int> point_ids;
              std::vector<unsigned int> point_counts;

              for (unsigned int i = 0; i < recv_buffer.size(); ++i)
                if (recv_buffer[i] > 0)
                  {
                    point_ids.push_back(potentially_relevant_points_offset[i]);
                    point_counts.push_back(recv_buffer[i]);
                  }

              if (point_ids.size() > 0)
                {
                  relevant_points_per_process_offset[other_rank] = point_ids;
                  relevant_points_per_process_count[other_rank]  = point_counts;

                  recv_ranks.push_back(other_rank);
                }
            });

        Utilities::MPI::ConsensusAlgorithms::Selector<double, unsigned int>(
          process, comm)
          .run();

        std::vector<unsigned int> quadrature_points_count(
          quadrature_points.size(), 0);
        std::vector<std::pair<unsigned int, unsigned int>> indices_temp;

        for (const auto &i : relevant_points_per_process_offset)
          {
            const unsigned int rank = i.first;


            const auto &relevant_points_offset =
              relevant_points_per_process_offset[rank];
            const auto &relevant_points_count =
              relevant_points_per_process_count[rank];

            for (unsigned int j = 0; j < relevant_points_offset.size(); ++j)
              for (unsigned int k = 0; k < relevant_points_count[j]; ++k)
                {
                  auto &qp_counter =
                    quadrature_points_count[relevant_points_offset[j]];
                  indices_temp.emplace_back(relevant_points_offset[j],
                                            qp_counter);

                  ++qp_counter;
                }
          }

        quadrature_points_ptr = {0};

        this->unique_mapping = true;
        for (const auto &i : quadrature_points_count)
          {
            this->unique_mapping &= (i == 1);
            quadrature_points_ptr.push_back(quadrature_points_ptr.back() + i);
          }

        indices = {};
        for (const auto i : indices_temp)
          indices.push_back(quadrature_points_ptr[i.first] + i.second);

        this->relevant_remote_points_per_process = {};

        send_ranks = {};
        send_ptr   = {0};

        for (const auto &i : relevant_remote_points_per_process)
          {
            auto &temp = this->relevant_remote_points_per_process;

            std::vector<
              std::tuple<std::pair<int, int>, unsigned int, Point<spacedim>>>
              full;
            full.reserve(i.second.size());

            for (const auto &j : i.second)
              full.emplace_back(j.first, full.size(), j.second);

            std::sort(full.begin(),
                      full.end(),
                      [](const auto &a, const auto &b) {
                        if (std::get<0>(a) != std::get<0>(b))
                          return std::get<0>(a) < std::get<0>(b);
                        else
                          return std::get<1>(a) < std::get<1>(b);
                      });

            std::pair<int, int> current = {-1, -1};

            for (const auto j : full)
              {
                if (current != std::get<0>(j))
                  {
                    current = std::get<0>(j);
                    std::get<0>(temp).emplace_back(current, 0);
                  }

                std::get<0>(temp).back().second++;

                std::get<1>(temp).push_back(std::get<2>(j));
                std::get<2>(temp).push_back(std::get<1>(j) + send_ptr.back());
              }

            send_ranks.push_back(i.first);
            send_ptr.push_back(std::get<2>(temp).size());
          }
      }

      /**
       * Evaluate function @p fu in the requested quadrature points. The result
       * is sorted according to rank.
       */
      template <typename T>
      void
      process(std::vector<T> &                    output,
              std::vector<T> &                    input,
              const std::function<void(
                std::vector<T> &,
                const std::tuple<
                  std::vector<std::pair<std::pair<int, int>, unsigned int>>,
                  std::vector<Point<spacedim>>,
                  std::vector<unsigned int>> &)> &fu) const
      {
        output.resize(quadrature_points_ptr.back());
        input.resize(
          (std::get<1>(this->relevant_remote_points_per_process).size()));

        fu(input, relevant_remote_points_per_process);

        // process remote quadrature points and send them away
        std::map<unsigned int, std::vector<char>> temp_map;

        std::vector<MPI_Request> requests;
        requests.reserve(send_ranks.size());

        const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

        // receive result

        std::map<unsigned int, std::vector<T>> temp_recv_map;

        for (unsigned int i = 0; i < send_ranks.size(); ++i)
          {
            if (send_ranks[i] == my_rank)
              {
                // process locally-owned values
                temp_recv_map[my_rank] =
                  std::vector<T>(input.begin() + send_ptr[i],
                                 input.begin() + send_ptr[i + 1]);
                continue;
              }

            temp_map[send_ranks[i]] =
              Utilities::pack(std::vector<T>(input.begin() + send_ptr[i],
                                             input.begin() + send_ptr[i + 1]));

            auto &buffer = temp_map[send_ranks[i]];

            requests.resize(requests.size() + 1);

            MPI_Isend(buffer.data(),
                      buffer.size(),
                      MPI_CHAR,
                      send_ranks[i],
                      11,
                      comm,
                      &requests.back());
          }

        for (unsigned int i = 0; i < recv_ranks.size(); ++i)
          {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, 11, comm, &status);

            int message_length;
            MPI_Get_count(&status, MPI_CHAR, &message_length);

            std::vector<char> buffer(message_length);

            MPI_Recv(buffer.data(),
                     buffer.size(),
                     MPI_CHAR,
                     status.MPI_SOURCE,
                     11,
                     comm,
                     MPI_STATUS_IGNORE);

            temp_recv_map[status.MPI_SOURCE] =
              Utilities::unpack<std::vector<T>>(buffer);
          }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

        auto it = indices.begin();
        for (const auto &j : temp_recv_map)
          for (const auto &i : j.second)
            output[*(it++)] = i;
      }


      const std::vector<unsigned int> &
      get_quadrature_points_ptr() const
      {
        return quadrature_points_ptr;
      }

      bool
      is_unique_mapping() const
      {
        return unique_mapping;
      }

      const Triangulation<dim, spacedim> &
      get_triangulation() const
      {
        return *tria;
      }
      const Mapping<dim, spacedim> &
      get_mapping() const
      {
        return *mapping;
      }

      bool
      is_ready() const
      {
        return true; // TODO
      }

    private:
      const double tolerance;

      SmartPointer<const Triangulation<dim, spacedim>> tria;
      SmartPointer<const Mapping<dim, spacedim>>       mapping;

      MPI_Comm comm;

      bool unique_mapping;

      // receiver side
      std::vector<unsigned int> quadrature_points_ptr;
      std::vector<unsigned int> indices;

      std::vector<unsigned int> recv_ranks;

      // sender side (TODO: merge)
      std::tuple<std::vector<std::pair<std::pair<int, int>, unsigned int>>,
                 std::vector<Point<spacedim>>,
                 std::vector<unsigned int>>
        relevant_remote_points_per_process;

      std::vector<unsigned int> send_ranks;
      std::vector<unsigned int> send_ptr;
    };

  } // end of namespace MPI
} // end of namespace Utilities


DEAL_II_NAMESPACE_CLOSE

#endif
