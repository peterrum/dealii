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

// Integrate surface tension on surface mesh and test the result on background
// mesh.

#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/mpi_consensus_algorithms.templates.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_point_evaluation.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

using namespace dealii;


using VectorType = LinearAlgebra::distributed::Vector<double>;

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

template <int dim, int spacedim>
std::shared_ptr<const Utilities::MPI::Partitioner>
create_partitioner(const DoFHandler<dim, spacedim> &dof_handler)
{
  IndexSet locally_relevant_dofs;

  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  return std::make_shared<const Utilities::MPI::Partitioner>(
    dof_handler.locally_owned_dofs(),
    locally_relevant_dofs,
    get_mpi_comm(dof_handler));
}

namespace dealii
{
  namespace VectorTools
  {
    template <int dim, int spacedim, typename VectorType>
    void
    get_position_vector(const DoFHandler<dim, spacedim> &dof_handler_dim,
                        VectorType &                  euler_coordinates_vector,
                        const Mapping<dim, spacedim> &mapping)
    {
      FEValues<dim, spacedim> fe_eval(
        mapping,
        dof_handler_dim.get_fe(),
        Quadrature<dim>(dof_handler_dim.get_fe().get_unit_support_points()),
        update_quadrature_points);

      Vector<double> temp;

      for (const auto &cell : dof_handler_dim.active_cell_iterators())
        {
          if (cell->is_locally_owned() == false)
            continue;

          fe_eval.reinit(cell);

          temp.reinit(fe_eval.dofs_per_cell);

          for (const auto q : fe_eval.quadrature_point_indices())
            {
              const auto point = fe_eval.quadrature_point(q);

              const unsigned int comp =
                dof_handler_dim.get_fe().system_to_component_index(q).first;

              temp[q] = point[comp];
            }

          cell->set_dof_values(temp, euler_coordinates_vector);
        }

      euler_coordinates_vector.update_ghost_values();
    }
  } // namespace VectorTools
} // namespace dealii

template <int dim, int spacedim, typename VectorType>
void
compute_normal(const Mapping<dim, spacedim> &   mapping,
               const DoFHandler<dim, spacedim> &dof_handler_dim,
               VectorType &                     normal_vector)
{
  FEValues<dim, spacedim> fe_eval_dim(
    mapping,
    dof_handler_dim.get_fe(),
    dof_handler_dim.get_fe().get_unit_support_points(),
    update_normal_vectors | update_gradients);

  Vector<double> normal_temp;

  for (const auto &cell : dof_handler_dim.active_cell_iterators())
    {
      if (cell->is_locally_owned() == false)
        continue;

      fe_eval_dim.reinit(cell);

      normal_temp.reinit(fe_eval_dim.dofs_per_cell);
      normal_temp = 0.0;

      for (const auto q : fe_eval_dim.quadrature_point_indices())
        {
          const auto normal = fe_eval_dim.normal_vector(q);

          const unsigned int comp =
            dof_handler_dim.get_fe().system_to_component_index(q).first;

          normal_temp[q] = normal[comp];
        }

      cell->set_dof_values(normal_temp, normal_vector);
    }

  normal_vector.update_ghost_values();
}



template <int dim, int spacedim, typename VectorType>
void
compute_curvature(const Mapping<dim, spacedim> &   mapping,
                  const DoFHandler<dim, spacedim> &dof_handler_dim,
                  const DoFHandler<dim, spacedim> &dof_handler,
                  const Quadrature<dim>            quadrature,
                  const VectorType &               normal_vector,
                  VectorType &                     curvature_vector)
{
  FEValues<dim, spacedim> fe_eval(mapping,
                                  dof_handler.get_fe(),
                                  quadrature,
                                  update_gradients);
  FEValues<dim, spacedim> fe_eval_dim(mapping,
                                      dof_handler_dim.get_fe(),
                                      quadrature,
                                      update_gradients);

  Vector<double> curvature_temp;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned() == false)
        continue;

      TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
        &dof_handler_dim.get_triangulation(),
        cell->level(),
        cell->index(),
        &dof_handler_dim);

      fe_eval.reinit(cell);
      fe_eval_dim.reinit(dof_cell_dim);

      curvature_temp.reinit(quadrature.size());

      std::vector<std::vector<Tensor<1, spacedim, double>>> normal_gradients(
        quadrature.size(), std::vector<Tensor<1, spacedim, double>>(spacedim));

      fe_eval_dim.get_function_gradients(normal_vector, normal_gradients);

      for (const auto q : fe_eval_dim.quadrature_point_indices())
        {
          double curvature = 0.0;

          for (unsigned c = 0; c < spacedim; ++c)
            curvature += normal_gradients[q][c][c];

          curvature_temp[q] = curvature;
        }

      cell->set_dof_values(curvature_temp, curvature_vector);
    }

  curvature_vector.update_ghost_values();
}



template <int dim, int spacedim>
void
print(std::tuple<
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>,
      std::vector<std::vector<Point<dim>>>,
      std::vector<std::vector<unsigned int>>,
      std::vector<std::vector<Point<spacedim>>>,
      std::vector<std::vector<unsigned int>>> result)
{
  for (unsigned int i = 0; i < std::get<0>(result).size(); ++i)
    {
      const unsigned int n_points = std::get<1>(result)[i].size();

      std::cout << std::get<0>(result)[i]->level() << " "
                << std::get<0>(result)[i]->index() << " x " << n_points
                << std::endl;

      for (unsigned int j = 0; j < n_points; ++j)
        {
          std::cout << std::get<1>(result)[i][j] << " - "
                    << std::get<2>(result)[i][j] << " - "
                    << std::get<3>(result)[i][j] << " - "
                    << std::get<4>(result)[i][j] << std::endl;
        }
    }
}


namespace dealii
{
  namespace GridTools
  {
    template <int spacedim>
    std::vector<std::vector<std::pair<unsigned int, Point<spacedim>>>>
    guess_point_owner_new(
      const std::vector<std::vector<BoundingBox<spacedim>>> &global_bboxes,
      const std::vector<Point<spacedim>> &                   points)
    {
      auto potentially_relevant_points_per_process =
        std::vector<std::vector<std::pair<unsigned int, Point<spacedim>>>>(
          global_bboxes.size());

      for (unsigned int i = 0; i < points.size(); ++i)
        {
          const auto &point = points[i];
          for (unsigned rank = 0; rank < global_bboxes.size(); ++rank)
            for (const auto &box : global_bboxes[rank])
              if (box.point_inside(point))
                {
                  potentially_relevant_points_per_process[rank].emplace_back(
                    i, point);
                  break;
                }
        }
      return potentially_relevant_points_per_process;
    }

    template <int dim, int spacedim>
    std::vector<std::tuple<std::pair<int, int>,
                           unsigned int,
                           unsigned int,
                           Point<dim>,
                           Point<spacedim>>>
    distributed_compute_point_locations_internal(
      const GridTools::Cache<dim, spacedim> &                cache,
      const std::vector<Point<spacedim>> &                   points,
      const std::vector<std::vector<BoundingBox<spacedim>>> &global_bboxes,
      const double                                           tolerance,
      const bool                                             perform_handshake)
    {
      const auto potentially_relevant_points_per_process =
        GridTools::guess_point_owner_new(global_bboxes, points);

      const std::vector<bool> marked_vertices;
      auto cell_hint = cache.get_triangulation().begin_active();

      const auto find_all_locally_owned_active_cells_around_point =
        [&](const Point<spacedim> &point) {
          std::vector<std::pair<
            typename Triangulation<dim, spacedim>::active_cell_iterator,
            Point<dim>>>
            locally_owned_active_cells_around_point;

          try
            {
              const auto first_cell = GridTools::find_active_cell_around_point(
                cache, point, cell_hint, marked_vertices, tolerance);

              cell_hint = first_cell.first;

              const auto active_cells_around_point =
                GridTools::find_all_active_cells_around_point(
                  cache.get_mapping(),
                  cache.get_triangulation(),
                  point,
                  tolerance,
                  first_cell);

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

      std::vector<std::tuple<std::pair<int, int>,
                             unsigned int,
                             unsigned int,
                             Point<dim>,
                             Point<spacedim>>>
        all_send;

      std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>
        all_recv;

      Utilities::MPI::ConsensusAlgorithms::AnonymousProcess<char, char> process(
        [&]() {
          std::vector<unsigned int> targets;

          for (unsigned int i = 0;
               i < potentially_relevant_points_per_process.size();
               ++i)
            if (potentially_relevant_points_per_process[i].size() > 0)
              targets.emplace_back(i);
          return targets;
        },
        [&](const unsigned int other_rank, std::vector<char> &send_buffer) {
          send_buffer =
            Utilities::pack(potentially_relevant_points_per_process[other_rank],
                            false);
        },
        [&](const unsigned int &     other_rank,
            const std::vector<char> &recv_buffer,
            std::vector<char> &      request_buffer) {
          const auto recv_buffer_unpacked = Utilities::unpack<
            std::vector<std::pair<unsigned int, Point<spacedim>>>>(recv_buffer,
                                                                   false);

          for (unsigned int i = 0; i < recv_buffer_unpacked.size(); ++i)
            {
              const auto &index_and_point = recv_buffer_unpacked[i];

              const auto cells_and_reference_positions =
                find_all_locally_owned_active_cells_around_point(
                  index_and_point.second);

              for (const auto &cell_and_reference_position :
                   cells_and_reference_positions)
                {
                  all_send.emplace_back(
                    std::pair<int, int>(
                      cell_and_reference_position.first->level(),
                      cell_and_reference_position.first->index()),
                    other_rank,
                    index_and_point.first,
                    cell_and_reference_position.second,
                    index_and_point.second);
                }

              if (perform_handshake)
                request_buffer[i] = cells_and_reference_positions.size();
            }
        },
        [&](const unsigned int other_rank, std::vector<char> &recv_buffer) {
          if (perform_handshake)
            {
              recv_buffer = Utilities::pack(
                std::vector<unsigned int>(
                  potentially_relevant_points_per_process[other_rank].size()),
                false);
            }
        },
        [&](const unsigned int       other_rank,
            const std::vector<char> &recv_buffer) {
          if (perform_handshake)
            {
              const auto recv_buffer_unpacked =
                Utilities::unpack<std::vector<unsigned int>>(recv_buffer);
              const auto &potentially_relevant_points =
                potentially_relevant_points_per_process[other_rank];

              for (unsigned int i = 0; i < recv_buffer_unpacked.size(); ++i)
                for (unsigned int j = 0; j < recv_buffer_unpacked[i]; ++j)
                  all_recv.emplace_back(other_rank,
                                        potentially_relevant_points[i].first,
                                        numbers::invalid_unsigned_int);
            }
        });

      Utilities::MPI::ConsensusAlgorithms::Selector<char, char>(
        process, get_mpi_comm(cache.get_triangulation()))
        .run();

      if (true)
        {
          std::sort(all_send.begin(),
                    all_send.end(),
                    [&](const auto &a, const auto &b) {
                      if (std::get<0>(a) != std::get<0>(b))
                        return std::get<0>(a) < std::get<0>(b);

                      if (std::get<1>(a) != std::get<1>(b))
                        return std::get<1>(a) < std::get<1>(b);

                      return std::get<2>(a) < std::get<2>(b);
                    });
        }

      if (perform_handshake)
        {
          std::sort(all_recv.begin(),
                    all_recv.end(),
                    [&](const auto &a, const auto &b) {
                      if (std::get<0>(a) != std::get<0>(b))
                        return std::get<0>(a) < std::get<0>(b);

                      return std::get<1>(a) < std::get<1>(b);
                    });

          for (unsigned int i = 0; i < all_recv.size(); ++i)
            std::get<2>(all_recv[i]) = i;

          std::sort(all_recv.begin(),
                    all_recv.end(),
                    [&](const auto &a, const auto &b) {
                      if (std::get<1>(a) != std::get<1>(b))
                        return std::get<1>(a) < std::get<1>(b);

                      if (std::get<0>(a) != std::get<0>(b))
                        return std::get<0>(a) < std::get<0>(b);

                      return std::get<2>(a) < std::get<2>(b);
                    });
        }

      return all_send;
    }

    template <int dim, int spacedim>
    std::tuple<
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>,
      std::vector<std::vector<Point<dim>>>,
      std::vector<std::vector<unsigned int>>,
      std::vector<std::vector<Point<spacedim>>>,
      std::vector<std::vector<unsigned int>>>
    distributed_compute_point_locations_new(
      const GridTools::Cache<dim, spacedim> &                cache,
      const std::vector<Point<spacedim>> &                   points,
      const std::vector<std::vector<BoundingBox<spacedim>>> &global_bboxes,
      const double                                           tolerance = 1e-8)
    {
      const auto all = distributed_compute_point_locations_internal(
        cache, points, global_bboxes, tolerance, false);

      std::tuple<std::vector<
                   typename Triangulation<dim, spacedim>::active_cell_iterator>,
                 std::vector<std::vector<Point<dim>>>,
                 std::vector<std::vector<unsigned int>>,
                 std::vector<std::vector<Point<spacedim>>>,
                 std::vector<std::vector<unsigned int>>>
        result;

      std::pair<int, int> dummy{-1, -1};

      for (unsigned int i = 0; i < all.size(); ++i)
        {
          if (dummy != std::get<0>(all[i]))
            {
              std::get<0>(result).push_back(
                typename Triangulation<dim, spacedim>::active_cell_iterator{
                  &cache.get_triangulation(),
                  std::get<0>(all[i]).first,
                  std::get<0>(all[i]).second});

              const unsigned int new_size = std::get<0>(result).size();

              std::get<1>(result).resize(new_size);
              std::get<2>(result).resize(new_size);
              std::get<3>(result).resize(new_size);
              std::get<4>(result).resize(new_size);

              dummy = std::get<0>(all[i]);
            }

          std::get<1>(result).back().push_back(
            std::get<3>(all[i])); // reference point
          std::get<2>(result).back().push_back(std::get<2>(all[i])); // index
          std::get<3>(result).back().push_back(
            std::get<4>(all[i])); // real point
          std::get<4>(result).back().push_back(std::get<1>(all[i])); // rank
        }

      return result;
    }
  } // namespace GridTools
} // namespace dealii



template <int dim, int spacedim>
void
cmp(const std::vector<Point<spacedim>> &quadrature_points,
    const Triangulation<dim, spacedim> &tria,
    const Mapping<dim, spacedim> &      mapping)
{
  const MPI_Comm comm = MPI_COMM_WORLD;

  const auto temp_1 = [&]() {
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

    const auto &cells = std::get<0>(temp);

    std::tuple<
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>,
      std::vector<std::vector<Point<dim>>>,
      std::vector<std::vector<unsigned int>>,
      std::vector<std::vector<Point<spacedim>>>,
      std::vector<std::vector<unsigned int>>>
      result;

    std::vector<std::tuple<int, int, unsigned int>> cells_sorted;

    for (unsigned int i = 0; i < cells.size(); ++i)
      cells_sorted.emplace_back(cells[i]->level(), cells[i]->index(), i);

    std::sort(cells_sorted.begin(), cells_sorted.end());

    for (unsigned int i = 0; i < cells.size(); ++i)
      {
        const unsigned int index = std::get<2>(cells_sorted[i]);
        std::get<0>(result).push_back(std::get<0>(temp)[index]);
        std::get<1>(result).push_back(std::get<1>(temp)[index]);
        std::get<2>(result).push_back(std::get<2>(temp)[index]);
        std::get<3>(result).push_back(std::get<3>(temp)[index]);
        std::get<4>(result).push_back(std::get<4>(temp)[index]);
      }

    return result;
  }();

  const auto temp_2 = [&]() {
    const double tolerance = 1e-6;

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

    const GridTools::Cache<dim, spacedim> cache(tria, mapping);

    return dealii::GridTools::distributed_compute_point_locations_new<dim,
                                                                      spacedim>(
      cache, quadrature_points, global_bounding_boxes, tolerance);
  }();

  std::cout << std::endl << std::endl << std::endl;
  print(temp_1);
  std::cout << std::endl << std::endl << std::endl;
  print(temp_2);
}



template <int dim, int spacedim, typename VectorType>
void
compute_force_vector_sharp_interface(
  const Mapping<dim, spacedim> &   surface_mapping,
  const DoFHandler<dim, spacedim> &surface_dofhandler,
  const DoFHandler<dim, spacedim> &surface_dofhandler_dim,
  const Quadrature<dim> &          surface_quadrature,
  const Mapping<spacedim> &        mapping,
  const DoFHandler<spacedim> &     dof_handler,
  const double                     surface_tension,
  const VectorType &               normal_vector,
  const VectorType &               curvature_vector,
  VectorType &                     force_vector)
{
  using T = Tensor<1, spacedim, double>;

  const auto integration_points = [&]() {
    std::vector<Point<spacedim>> integration_points;

    FEValues<dim, spacedim> fe_eval(surface_mapping,
                                    surface_dofhandler.get_fe(),
                                    surface_quadrature,
                                    update_values | update_quadrature_points |
                                      update_JxW_values);

    for (const auto &cell :
         surface_dofhandler.get_triangulation().active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        fe_eval.reinit(cell);

        for (const auto q : fe_eval.quadrature_point_indices())
          integration_points.push_back(fe_eval.quadrature_point(q));
      }

    return integration_points;
  }();

  cmp(integration_points, dof_handler.get_triangulation(), mapping);

  Utilities::MPI::RemotePointEvaluation<spacedim, spacedim> eval;
  eval.reinit(integration_points, dof_handler.get_triangulation(), mapping);

  const auto integration_values = [&]() {
    std::vector<T> integration_values;

    FEValues<dim, spacedim> fe_eval(surface_mapping,
                                    surface_dofhandler.get_fe(),
                                    surface_quadrature,
                                    update_values | update_quadrature_points |
                                      update_JxW_values);
    FEValues<dim, spacedim> fe_eval_dim(surface_mapping,
                                        surface_dofhandler_dim.get_fe(),
                                        surface_quadrature,
                                        update_values);

    const auto &tria_surface = surface_dofhandler.get_triangulation();

    for (const auto &cell : tria_surface.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell(
          &tria_surface, cell->level(), cell->index(), &surface_dofhandler);
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
          &tria_surface, cell->level(), cell->index(), &surface_dofhandler_dim);

        fe_eval.reinit(dof_cell);
        fe_eval_dim.reinit(dof_cell_dim);

        std::vector<double>         curvature_values(fe_eval.dofs_per_cell);
        std::vector<Vector<double>> normal_values(fe_eval.dofs_per_cell,
                                                  Vector<double>(spacedim));

        fe_eval.get_function_values(curvature_vector, curvature_values);
        fe_eval_dim.get_function_values(normal_vector, normal_values);

        for (const auto q : fe_eval_dim.quadrature_point_indices())
          {
            T result;
            for (unsigned int i = 0; i < spacedim; ++i)
              result[i] = -curvature_values[q] * normal_values[q][i] *
                          fe_eval.JxW(q) * surface_tension;

            integration_values.push_back(result);
          }
      }

    return integration_values;
  }();

  const auto fu = [&](const auto &values, const auto &quadrature_points) {
    AffineConstraints<double> constraints; // TODO: use the right ones

    FEPointEvaluation<spacedim, spacedim> phi_force(mapping,
                                                    dof_handler.get_fe());

    std::vector<double>                  buffer;
    std::vector<types::global_dof_index> local_dof_indices;

    unsigned int i = 0;

    for (const auto &cells_and_n : std::get<0>(quadrature_points))
      {
        typename DoFHandler<spacedim>::active_cell_iterator cell = {
          &eval.get_triangulation(),
          cells_and_n.first.first,
          cells_and_n.first.second,
          &dof_handler};

        local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
        buffer.resize(cell->get_fe().n_dofs_per_cell());

        cell->get_dof_indices(local_dof_indices);

        AssertIndexRange(i + cells_and_n.second,
                         std::get<1>(quadrature_points).size() + 1);

        const ArrayView<const Point<spacedim>> unit_points(
          std::get<1>(quadrature_points).data() + i, cells_and_n.second);

        const ArrayView<const T> force_JxW(values.data() + i,
                                           cells_and_n.second);

        for (unsigned int q = 0; q < unit_points.size(); ++q)
          phi_force.submit_value(force_JxW[q], q);

        phi_force.integrate(cell, unit_points, buffer, EvaluationFlags::values);

        constraints.distribute_local_to_global(buffer,
                                               local_dof_indices,
                                               force_vector);

        i += unit_points.size();
      }
  };

  std::vector<T> buffer;

  eval.template process_and_evaluate<T>(integration_values, buffer, fu);
}



template <int dim>
void
test()
{
  const unsigned int spacedim = dim + 1;

  const unsigned int fe_degree      = 3;
  const unsigned int mapping_degree = fe_degree;
  const unsigned int n_refinements  = 5;

  parallel::shared::Triangulation<dim, spacedim> tria(
    MPI_COMM_WORLD, Triangulation<dim, spacedim>::none, true);
#if false
  GridGenerator::hyper_sphere(tria, Point<spacedim>(), 0.5);
#else
  GridGenerator::hyper_sphere(tria, Point<spacedim>(0.02, 0.03), 0.5);
#endif
  tria.refine_global(n_refinements);

  // quadrature rule and FE for curvature
  FE_Q<dim, spacedim>       fe(fe_degree);
  QGaussLobatto<dim>        quadrature(fe_degree + 1);
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // FE for normal
  FESystem<dim, spacedim>   fe_dim(fe, spacedim);
  DoFHandler<dim, spacedim> dof_handler_dim(tria);
  dof_handler_dim.distribute_dofs(fe_dim);

  // Set up MappingFEField
  Vector<double> euler_vector(dof_handler_dim.n_dofs());
  VectorTools::get_position_vector(dof_handler_dim,
                                   euler_vector,
                                   MappingQGeneric<dim, spacedim>(
                                     mapping_degree));
  MappingFEField<dim, spacedim> mapping(dof_handler_dim, euler_vector);


  // compute normal vector
  VectorType normal_vector(create_partitioner(dof_handler_dim));
  compute_normal(mapping, dof_handler_dim, normal_vector);

  // compute curvature
  VectorType curvature_vector(create_partitioner(dof_handler));
  compute_curvature(mapping,
                    dof_handler_dim,
                    dof_handler,
                    quadrature,
                    normal_vector,
                    curvature_vector);

#if false
  const unsigned int background_n_global_refinements = 6;
#else
  const unsigned int background_n_global_refinements =
    Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1 ? 80 : 80;
#endif
  const unsigned int background_fe_degree = 2;

  parallel::shared::Triangulation<spacedim> background_tria(
    MPI_COMM_WORLD, Triangulation<spacedim>::none, true);
#if false
  GridGenerator::hyper_cube(background_tria, -1.0, +1.0);
#else
  GridGenerator::subdivided_hyper_cube(background_tria,
                                       background_n_global_refinements,
                                       -2.5,
                                       2.5);
#endif
  if (background_n_global_refinements < 20)
    background_tria.refine_global(background_n_global_refinements);

  FESystem<spacedim>   background_fe(FE_Q<spacedim>{background_fe_degree},
                                   spacedim);
  DoFHandler<spacedim> background_dof_handler(background_tria);
  background_dof_handler.distribute_dofs(background_fe);

  MappingQ1<spacedim> background_mapping;

  VectorType force_vector_sharp_interface(
    create_partitioner(background_dof_handler));

  // write computed vectors to Paraview
  if (false)
    {
      GridOut().write_mesh_per_processor_as_vtu(tria, "grid_surface");
      GridOut().write_mesh_per_processor_as_vtu(background_tria,
                                                "grid_background");
    }

  compute_force_vector_sharp_interface(mapping,
                                       dof_handler,
                                       dof_handler_dim,
                                       QGauss<dim>(fe_degree + 1),
                                       background_mapping,
                                       background_dof_handler,
                                       1.0,
                                       normal_vector,
                                       curvature_vector,
                                       force_vector_sharp_interface);

  force_vector_sharp_interface.update_ghost_values();

  // write computed vectors to Paraview
  if (true)
    {
      DataOutBase::VtkFlags flags;
      // flags.write_higher_order_cells = true;

      DataOut<dim, DoFHandler<dim, spacedim>> data_out;
      data_out.set_flags(flags);
      data_out.add_data_vector(dof_handler, curvature_vector, "curvature");
      data_out.add_data_vector(dof_handler_dim, normal_vector, "normal");

      data_out.build_patches(mapping,
                             fe_degree + 1,
                             DataOut<dim, DoFHandler<dim, spacedim>>::
                               CurvedCellRegion::curved_inner_cells);
      data_out.write_vtu_with_pvtu_record("./",
                                          "data_surface",
                                          0,
                                          MPI_COMM_WORLD);
    }

  if (true)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<spacedim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(background_dof_handler);
      data_out.add_data_vector(background_dof_handler,
                               force_vector_sharp_interface,
                               "force");

      data_out.build_patches(background_mapping, background_fe_degree + 1);
      data_out.write_vtu_with_pvtu_record("./",
                                          "data_background",
                                          0,
                                          MPI_COMM_WORLD);
    }

  force_vector_sharp_interface.print(deallog.get_file_stream());
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MPILogInitAll                    all;

  test<1>();
}
