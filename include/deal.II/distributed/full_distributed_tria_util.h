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

#ifndef dealii_full_distributed_tria_util.h
#define dealii_full_distributed_tria_util .h

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/full_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_tools.h>

#include <fstream>
#include <functional>

DEAL_II_NAMESPACE_OPEN

namespace parallel
{
  namespace fullydistributed
  {
    namespace Utilities
    {
      template <typename CELL>
      void
      set_flag_reverse(CELL cell)
      {
        cell->set_user_flag();
        if (cell->level() != 0)
          set_flag_reverse(cell->parent());
      }


      template <int dim>
      unsigned int
      convert_binary_to_gid(
        const std::array<unsigned int, 4> binary_representation)
      {
        const unsigned int coarse_cell_id = binary_representation[0];

        const unsigned int n_child_indices = binary_representation[1] >> 2;

        const unsigned int children_per_value =
          sizeof(CellId::binary_type::value_type) * 8 / dim;
        unsigned int child_level  = 0;
        unsigned int binary_entry = 2;

        std::vector<unsigned int> cell_indices;

        while (child_level < n_child_indices)
          {
            Assert(binary_entry < binary_representation.size(),
                   ExcInternalError());

            for (unsigned int j = 0; j < children_per_value; ++j)
              {
                unsigned int cell_index =
                  (((binary_representation[binary_entry] >> (j * dim))) &
                   (GeometryInfo<dim>::max_children_per_cell - 1));
                cell_indices.push_back(cell_index);
                ++child_level;
                if (child_level == n_child_indices)
                  break;
              }
            ++binary_entry;
          }

        unsigned int temp = coarse_cell_id;
        for (auto i : cell_indices)
          {
            temp = temp * GeometryInfo<dim>::max_children_per_cell + i;
          }

        return temp;
      }


      template <int dim, int spacedim = dim>
      ConstructionData<dim, spacedim>
      copy_from_triangulation(
        const dealii::Triangulation<dim, spacedim> &tria,
        const Triangulation<dim, spacedim> &        tria_pft,
        const unsigned int my_rank_in = numbers::invalid_unsigned_int)
      {
        const MPI_Comm comm = tria_pft.get_communicator();

        if (auto tria_pdt = dynamic_cast<
              const parallel::distributed::Triangulation<dim, spacedim> *>(
              &tria))
          AssertThrow(comm == tria_pdt->get_communicator(),
                      ExcMessage("MPI communicators do not match."));

        unsigned int my_rank = my_rank_in;
        AssertThrow(my_rank == numbers::invalid_unsigned_int ||
                      my_rank < dealii::Utilities::MPI::n_mpi_processes(comm),
                    ExcMessage(
                      "Rank has to be smaller than available processes."));

        if (auto tria_pdt = dynamic_cast<
              const parallel::distributed::Triangulation<dim, spacedim> *>(
              &tria))
          {
            if (my_rank == numbers::invalid_unsigned_int ||
                my_rank == dealii::Utilities::MPI::this_mpi_process(comm))
              my_rank = dealii::Utilities::MPI::this_mpi_process(comm);
            else
              AssertThrow(false,
                          ExcMessage("PDT: y_rank has to equal global rank."));
          }
        else if (auto tria_serial =
                   dynamic_cast<const dealii::Triangulation<dim, spacedim> *>(
                     &tria))
          {
            if (my_rank == numbers::invalid_unsigned_int)
              my_rank = dealii::Utilities::MPI::this_mpi_process(comm);
          }
        else
          {
            AssertThrow(false,
                        ExcMessage(
                          "This type of triangulation is not supported!"));
          }

        ConstructionData<dim, spacedim> cd;

        auto &cells    = cd.cells;
        auto &vertices = cd.vertices;
        auto &coarse_cell_index_to_coarse_cell_id =
          cd.coarse_cell_index_to_coarse_cell_id;
        auto &cell_infos = cd.cell_infos;

        auto add_vertices_of_cell_to_vertices_owned_by_loclly_owned_cells =
          [](auto &cell, auto &vertices_owned_by_loclly_owned_cells) mutable {
            for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; i++)
              if (cell->has_periodic_neighbor(i))
                {
                  auto face_t = cell->face(i);
                  auto face_n = cell->periodic_neighbor(i)->face(
                    cell->periodic_neighbor_face_no(i));
                  for (unsigned int j = 0;
                       j < GeometryInfo<dim>::vertices_per_face;
                       j++)
                    {
                      vertices_owned_by_loclly_owned_cells.insert(
                        face_t->vertex_index(j));
                      vertices_owned_by_loclly_owned_cells.insert(
                        face_n->vertex_index(j));
                    }
                }


            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 v++)
              vertices_owned_by_loclly_owned_cells.insert(
                cell->vertex_index(v));
          };

        if (!tria_pft.is_multilevel_hierarchy_constructed())
          {
            // 2) collect vertices of active locally owned cells
            std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
            for (auto cell : tria.cell_iterators())
              if (cell->active() && cell->subdomain_id() == my_rank)
                add_vertices_of_cell_to_vertices_owned_by_loclly_owned_cells(
                  cell, vertices_owned_by_loclly_owned_cells);

            // helper function to determine if cell is locally relevant
            auto is_locally_relevant = [&](auto &cell) {
              for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                   v++)
                if (vertices_owned_by_loclly_owned_cells.find(
                      cell->vertex_index(v)) !=
                    vertices_owned_by_loclly_owned_cells.end())
                  return true;
              return false;
            };

            // 3) process all local and ghost cells: setup needed data
            // structures and collect all locally relevant vertices
            // for second sweep
            std::map<unsigned int, unsigned int> vertices_locally_relevant;
            cell_infos.push_back(std::vector<CellInfo<dim>>());
            auto &part = cell_infos[0];

            unsigned int cell_counter = 0;
            for (auto cell : tria.cell_iterators())
              if (cell->active() && is_locally_relevant(cell))
                {
                  // a) extract cell definition (with old numbering of
                  // vertices)
                  CellData<dim> cell_data;
                  cell_data.material_id = cell->material_id();
                  cell_data.manifold_id = cell->manifold_id();
                  for (unsigned int v = 0;
                       v < GeometryInfo<dim>::vertices_per_cell;
                       v++)
                    cell_data.vertices[v] = cell->vertex_index(v);
                  cells.push_back(cell_data);

                  // b) save indices of each vertex of this cell
                  for (unsigned int v = 0;
                       v < GeometryInfo<dim>::vertices_per_cell;
                       v++)
                    vertices_locally_relevant[cell->vertex_index(v)] =
                      numbers::invalid_unsigned_int;

                  CellInfo<dim> cell_info;

                  // c) save boundary_ids of each face of this cell
                  for (unsigned int f = 0;
                       f < GeometryInfo<dim>::faces_per_cell;
                       f++)
                    {
                      types::boundary_id boundary_ind =
                        cell->face(f)->boundary_id();
                      if (boundary_ind != numbers::internal_face_boundary_id)
                        cell_info.boundary_ids.emplace_back(f, boundary_ind);
                    }

                  // e) save translation for corase grid: lid -> gid
                  coarse_cell_index_to_coarse_cell_id.push_back(
                    convert_binary_to_gid<dim>(
                      cell->id().template to_binary<dim>()));

                  cell_info.manifold_id = cell->manifold_id();

                  if (spacedim == 3)
                    {
                      for (unsigned int quad = 0;
                           quad < GeometryInfo<spacedim>::quads_per_cell;
                           quad++)
                        cell_info.manifold_quad_ids[quad] =
                          cell->quad(quad)->manifold_id();
                    }

                  if (spacedim >= 2)
                    {
                      for (unsigned int line = 0;
                           line < GeometryInfo<spacedim>::lines_per_cell;
                           line++)
                        cell_info.manifold_line_ids[line] =
                          cell->line(line)->manifold_id();
                    }

                  CellId::binary_type id;
                  id.fill(0);
                  id[0] = coarse_cell_index_to_coarse_cell_id.back();
                  id[1] = dim;
                  id[2] = 0;
                  id[3] = 0;

                  cell_info.id                 = id;
                  cell_info.subdomain_id       = cell->subdomain_id();
                  cell_info.level_subdomain_id = numbers::invalid_subdomain_id;

                  part.push_back(cell_info);

                  cell_counter++;
                }

            std::map<int, int> coarse_cell_id_to_coarse_cell_index;
            for (unsigned int i = 0;
                 i < coarse_cell_index_to_coarse_cell_id.size();
                 i++)
              coarse_cell_id_to_coarse_cell_index
                [coarse_cell_index_to_coarse_cell_id[i]] = i;

            std::sort(part.begin(), part.end(), [&](auto a, auto b) {
              auto a_index = a.id;
              a_index[0]   = coarse_cell_id_to_coarse_cell_index.at(a_index[0]);
              auto b_index = b.id;
              b_index[0]   = coarse_cell_id_to_coarse_cell_index.at(b_index[0]);

              return convert_binary_to_gid<dim>(a_index) <
                     convert_binary_to_gid<dim>(b_index);
            });

            // 4) enumerate locally relevant
            unsigned int vertex_counter = 0;
            for (auto &vertex : vertices_locally_relevant)
              {
                vertices.push_back(tria.get_vertices()[vertex.first]);
                vertex.second = vertex_counter++;
              }

            // 5) correct vertices of cells (make them local)
            for (auto &cell : cells)
              for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                   v++)
                cell.vertices[v] = vertices_locally_relevant[cell.vertices[v]];
          }
        else
          {
            for (auto cell : tria.cell_iterators_on_level(0))
              cell->recursively_clear_user_flag();

            for (unsigned int level =
                   tria.get_triangulation().n_global_levels() - 1;
                 level != numbers::invalid_unsigned_int;
                 level--)
              {
                std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
                for (auto cell : tria.cell_iterators_on_level(level))
                  if (cell->level_subdomain_id() == my_rank ||
                      (cell->active() && cell->subdomain_id() == my_rank))
                    add_vertices_of_cell_to_vertices_owned_by_loclly_owned_cells(
                      cell, vertices_owned_by_loclly_owned_cells);

                for (auto cell : tria.active_cell_iterators())
                  if (cell->subdomain_id() == my_rank)
                    add_vertices_of_cell_to_vertices_owned_by_loclly_owned_cells(
                      cell, vertices_owned_by_loclly_owned_cells);

                // helper function to determine if cell is locally relevant
                auto is_locally_relevant = [&](auto &cell) {
                  for (unsigned int v = 0;
                       v < GeometryInfo<dim>::vertices_per_cell;
                       v++)
                    if (vertices_owned_by_loclly_owned_cells.find(
                          cell->vertex_index(v)) !=
                        vertices_owned_by_loclly_owned_cells.end())
                      return true;
                  return false;
                };

                for (auto cell : tria.cell_iterators_on_level(level))
                  if (is_locally_relevant(cell))
                    set_flag_reverse(cell);
              }



            // 2) collect vertices of cells on level 0
            std::map<unsigned int, unsigned int> vertices_locally_relevant;

            unsigned int cell_counter = 0;
            for (auto cell : tria.cell_iterators_on_level(0))
              {
                if (!cell->user_flag_set())
                  continue;

                // b) extract cell definition (with old numbering of vertices)
                CellData<dim> cell_data;
                cell_data.material_id = cell->material_id();
                cell_data.manifold_id = cell->manifold_id();
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_cell;
                     v++)
                  cell_data.vertices[v] = cell->vertex_index(v);
                cells.push_back(cell_data);

                // c) save indices of each vertex of this cell
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_cell;
                     v++)
                  vertices_locally_relevant[cell->vertex_index(v)] =
                    numbers::invalid_unsigned_int;

                // e) save translation for corase grid: lid -> gid
                coarse_cell_index_to_coarse_cell_id.push_back(
                  convert_binary_to_gid<dim>(
                    cell->id().template to_binary<dim>()));

                cell_counter++;
              }

            // 4) enumerate locally relevant
            unsigned int vertex_counter = 0;
            for (auto &vertex : vertices_locally_relevant)
              {
                vertices.push_back(tria.get_vertices()[vertex.first]);
                vertex.second = vertex_counter++;
              }

            // 5) correct vertices of cells (make them local)
            for (auto &cell : cells)
              for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                   v++)
                cell.vertices[v] = vertices_locally_relevant[cell.vertices[v]];


            std::map<int, int> coarse_cell_id_to_coarse_cell_index;
            for (unsigned int i = 0;
                 i < coarse_cell_index_to_coarse_cell_id.size();
                 i++)
              coarse_cell_id_to_coarse_cell_index
                [coarse_cell_index_to_coarse_cell_id[i]] = i;

            for (unsigned int level = 0;
                 level < tria.get_triangulation().n_global_levels();
                 level++)
              {
                std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
                for (auto cell : tria.cell_iterators_on_level(level))
                  if (cell->level_subdomain_id() == my_rank ||
                      (cell->active() && cell->subdomain_id() == my_rank))
                    add_vertices_of_cell_to_vertices_owned_by_loclly_owned_cells(
                      cell, vertices_owned_by_loclly_owned_cells);

                // helper function to determine if cell is locally relevant
                auto is_locally_relevant = [&](auto &cell) {
                  for (unsigned int v = 0;
                       v < GeometryInfo<dim>::vertices_per_cell;
                       v++)
                    if (vertices_owned_by_loclly_owned_cells.find(
                          cell->vertex_index(v)) !=
                        vertices_owned_by_loclly_owned_cells.end())
                      return true;
                  return false;
                };


                std::set<unsigned int>
                  vertices_owned_by_loclly_owned_cells_strong;
                for (auto cell : tria.active_cell_iterators())
                  if (cell->subdomain_id() == my_rank)
                    add_vertices_of_cell_to_vertices_owned_by_loclly_owned_cells(
                      cell, vertices_owned_by_loclly_owned_cells_strong);

                // helper function to determine if cell is locally relevant
                auto is_locally_relevant_strong = [&](auto &cell) {
                  for (unsigned int v = 0;
                       v < GeometryInfo<dim>::vertices_per_cell;
                       v++)
                    if (vertices_owned_by_loclly_owned_cells_strong.find(
                          cell->vertex_index(v)) !=
                        vertices_owned_by_loclly_owned_cells_strong.end())
                      return true;
                  return false;
                };


                cell_infos.push_back(std::vector<CellInfo<dim>>());
                auto &part = cell_infos.back();
                for (auto cell : tria.cell_iterators_on_level(level))
                  {
                    if (!(cell->user_flag_set()))
                      continue;

                    auto id = cell->id().template to_binary<dim>();
                    // id[0]   = coarse_cell_id_to_coarse_cell_index[id[0]];

                    CellInfo<dim> cell_info;

                    // d) save boundary_ids of each face of this cell
                    for (unsigned int f = 0;
                         f < GeometryInfo<dim>::faces_per_cell;
                         f++)
                      {
                        types::boundary_id boundary_ind =
                          cell->face(f)->boundary_id();
                        if (boundary_ind != numbers::internal_face_boundary_id)
                          cell_info.boundary_ids.emplace_back(f, boundary_ind);
                      }

                    cell_info.manifold_id = cell->manifold_id();

                    if (spacedim == 3)
                      {
                        for (unsigned int quad = 0;
                             quad < GeometryInfo<spacedim>::quads_per_cell;
                             quad++)
                          cell_info.manifold_quad_ids[quad] =
                            cell->quad(quad)->manifold_id();
                      }

                    if (spacedim >= 2)
                      {
                        for (unsigned int line = 0;
                             line < GeometryInfo<spacedim>::lines_per_cell;
                             line++)
                          cell_info.manifold_line_ids[line] =
                            cell->line(line)->manifold_id();
                      }

                    if (cell->active() && is_locally_relevant_strong(cell))
                      {
                        cell_info.id           = id;
                        cell_info.subdomain_id = cell->subdomain_id(),
                        cell_info.level_subdomain_id =
                          cell->level_subdomain_id();
                      }
                    else if (is_locally_relevant(cell))
                      {
                        cell_info.id = id;
                        cell_info.subdomain_id =
                          numbers::artificial_subdomain_id;
                        cell_info.level_subdomain_id =
                          cell->level_subdomain_id();
                      }
                    else
                      {
                        cell_info.id = id;
                        cell_info.subdomain_id =
                          numbers::artificial_subdomain_id;
                        cell_info.subdomain_id =
                          numbers::artificial_subdomain_id;
                      }
                    part.push_back(cell_info);
                  }

                std::sort(part.begin(), part.end(), [&](auto a, auto b) {
                  auto a_index = a.id;
                  a_index[0] =
                    coarse_cell_id_to_coarse_cell_index.at(a_index[0]);
                  auto b_index = b.id;
                  b_index[0] =
                    coarse_cell_id_to_coarse_cell_index.at(b_index[0]);

                  return convert_binary_to_gid<dim>(a_index) <
                         convert_binary_to_gid<dim>(b_index);
                });
              }
          }

        return cd;
      }
    } // namespace Utilities
  }   // namespace fullydistributed
} // namespace parallel


DEAL_II_NAMESPACE_CLOSE

#endif
