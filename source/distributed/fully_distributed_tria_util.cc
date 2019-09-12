// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
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

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/fully_distributed_tria_util.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

DEAL_II_NAMESPACE_OPEN

namespace parallel
{
  namespace fullydistributed
  {
    namespace Utilities
    {
      namespace
      {
        /**
         * Set the user_flag of a cell and of all its parent cells.
         */
        template <int dim, int spacedim>
        void
        set_user_flag_reverse(TriaIterator<CellAccessor<dim, spacedim>> cell)
        {
          cell->set_user_flag();
          if (cell->level() != 0)
            set_user_flag_reverse(cell->parent());
        }


        /**
         * Convert the binary representation of a CellId to coarse-cell id as
         * if the finest level were the coarsest level ("level coarse-grid id").
         */
        template <int dim>
        types::coarse_cell_id
        convert_cell_id_binary_type_to_level_coarse_cell_id(
          const typename CellId::binary_type &binary_representation)
        {
          // exploiting the structure of CellId::binary_type
          // see also the documentation of CellId

          // actual coarse-grid id
          const unsigned int coarse_cell_id  = binary_representation[0];
          const unsigned int n_child_indices = binary_representation[1] >> 2;

          const unsigned int children_per_value =
            sizeof(CellId::binary_type::value_type) * 8 / dim;
          unsigned int child_level  = 0;
          unsigned int binary_entry = 2;

          // path to the get to the cell
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

          // compute new coarse-grid id: c_{i+1} = c_{i}*2^dim + q;
          types::coarse_cell_id level_coarse_cell_id = coarse_cell_id;
          for (auto i : cell_indices)
            level_coarse_cell_id =
              level_coarse_cell_id * GeometryInfo<dim>::max_children_per_cell +
              i;

          return level_coarse_cell_id;
        }
      } // namespace


      template <int dim, int spacedim>
      ConstructionData<dim, spacedim>
      create_construction_data_from_triangulation(
        const dealii::Triangulation<dim, spacedim> &tria,
        const Triangulation<dim, spacedim> &        tria_pft,
        const unsigned int                          my_rank_in)
      {
        const MPI_Comm comm = tria_pft.get_communicator();

        if (auto tria_pdt = dynamic_cast<
              const parallel::distributed::Triangulation<dim, spacedim> *>(
              &tria))
          AssertThrow(comm == tria_pdt->get_communicator(),
                      ExcMessage("MPI communicators do not match."));

        // First, figure out for what rank we are supposed to build the
        // ConstructionData object
        unsigned int my_rank = my_rank_in;
        AssertThrow(my_rank == numbers::invalid_unsigned_int ||
                      my_rank < dealii::Utilities::MPI::n_mpi_processes(comm),
                    ExcMessage(
                      "Rank has to be smaller than available processes."));

        if (auto tria_pdt = dynamic_cast<
              const parallel::distributed::Triangulation<dim, spacedim> *>(
              &tria))
          {
            AssertThrow(
              my_rank == numbers::invalid_unsigned_int ||
                my_rank == dealii::Utilities::MPI::this_mpi_process(comm),
              ExcMessage(
                "If parallel::distributed::Triangulation as source triangulation, my_rank has to equal global rank."));

            my_rank = dealii::Utilities::MPI::this_mpi_process(comm);
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

        ConstructionData<dim, spacedim> construction_data;

        auto &cells    = construction_data.coarse_cells;
        auto &vertices = construction_data.coarse_cell_vertices;
        auto &coarse_cell_index_to_coarse_cell_id =
          construction_data.coarse_cell_index_to_coarse_cell_id;
        auto &cell_infos = construction_data.cell_infos;

        // helper function, which collects all vertices belonging to a cell
        // (also taking into account periodicity)
        auto add_vertices_of_cell_to_vertices_owned_by_loclly_owned_cells =
          [](auto &cell, auto &vertices_owned_by_loclly_owned_cells) mutable {
            // add vertices belonging to a periodic neighbor
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

            // add local vertices
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 v++)
              vertices_owned_by_loclly_owned_cells.insert(
                cell->vertex_index(v));
          };

        // check if multilevel hierarchy should be constructed
        if (tria_pft.is_multilevel_hierarchy_constructed() == false)
          {
            AssertThrow(
              tria.has_hanging_nodes() == false,
              ExcMessage(
                "Hanging nodes are only supported if multilevel hierarchy is constructed!"));

            // 1) collect vertices of active locally owned cells
            std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
            for (auto cell : tria.cell_iterators())
              if (cell->active() && cell->subdomain_id() == my_rank)
                add_vertices_of_cell_to_vertices_owned_by_loclly_owned_cells(
                  cell, vertices_owned_by_loclly_owned_cells);

            // helper function to determine if cell is locally relevant
            // (i.e. a cell which is connected via a vertex via a locally owned
            // active cell)
            auto is_locally_relevant = [&](auto &cell) {
              for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                   v++)
                if (vertices_owned_by_loclly_owned_cells.find(
                      cell->vertex_index(v)) !=
                    vertices_owned_by_loclly_owned_cells.end())
                  return true;
              return false;
            };

            // 2) process all local and ghost cells: setup needed data
            // structures and collect all locally relevant vertices
            // for second sweep
            std::map<unsigned int, unsigned int> vertices_locally_relevant;
            cell_infos.push_back(std::vector<CellData<dim>>());
            auto &part = cell_infos[0];

            for (auto cell : tria.cell_iterators())
              if (cell->active() && is_locally_relevant(cell))
                {
                  // to be filled
                  CellData<dim> cell_info;

                  // a) extract cell definition (with old numbering of
                  // vertices)
                  dealii::CellData<dim> cell_data;
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

                  // d) compute a new coarse-cell id (by ignoring all parent
                  // level)
                  types::coarse_cell_id new_coarse_cell_id =
                    convert_cell_id_binary_type_to_level_coarse_cell_id<dim>(
                      cell->id().template to_binary<dim>());

                  //    store the coarse cell id
                  cell_info.id = CellId(new_coarse_cell_id, 0, nullptr)
                                   .template to_binary<dim>();

                  //    save coarse_cell_index -> coarse_cell_id mapping
                  coarse_cell_index_to_coarse_cell_id.push_back(
                    new_coarse_cell_id);

                  // e) store manifold id of cell
                  cell_info.manifold_id = cell->manifold_id();

                  // ... of lines
                  if (spacedim >= 2)
                    for (unsigned int line = 0;
                         line < GeometryInfo<spacedim>::lines_per_cell;
                         line++)
                      cell_info.manifold_line_ids[line] =
                        cell->line(line)->manifold_id();

                  // ... of quads
                  if (spacedim == 3)
                    for (unsigned int quad = 0;
                         quad < GeometryInfo<spacedim>::quads_per_cell;
                         quad++)
                      cell_info.manifold_quad_ids[quad] =
                        cell->quad(quad)->manifold_id();

                  // f) store subdomain_id
                  cell_info.subdomain_id = cell->subdomain_id();

                  // g) store invalid level_subdomain_id (since multilevel
                  //    hierarchy is not constructed)
                  cell_info.level_subdomain_id = numbers::invalid_subdomain_id;

                  part.push_back(cell_info);
                }

            // 3) enumerate locally relevant vertices
            unsigned int vertex_counter = 0;
            for (auto &vertex : vertices_locally_relevant)
              {
                vertices.push_back(tria.get_vertices()[vertex.first]);
                vertex.second = vertex_counter++;
              }

            // 4) correct vertices of cells (make them local)
            for (auto &cell : cells)
              for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                   v++)
                cell.vertices[v] = vertices_locally_relevant[cell.vertices[v]];
          }
        else
          {
            // 1) collect locally relevant cells (set user_flag)

            // 1a) clear user_flags
            for (auto &cell : tria)
              cell.recursively_clear_user_flag();

            // 1b) loop over levels (from fine to coarse) and mark on each level
            //     the locally relevant cells
            for (unsigned int level =
                   tria.get_triangulation().n_global_levels() - 1;
                 level != numbers::invalid_unsigned_int;
                 level--)
              {
                // collect vertices connected to a (on any level) locally owned
                // cell
                std::set<unsigned int> vertices_owned_by_loclly_owned_cells;
                for (auto cell : tria.cell_iterators_on_level(level))
                  if (cell->level_subdomain_id() == my_rank ||
                      (cell->active() && cell->subdomain_id() == my_rank))
                    add_vertices_of_cell_to_vertices_owned_by_loclly_owned_cells(
                      cell, vertices_owned_by_loclly_owned_cells);

                // helper function to determine if cell is locally relevant
                // (i.e. a cell which is connected via a vertex via a locally
                // owned cell)
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

                // mark all locally relevant cells
                for (auto cell : tria.cell_iterators_on_level(level))
                  if (is_locally_relevant(cell))
                    set_user_flag_reverse(cell);
              }

            // 2) setup coarse-grid triangulation
            {
              std::map<unsigned int, unsigned int> vertices_locally_relevant;

              // a) loop over all cells
              for (auto cell : tria.cell_iterators_on_level(0))
                {
                  if (!cell->user_flag_set())
                    continue;

                  // extract cell definition (with old numbering of vertices)
                  dealii::CellData<dim> cell_data;
                  cell_data.material_id = cell->material_id();
                  cell_data.manifold_id = cell->manifold_id();
                  for (unsigned int v = 0;
                       v < GeometryInfo<dim>::vertices_per_cell;
                       v++)
                    cell_data.vertices[v] = cell->vertex_index(v);
                  cells.push_back(cell_data);

                  // save indices of each vertex of this cell
                  for (unsigned int v = 0;
                       v < GeometryInfo<dim>::vertices_per_cell;
                       v++)
                    vertices_locally_relevant[cell->vertex_index(v)] =
                      numbers::invalid_unsigned_int;

                  // save translation for corase grid: lid -> gid
                  coarse_cell_index_to_coarse_cell_id.push_back(
                    cell->id().get_coarse_cell_id());
                }

              // b) enumerate locally relevant vertices
              unsigned int vertex_counter = 0;
              for (auto &vertex : vertices_locally_relevant)
                {
                  vertices.push_back(tria.get_vertices()[vertex.first]);
                  vertex.second = vertex_counter++;
                }

              // c) correct vertices of cells (make them local)
              for (auto &cell : cells)
                for (unsigned int v = 0;
                     v < GeometryInfo<dim>::vertices_per_cell;
                     v++)
                  cell.vertices[v] =
                    vertices_locally_relevant[cell.vertices[v]];
            }


            // 3) collect info of each cell
            cell_infos.resize(tria.get_triangulation().n_global_levels());

            for (unsigned int level = 0;
                 level < tria.get_triangulation().n_global_levels();
                 level++)
              {
                // collect local vertices on level
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

                auto &level_cell_infos = cell_infos[level];
                for (auto cell : tria.cell_iterators_on_level(level))
                  {
                    // check if cell is locally relevant
                    if (!(cell->user_flag_set()))
                      continue;

                    CellData<dim> cell_info;

                    // save coarse-cell id
                    cell_info.id = cell->id().template to_binary<dim>();

                    // save boundary_ids of each face of this cell
                    for (unsigned int f = 0;
                         f < GeometryInfo<dim>::faces_per_cell;
                         f++)
                      {
                        types::boundary_id boundary_ind =
                          cell->face(f)->boundary_id();
                        if (boundary_ind != numbers::internal_face_boundary_id)
                          cell_info.boundary_ids.emplace_back(f, boundary_ind);
                      }

                    // save manifold id
                    {
                      // ... of cell
                      cell_info.manifold_id = cell->manifold_id();

                      // ... of lines
                      if (spacedim >= 2)
                        for (unsigned int line = 0;
                             line < GeometryInfo<spacedim>::lines_per_cell;
                             line++)
                          cell_info.manifold_line_ids[line] =
                            cell->line(line)->manifold_id();

                      // ... of quads
                      if (spacedim == 3)
                        for (unsigned int quad = 0;
                             quad < GeometryInfo<spacedim>::quads_per_cell;
                             quad++)
                          cell_info.manifold_quad_ids[quad] =
                            cell->quad(quad)->manifold_id();
                    }

                    // subdomain and level subdomain id
                    cell_info.subdomain_id = numbers::artificial_subdomain_id;
                    cell_info.level_subdomain_id =
                      numbers::artificial_subdomain_id;

                    if (is_locally_relevant(cell))
                      {
                        cell_info.level_subdomain_id =
                          cell->level_subdomain_id();
                        if (cell->active())
                          cell_info.subdomain_id = cell->subdomain_id();
                      }

                    level_cell_infos.push_back(cell_info);
                  }
              }
          }

        return construction_data;
      }
    } // namespace Utilities
  }   // namespace fullydistributed
} // namespace parallel



/*-------------- Explicit Instantiations -------------------------------*/
#include "fully_distributed_tria_util.inst"


DEAL_II_NAMESPACE_CLOSE