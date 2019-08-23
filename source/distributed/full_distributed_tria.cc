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


#include <deal.II/base/logstream.h>
#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/full_distributed_tria.h>
#include <deal.II/distributed/p4est_wrappers.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>


DEAL_II_NAMESPACE_OPEN

namespace parallel
{
  namespace fullydistributed
  {
    namespace internal
    {
      template <int dim>
      bool
      parent(typename CellId::binary_type c, typename CellId::binary_type p)
      {
        if (c[0] != p[0])
          return false; // has same coarse cell
        if ((c[1] >> 2) != ((p[1] >> 2) + 1))
          return false; // level has one difference


        const unsigned int n_child_indices = p[1] >> 2;
        const unsigned int children_per_value =
          sizeof(CellId::binary_type::value_type) * 8 / dim;
        unsigned int child_level  = 0;
        unsigned int binary_entry = 2;

        // Loop until all child indices have been written
        while (child_level < n_child_indices)
          {
            Assert(binary_entry < c.size(), ExcInternalError());

            for (unsigned int j = 0; j < children_per_value; ++j)
              {
                if ((((c[binary_entry] >> (j * dim))) &
                     (GeometryInfo<dim>::max_children_per_cell - 1)) !=
                    (((p[binary_entry] >> (j * dim))) &
                     (GeometryInfo<dim>::max_children_per_cell - 1)))
                  return false;


                ++child_level;
                if (child_level == n_child_indices)
                  break;
              }

            ++binary_entry;
          }
        return true;
      }
    } // namespace internal

    template <int dim, int spacedim>
    void
    Triangulation<dim, spacedim>::reinit(
      ConstructionData<dim, spacedim> &construction_data)
    {
      // check if there are locally relevant coarse-grid cells
      if (construction_data.cells.size() == 0) // no
        {
          // 1) create a dummy hypercube
          Point<spacedim> p1, p2;
          for (unsigned int i = 0; i < dim; ++i)
            {
              p1(i) = 0.0;
              p2(i) = 1.0;
            }

          std::vector<Point<spacedim>> vertices(
            GeometryInfo<dim>::vertices_per_cell);
          switch (dim)
            {
              case 1:
                vertices[0] = p1;
                vertices[1] = p2;
                break;
              case 2:
                vertices[0] = vertices[1] = p1;
                vertices[2] = vertices[3] = p2;

                vertices[1](0) = p2(0);
                vertices[2](0) = p1(0);
                break;
              case 3:
                vertices[0] = vertices[1] = vertices[2] = vertices[3] = p1;
                vertices[4] = vertices[5] = vertices[6] = vertices[7] = p2;

                vertices[1](0) = p2(0);
                vertices[2](1) = p2(1);
                vertices[3](0) = p2(0);
                vertices[3](1) = p2(1);

                vertices[4](0) = p1(0);
                vertices[4](1) = p1(1);
                vertices[5](1) = p1(1);
                vertices[6](0) = p1(0);


                break;
              default:
                Assert(false, ExcNotImplemented());
            }

          std::vector<CellData<dim>> cells(1);
          for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell;
               ++i)
            cells[0].vertices[i] = i;
          cells[0].material_id = 0;
          SubCellData subcelldata;
          dealii::parallel::Triangulation<dim, spacedim>::create_triangulation(
            vertices, cells, subcelldata);

          // 2) mark cell as artificial
          for (auto cell = this->begin(); cell != this->end(); cell++)
            {
              cell->set_subdomain_id(dealii::numbers::artificial_subdomain_id);
              cell->set_level_subdomain_id(
                dealii::numbers::artificial_subdomain_id);
            }

          // 3) setup dummy mapping between locally relevant coarse-grid cells
          //    and global cells
          this->coarse_gid_to_lid.emplace_back(0, 0);
          this->coarse_lid_to_gid.emplace_back(0, 0);
        }
      else
        {
          auto &boundary_ids = construction_data.boundary_ids;
          auto &cells        = construction_data.cells;
          auto &vertices     = construction_data.vertices;
          auto &parts        = construction_data.parts;



          // create inverse map
          std::map<int, int> coarse_gid_to_lid;
          for (auto &i : coarse_lid_to_gid)
            coarse_gid_to_lid[i.second] = i.first;



          // convert map to vector and save data structures
          for (auto i : construction_data.coarse_lid_to_gid)
            this->coarse_lid_to_gid.emplace_back(i);

          for (auto i : coarse_gid_to_lid)
            this->coarse_gid_to_lid.emplace_back(i);



          // 1) create coarse grid
          SubCellData subcelldata;
          dealii::parallel::Triangulation<dim, spacedim>::create_triangulation(
            vertices, cells, subcelldata);

          for (auto c = this->begin(); c != this->end(); c++)
            c->set_all_manifold_ids(c->material_id());


          // 2) set boundary ids
          int c = 0;
          for (auto cell = this->begin_active(); cell != this->end(); ++cell)
            for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; i++)
              {
                unsigned int boundary_ind = boundary_ids[c++];
                if (boundary_ind != numbers::internal_face_boundary_id)
                  cell->face(i)->set_boundary_id(boundary_ind);
              }

          // 3) create all cell levels
          for (unsigned int ref_counter = 1; ref_counter < parts.size();
               ref_counter++)
            {
              auto coarse_cell    = this->begin(ref_counter - 1);
              auto fine_cell_info = parts[ref_counter].begin();

              for (; fine_cell_info != parts[ref_counter].end();
                   ++fine_cell_info)
                {
                  auto temp = fine_cell_info->index;
                  temp[0]   = coarse_cell_index_to_coarse_cell_id(temp[0]);
                  while (!internal::parent<dim>(
                    temp, coarse_cell->id().template to_binary<dim>()))
                    coarse_cell++;

                  coarse_cell->set_refine_flag();
                }

              dealii::Triangulation<dim, spacedim>::
                execute_coarsening_and_refinement();
            }



          // 4a) set all cells artificial
          for (auto cell = this->begin(); cell != this->end(); cell++)
            {
              if (cell->active())
                cell->set_subdomain_id(
                  dealii::numbers::artificial_subdomain_id);

              if (settings & construct_multigrid_hierarchy)
                cell->set_level_subdomain_id(
                  dealii::numbers::artificial_subdomain_id);
            }



          // 4b) set actual (level_)subdomain_ids
          for (unsigned int ref_counter = 0; ref_counter < parts.size();
               ref_counter++)
            {
              auto cell      = this->begin(ref_counter);
              auto cell_info = parts[ref_counter].begin();
              for (; cell_info != parts[ref_counter].end(); ++cell_info)
                {
                  auto temp = cell_info->index;
                  temp[0]   = coarse_cell_index_to_coarse_cell_id(temp[0]);
                  while (temp != cell->id().template to_binary<dim>())
                    cell++;

                  if (cell->active())
                    cell->set_subdomain_id(cell_info->subdomain_id);

                  if (settings & construct_multigrid_hierarchy)
                    cell->set_level_subdomain_id(cell_info->level_subdomain_id);
                }
            }
        }

      update_number_cache();
    }



    template <int dim, int spacedim>
    void
    Triangulation<dim, spacedim>::create_triangulation(
      const std::vector<Point<spacedim>> &vertices,
      const std::vector<CellData<dim>> &  cells,
      const SubCellData &                 subcelldata)
    {
      AssertThrow(false, ExcMessage("Use the reinit-method instead!"));

      dealii::parallel::Triangulation<dim, spacedim>::create_triangulation(
        vertices, cells, subcelldata);
    }



    template <int dim, int spacedim>
    Triangulation<dim, spacedim>::Triangulation(MPI_Comm mpi_communicaton,
                                                Settings settings)
      : Triangulation<dim, spacedim>(mpi_communicaton,
                                     mpi_communicaton,
                                     settings)
    {}



    template <int dim, int spacedim>
    Triangulation<dim, spacedim>::Triangulation(
      MPI_Comm mpi_communicator,
      MPI_Comm mpi_communicator_coarse,
      Settings settings)
      : parallel::DistributedTriangulationBase<dim, spacedim>(
          mpi_communicator,
          (settings & construct_multigrid_hierarchy) ?
            static_cast<
              typename dealii::Triangulation<dim, spacedim>::MeshSmoothing>(
              dealii::Triangulation<dim>::none |
              Triangulation<dim,
                            spacedim>::limit_level_difference_at_vertices) :
            static_cast<
              typename dealii::Triangulation<dim, spacedim>::MeshSmoothing>(
              dealii::Triangulation<dim>::none),
          false)
      , settings(settings)
      , mpi_communicator_coarse(mpi_communicator_coarse)
    {}



    template <int dim, int spacedim>
    void
    Triangulation<dim, spacedim>::update_number_cache()
    {
      parallel::Triangulation<dim, spacedim>::update_number_cache();

      if (settings & construct_multigrid_hierarchy)
        parallel::Triangulation<dim, spacedim>::fill_level_ghost_owners();
    }



    template <int dim, int spacedim>
    void
    Triangulation<dim, spacedim>::execute_coarsening_and_refinement()
    {
      Assert(false, ExcNotImplemented());
    }



    template <int dim, int spacedim>
    bool
    Triangulation<dim, spacedim>::prepare_coarsening_and_refinement()
    {
      return dealii::Triangulation<dim, spacedim>::
        prepare_coarsening_and_refinement();
    }



    template <int dim, int spacedim>
    bool
    Triangulation<dim, spacedim>::has_hanging_nodes() const
    {
      Assert(false, ExcNotImplemented());
      return false;
    }



    template <int dim, int spacedim>
    std::size_t
    Triangulation<dim, spacedim>::memory_consumption() const
    {
      std::size_t mem =
        this->dealii::parallel::TriangulationBase<dim, spacedim>::
          memory_consumption() +
        MemoryConsumption::memory_consumption(coarse_gid_to_lid) +
        MemoryConsumption::memory_consumption(coarse_lid_to_gid);
      return mem;
    }



    template <int dim, int spacedim>
    void
    Triangulation<dim, spacedim>::add_periodicity(
      const std::vector<GridTools::PeriodicFacePair<cell_iterator>>
        &periodicity_vector)
    {
      dealii::Triangulation<dim, spacedim>::add_periodicity(periodicity_vector);
    }



    template <int dim, int spacedim>
    std::map<unsigned int, std::set<dealii::types::subdomain_id>>
    Triangulation<dim, spacedim>::compute_vertices_with_ghost_neighbors() const
    {
      std::vector<bool> vertex_of_own_cell(this->n_vertices(), false);

      // collect nodes coinciding due to periodicity
      std::map<unsigned int, unsigned int> periodic_map;
      for (auto &cell : this->active_cell_iterators())
        if (cell->is_locally_owned() || cell->is_ghost())
          {
            for (unsigned int i = 0; i < GeometryInfo<dim>::faces_per_cell; i++)
              {
                if (cell->has_periodic_neighbor(i) &&
                    cell->periodic_neighbor(i)->active())
                  {
                    auto face_t = cell->face(i);
                    auto face_n = cell->periodic_neighbor(i)->face(
                      cell->periodic_neighbor_face_no(i));
                    for (unsigned int j = 0;
                         j < GeometryInfo<dim>::vertices_per_face;
                         j++)
                      {
                        auto         v_t  = face_t->vertex_index(j);
                        auto         v_n  = face_n->vertex_index(j);
                        unsigned int temp = std::min(v_t, v_n);
                        {
                          auto it = periodic_map.find(v_t);
                          if (it != periodic_map.end())
                            temp = std::min(temp, it->second);
                        }
                        {
                          auto it = periodic_map.find(v_n);
                          if (it != periodic_map.end())
                            temp = std::min(temp, it->second);
                        }
                        periodic_map[v_t] = temp;
                        periodic_map[v_n] = temp;
                      }
                  }
              }
          }

      // compress map
      for (auto &p : periodic_map)
        {
          if (p.first == p.second)
            continue;
          unsigned int temp = p.second;
          while (temp != periodic_map[temp])
            temp = periodic_map[temp];
          p.second = temp;
        }

      // check if map is actually compressed
      for (auto p : periodic_map)
        {
          if (p.first == p.second)
            continue;
          auto pp = periodic_map.find(p.second);
          if (pp->first == pp->second)
            continue;
          AssertThrow(false, ExcMessage("Map has to be compressed!"));
        }

      std::map<unsigned int, std::set<unsigned int>> sets;
      for (auto p : periodic_map)
        sets[p.second] = std::set<unsigned int>();

      for (auto p : periodic_map)
        sets[p.second].insert(p.first);

      std::map<unsigned int, std::set<unsigned int>> sets2;
      for (auto &s : sets)
        {
          for (auto &ss : s.second)
            sets2[ss] = s.second;
        }

      for (const auto &cell : this->active_cell_iterators())
        if (cell->is_locally_owned())
          {
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 ++v)
              vertex_of_own_cell[cell->vertex_index(v)] = true;
          }

      std::map<unsigned int, std::set<dealii::types::subdomain_id>> result;
      for (const auto &cell : this->active_cell_iterators())
        if (cell->is_ghost())
          {
            const types::subdomain_id owner = cell->subdomain_id();
            for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell;
                 ++v)
              {
                if (vertex_of_own_cell[cell->vertex_index(v)])
                  result[cell->vertex_index(v)].insert(owner);

                // mark also nodes coinciding due to periodicity
                auto t = sets2.find(cell->vertex_index(v));
                if (t != sets2.end())
                  for (auto i : t->second)
                    if (vertex_of_own_cell[i])
                      result[i].insert(owner);
              }
          }

      return result;
    }



    template <int dim, int spacedim>
    bool
    Triangulation<dim, spacedim>::is_multilevel_hierarchy_constructed() const
    {
      return (settings & construct_multigrid_hierarchy);
    }



    template <int dim, int spacedim>
    MPI_Comm
    Triangulation<dim, spacedim>::get_coarse_communicator() const
    {
      return mpi_communicator_coarse;
    }



    template <int dim, int spacedim>
    unsigned int
    Triangulation<dim, spacedim>::coarse_cell_id_to_coarse_cell_index(
      const types::coarse_cell_id coarse_cell_id) const
    {
      auto coarse_cell_index =
        std::lower_bound(coarse_gid_to_lid.begin(),
                         coarse_gid_to_lid.end(),
                         coarse_cell_id,
                         [](const auto &pair, const auto &val) {
                           return pair.first < val;
                         });
      Assert(coarse_cell_index != coarse_gid_to_lid.cend(),
             ExcMessage("Coarse cell index not found!"));
      return coarse_cell_index->second;
    }



    template <int dim, int spacedim>
    types::coarse_cell_id
    Triangulation<dim, spacedim>::coarse_cell_index_to_coarse_cell_id(
      const unsigned int coarse_cell_index) const
    {
      auto coarse_cell_id =
        std::lower_bound(coarse_lid_to_gid.begin(),
                         coarse_lid_to_gid.end(),
                         coarse_cell_index,
                         [](const auto &pair, const auto &val) {
                           return pair.first < val;
                         });
      Assert(coarse_cell_id != coarse_lid_to_gid.cend(),
             ExcMessage("Coarse cell id not found!"));
      return coarse_cell_id->second;
    }



  } // namespace fullydistributed
} // namespace parallel



/*-------------- Explicit Instantiations -------------------------------*/
#include "full_distributed_tria.inst"


DEAL_II_NAMESPACE_CLOSE
