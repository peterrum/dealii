// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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


// Create a serial triangulation and copy it.

#define MPI_SM_SIZE 2

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/sm_mpi_vector.h>

#include "../tests.h"

using namespace dealii;

template <int dim,
          int degree,
          typename Number,
          typename VectorizedArrayType = VectorizedArray<Number>>
void
test(const MPI_Comm comm)
{
  // 1) create triangulation
  parallel::distributed::Triangulation<dim> tria(comm);
  GridGenerator::subdivided_hyper_cube(tria, 2);

  tria.refine_global(1);

  // 2) create dof_handler so that cells are enumerated globally uniquelly
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_DGQ<dim>(0));

  // 3) setup data structures for partitioner
  std::vector<types::global_dof_index> local_cells;
  std::vector<std::pair<types::global_dof_index, std::vector<unsigned int>>>
    local_ghost_faces;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_artificial())
        continue;

      std::vector<types::global_dof_index> id(1);
      cell->get_dof_indices(id);
      if (cell->is_locally_owned()) // local cell
        local_cells.emplace_back(id.front());
      else // ghost cell
        {
          std::vector<unsigned int> faces;

          for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
               face++)
            {
              if (cell->at_boundary(face) ||
                  !cell->neighbor(face)->is_locally_owned())
                continue;

              faces.push_back(face);
            }

          if (!faces.empty())
            local_ghost_faces.emplace_back(id.front(), faces);
        }
    }

  // 4) setup partitioner
  LinearAlgebra::SharedMPI::
    Partitioner<dim, degree, Number, VectorizedArrayType>
      partitioner;
  partitioner.configure(false);
  partitioner.reinit(local_cells, local_ghost_faces, comm);

  deallog << partitioner.local_size() << " " //
          << partitioner.ghost_size() << std::endl;

  Number *              data_this;
  std::vector<double *> data_others;

  partitioner.initialize_dof_vector(data_this, data_others, true);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  test<2, 1, double>(comm);
}