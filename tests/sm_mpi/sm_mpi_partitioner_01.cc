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

#include <deal.II/lac/sm_mpi_vector.h>

#include "../tests.h"

using namespace dealii;

void
test(const MPI_Comm comm)
{
  LinearAlgebra::SharedMPI::Partitioner partitioner;

  std::vector<types::global_dof_index> local_cells;
  std::vector<std::pair<types::global_dof_index, std::vector<unsigned int>>>
    local_ghost_faces;

  partitioner.reinit(local_cells, local_ghost_faces, comm);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const MPI_Comm comm = MPI_COMM_WORLD;

  test(comm);
}