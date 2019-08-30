// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2018 by the deal.II authors
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


// create a tria mesh and copy it

#include <deal.II/base/mpi.h>

#include <deal.II/distributed/full_distributed_tria.h>
#include <deal.II/distributed/full_distributed_tria_util.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"

using namespace dealii;

template <int dim>
void
test(int n_refinements, const int n_subdivisions, MPI_Comm comm)
{
  // create pdt
  Triangulation<dim> basetria(
    Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::subdivided_hyper_cube(basetria, n_subdivisions);
  basetria.refine_global(n_refinements);

  GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm),
                                     basetria,
                                     SparsityTools::Partitioner::metis);
  GridTools::partition_multigrid_levels(basetria);

  // create instance of pft
  parallel::fullydistributed::Triangulation<dim> tria_pft(
    comm,
    parallel::fullydistributed::Triangulation<
      dim>::construct_multigrid_hierarchy);

  // extract relevant information form serial triangulation
  auto construction_data =
    parallel::fullydistributed::Utilities::copy_from_triangulation(basetria,
                                                                   tria_pft);

  // actually create triangulation
  tria_pft.reinit(construction_data);

  // test triangulation
  FE_Q<dim>       fe(2);
  DoFHandler<dim> dof_handler(tria_pft);
  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const int      dim            = 2;
  const int      n_refinements  = 1;
  const int      n_subdivisions = 1;
  const MPI_Comm comm           = MPI_COMM_WORLD;

  if (dim == 1)
    test<1>(n_refinements, n_subdivisions, comm);
  else if (dim == 2)
    test<2>(n_refinements, n_subdivisions, comm);
  else if (dim == 3)
    test<3>(n_refinements, n_subdivisions, comm);
}
