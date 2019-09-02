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

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/fully_distributed_tria_util.h>
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
test(int n_refinements, MPI_Comm comm)
{
  // create pdt
  Triangulation<dim> basetria;
  GridGenerator::hyper_L(basetria);
  basetria.refine_global(n_refinements);

  GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm),
                                     basetria,
                                     SparsityTools::Partitioner::metis);

  // create instance of pft
  parallel::fullydistributed::Triangulation<dim> tria_pft(comm);

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

  // output meshes as VTU
  // GridOut grid_out;
  // if(Utilities::MPI::this_mpi_process(comm) == 0)
  //  grid_out.write_mesh_per_processor_as_vtu(basetria, "tria", true, true);
  // grid_out.write_mesh_per_processor_as_vtu(tria_pft, "tria_pft", true, true);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const int      n_refinements = 3;
  const MPI_Comm comm          = MPI_COMM_WORLD;

  try
    {
      // deallog.push("1d");
      // test<1>(n_refinements, comm);
      // deallog.pop();
      deallog.push("2d");
      test<2>(n_refinements, comm);
      deallog.pop();
      deallog.push("3d");
      test<3>(n_refinements, comm);
      deallog.pop();
    }
  catch (...)
    {
      deallog << " failed...." << std::endl;
    }
}
