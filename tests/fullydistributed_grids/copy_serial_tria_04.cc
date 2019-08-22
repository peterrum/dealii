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

#include <deal.II/distributed/fullydistributed_tria.h>
#include <deal.II/distributed/fullydistributed_tria_util.h>
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
test(const int n_refinements, const int n_subdivisions, MPI_Comm comm)
{
  const double left  = 0;
  const double right = 1;

  auto add_periodicy = [&](dealii::Triangulation<dim> &tria,
                           const int                   offset = 0) {
    std::vector<
      GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
         periodic_faces;
    auto cell = tria.begin();
    auto endc = tria.end();
    for (; cell != endc; ++cell)
      for (unsigned int face_number = 0;
           face_number < GeometryInfo<dim>::faces_per_cell;
           ++face_number)
        if (std::fabs(cell->face(face_number)->center()(0) - left) < 1e-12)
          cell->face(face_number)->set_all_boundary_ids(0 + offset);
        else if (std::fabs(cell->face(face_number)->center()(0) - right) <
                 1e-12)
          cell->face(face_number)->set_all_boundary_ids(1 + offset);

    GridTools::collect_periodic_faces(
      tria, 0 + offset, 1 + offset, 0, periodic_faces);

    tria.add_periodicity(periodic_faces);
  };

  Triangulation<dim> basetria;
  GridGenerator::subdivided_hyper_cube(basetria, n_subdivisions);
  // new: add periodicy on serial mesh
  add_periodicy(basetria);
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

  // new: add periodicy on fullydistributed mesh (!!!)
  add_periodicy(tria_pft, 2);
}

int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  const int      dim            = 2;
  const int      n_refinements  = 4;
  const int      n_subdivisions = 8;
  const MPI_Comm comm           = MPI_COMM_WORLD;

  if (dim == 1)
    test<1>(n_refinements, n_subdivisions, comm);
  else if (dim == 2)
    test<2>(n_refinements, n_subdivisions, comm);
  else if (dim == 3)
    test<3>(n_refinements, n_subdivisions, comm);
}
