// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


// Test
// TriangulationDescription::Utilities::create_description_from_triangulation()
// with repartitioning capabilities (partition immersed mesh as the background
// mesh).

#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/repartitioning_policy_tools.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "../grid/tests.h"



template <int dim, int spacedim = dim>
class MyPolicy : public RepartitioningPolicyTools::Base<dim, spacedim>
{
public:
  MyPolicy(const Triangulation<dim, spacedim> &tria_background)
    : tria_background(tria_background)
  {}

  virtual LinearAlgebra::distributed::Vector<double>
  partition(const Triangulation<dim, spacedim> &tria_immersed) const override
  {
    const unsigned int n_q_points = 2;

    // 1) collect centers of immeresed mesh
    std::vector<Point<spacedim>> points;

    Quadrature<dim> quadrature;

    if (n_q_points == 1)
      quadrature = QGauss<dim>(1);
    else
      quadrature = QGaussLobatto<dim>(2);

    for (const auto &cell : tria_immersed.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          for (const auto &p : quadrature.get_points())
            {
              points.push_back(mapping.transform_unit_to_real_cell(cell, p));
            }
        }

    // 2) determine owner on background mesh
    Utilities::MPI::RemotePointEvaluation<dim, spacedim> rpe;
    rpe.reinit(points, tria_background, mapping);

    const auto evaluate_function = [&](const ArrayView<double> &values,
                                       const auto              &cell_data) {
      for (const auto cell : cell_data.cell_indices())
        {
          const auto unit_points = cell_data.get_unit_points(cell);
          const auto local_value = cell_data.get_data_view(cell, values);

          for (unsigned int q = 0; q < unit_points.size(); ++q)
            local_value[q] = Utilities::MPI::this_mpi_process(
              tria_background.get_communicator());
        }
    };

    const std::vector<double> point_ranks =
      rpe.template evaluate_and_process<double>(evaluate_function);

    std::vector<std::vector<unsigned int>> cell_ranks(
      tria_immersed.n_active_cells());

    unsigned int counter = 0;
    for (const auto &cell : tria_immersed.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          unsigned int rank = numbers::invalid_unsigned_int;

          unsigned int start =
            rpe.get_point_ptrs()[counter * quadrature.size()];
          unsigned int end =
            rpe.get_point_ptrs()[(counter + 1) * quadrature.size()];

          for (unsigned int i = start; i < end; ++i)
            cell_ranks[cell->active_cell_index()].push_back(point_ranks[i]);

          counter++;
        }



    const auto tria =
      dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
        &tria_immersed);

    Assert(tria, ExcNotImplemented());

    // 3) set partitioning
    LinearAlgebra::distributed::Vector<double> partition(
      tria->global_active_cell_index_partitioner().lock());

    for (const auto &cell : tria_immersed.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          unsigned int rank = numbers::invalid_unsigned_int;

          for (const auto rank_i : cell_ranks[cell->active_cell_index()])
            rank = std::min<unsigned int>(rank, rank_i);

          partition[cell->global_active_cell_index()] = rank;

          counter++;
        }

    partition.update_ghost_values();

    return partition;
  }

private:
  const Triangulation<dim, spacedim> &tria_background;
  const MappingQ1<dim, spacedim>      mapping; // TODO
};


template <int dim>
void
output_mesh(const Triangulation<dim> &tria_background, const std::string label)
{
  DataOut<dim> data_out_background;
  data_out_background.attach_triangulation(tria_background);

  Vector<double> ranks(tria_background.n_active_cells());
  ranks = Utilities::MPI::this_mpi_process(tria_background.get_communicator());
  data_out_background.add_data_vector(ranks, "ranks");
  data_out_background.build_patches();
  data_out_background.write_vtu_in_parallel(label,
                                            tria_background.get_communicator());
}


template <int dim>
void
test(const unsigned int v)
{
  const MPI_Comm comm = MPI_COMM_WORLD;

  // create background mesh
  parallel::distributed::Triangulation<dim> tria_background(comm);
  GridGenerator::hyper_cube(tria_background, -1, +1);

  if (v == 0)
    tria_background.refine_global(5);
  else
    tria_background.refine_global(6);


  // create immersed mesh (default partitioning)
  parallel::distributed::Triangulation<dim> tria_immersed_old(comm);
  GridGenerator::hyper_ball(tria_immersed_old, Point<dim>(0.1, 0.2), 0.5);

  if (v == 0)
    tria_immersed_old.refine_global(5);
  else
    tria_immersed_old.refine_global(3);

  // create immersed mesh with partitioning as in the case of the
  // background mesh
  MyPolicy<dim> policy_0(tria_background);
  const auto    partition_0 = policy_0.partition(tria_immersed_old);

  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation(
      tria_immersed_old, partition_0);

  parallel::fullydistributed::Triangulation<dim> tria_immersed_new(comm);
  for (const auto i : tria_immersed_old.get_manifold_ids())
    if (i != numbers::flat_manifold_id)
      tria_immersed_new.set_manifold(i, tria_immersed_old.get_manifold(i));

  tria_immersed_new.create_triangulation(construction_data);

  // output meshes
  output_mesh(tria_background, "mesh_background.vtu");
  output_mesh(tria_immersed_old, "mesh_immersed_old.vtu");
  output_mesh(tria_immersed_new, "mesh_immersed_new.vtu");

  // print statistics
  print_statistics(tria_immersed_new);
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  MPILogInitAll                    all;

  test<2>(0);
  test<2>(1);
}
