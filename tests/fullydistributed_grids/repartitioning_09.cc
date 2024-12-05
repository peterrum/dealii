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

#include <deal.II/multigrid/mg_transfer_global_coarsening.templates.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "../grid/tests.h"



template <int dim, int spacedim = dim>
class MyPolicy : public RepartitioningPolicyTools::Base<dim, spacedim>
{
public:
  MyPolicy(const Triangulation<dim, spacedim> &tria_background,
           const bool                          immersed_identification)
    : tria_background(&tria_background)
    , dof_handler_background(nullptr)
    , immersed_identification(immersed_identification)
  {}

  MyPolicy(const DoFHandler<dim, spacedim> &dof_handler_background,
           const bool                       immersed_identification)
    : tria_background(&dof_handler_background.get_triangulation())
    , dof_handler_background(&dof_handler_background)
    , immersed_identification(immersed_identification)
  {}

  virtual LinearAlgebra::distributed::Vector<double>
  partition(const Triangulation<dim, spacedim> &tria_immersed) const override
  {
    const unsigned int fe_degree = 2;

    std::vector<std::vector<unsigned int>> cell_ranks(
      tria_immersed.n_active_cells());

    if (immersed_identification)
      {
        std::vector<Point<spacedim>> points;

        Quadrature<dim> quadrature;

        if (fe_degree == 0)
          quadrature = QGauss<dim>(1);
        else
          quadrature = QGaussLobatto<dim>(fe_degree + 1);

        for (const auto &cell : tria_immersed.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              for (const auto &p : quadrature.get_points())
                {
                  points.push_back(
                    mapping.transform_unit_to_real_cell(cell, p));
                }
            }

        Utilities::MPI::RemotePointEvaluation<dim, spacedim> rpe;
        rpe.reinit(points, *tria_background, mapping);

        const auto evaluate_function = [&](const ArrayView<double> &values,
                                           const auto              &cell_data) {
          for (const auto cell : cell_data.cell_indices())
            {
              const auto unit_points = cell_data.get_unit_points(cell);
              const auto local_value = cell_data.get_data_view(cell, values);

              for (unsigned int q = 0; q < unit_points.size(); ++q)
                local_value[q] = Utilities::MPI::this_mpi_process(
                  tria_background->get_communicator());
            }
        };

        const std::vector<double> point_ranks =
          rpe.template evaluate_and_process<double>(evaluate_function);

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
      }
    else
      {
        std::vector<Point<spacedim>> points; // TODO: eliminate duplicate points

        if (dof_handler_background == nullptr)
          {
            Quadrature<dim> quadrature;

            if (fe_degree == 0)
              quadrature = QGauss<dim>(1);
            else
              quadrature = QGaussLobatto<dim>(fe_degree + 1);

            for (const auto &cell : tria_background->active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  for (const auto &p : quadrature.get_points())
                    {
                      points.push_back(
                        mapping.transform_unit_to_real_cell(cell, p));
                    }
                }
          }
        else
          {
            std::tie(points, std::ignore, std::ignore) =
              internal::collect_unconstrained_unique_support_points(
                *dof_handler_background, mapping, AffineConstraints<double>());
          }

        Utilities::MPI::RemotePointEvaluation<dim, spacedim> rpe;
        rpe.reinit(points, tria_immersed, mapping);

        std::vector<double> integration_values(
          points.size(),
          Utilities::MPI::this_mpi_process(
            tria_background->get_communicator()));

        const auto integration_function = [&](const auto &values,
                                              const auto &cell_data) {
          for (const auto cell : cell_data.cell_indices())
            {
              const auto unit_points = cell_data.get_unit_points(cell);
              const auto local_value = cell_data.get_data_view(cell, values);

              for (unsigned int q = 0; q < unit_points.size(); ++q)
                cell_ranks[cell_data.get_active_cell_iterator(cell)
                             ->active_cell_index()]
                  .push_back(local_value[q]);
            }
        };

        rpe.template process_and_evaluate<double>(integration_values,
                                                  integration_function);
      }

    const auto tria =
      dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
        &tria_immersed);

    Assert(tria, ExcNotImplemented());

    // 3) set partitioning
    LinearAlgebra::distributed::Vector<double> partition(
      tria->global_active_cell_index_partitioner().lock());


    const auto reduce = [](const auto &data) -> unsigned int {
      if (false /*smallest rank*/)
        {
          unsigned int rank = numbers::invalid_unsigned_int;

          for (const auto rank_i : data)
            rank = std::min<unsigned int>(rank, rank_i);

          return rank;
        }
      else if (true /*smallest rank with most hits*/)
        {
          std::map<unsigned int, unsigned int> rank_counter;

          for (const auto rank_i : data)
            rank_counter[rank_i] = 0;
          for (const auto rank_i : data)
            rank_counter[rank_i]++;

          const auto pr =
            std::max_element(rank_counter.begin(),
                             rank_counter.end(),
                             [](const auto &p1, const auto &p2) {
                               if (p1.second != p2.second)
                                 return p1.second < p2.second;

                               return p1.first > p2.first; // stable search
                             });

          return pr->first;
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());

          return 0; // TODO
        }
    };

    for (const auto &cell : tria_immersed.active_cell_iterators())
      if (cell->is_locally_owned())
        partition[cell->global_active_cell_index()] =
          reduce(cell_ranks[cell->active_cell_index()]);

    partition.update_ghost_values();

    return partition;
  }

private:
  const ObserverPointer<const Triangulation<dim, spacedim>> tria_background;
  const ObserverPointer<const DoFHandler<dim, spacedim>> dof_handler_background;
  const MappingQ1<dim, spacedim>                         mapping; // TODO
  const bool immersed_identification;
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

  DoFHandler<dim> dof_handler_background(tria_background);
  dof_handler_background.distribute_dofs(FE_Q<dim>(2));

  // create immersed mesh (default partitioning)
  parallel::distributed::Triangulation<dim> tria_immersed_old(comm);
  GridGenerator::hyper_ball(tria_immersed_old, Point<dim>(0.1, 0.2), 0.5);

  if (v == 0)
    tria_immersed_old.refine_global(5);
  else
    tria_immersed_old.refine_global(3);

  // create immersed mesh with partitioning as in the case of the
  // background mesh
  std::shared_ptr<MyPolicy<dim>> policy_0;

  if (v == 0 || v == 1)
    policy_0 = std::make_shared<MyPolicy<dim>>(tria_background, v == 0);
  else
    policy_0 = std::make_shared<MyPolicy<dim>>(dof_handler_background, v == 0);

  const auto partition_0 = policy_0->partition(tria_immersed_old);

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
  test<2>(2);
}
