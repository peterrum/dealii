// // ---------------------------------------------------------------------
// //
// // Copyright (C) 2020 - 2022 by the deal.II authors
// //
// // This file is part of the deal.II library.
// //
// // The deal.II library is free software; you can use it, redistribute
// // it, and/or modify it under the terms of the GNU Lesser General
// // Public License as published by the Free Software Foundation; either
// // version 2.1 of the License, or (at your option) any later version.
// // The full text of the license can be found in the file LICENSE.md at
// // the top level directory of deal.II.
// //
// // ---------------------------------------------------------------------


/**
 * Test global-coarsening multigrid with non-nested levels for 3D deformed
 * geometries.
 */


#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <chrono>

#include "multigrid_util.h"

template <int dim = 2, typename Number = double>
void
test(const unsigned int n_refinements,
     const unsigned int fe_degree_fine,
     double             edge_size = 0.1)
{
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  const unsigned int min_level = 0;
  const unsigned int max_level = n_refinements;

  MGLevelObject<Triangulation<dim>>        triangulations(min_level, max_level);
  MGLevelObject<DoFHandler<dim>>           dof_handlers(min_level, max_level);
  MGLevelObject<AffineConstraints<Number>> constraints(min_level, max_level);
  MGLevelObject<MappingQ1<dim>>            mappings(min_level, max_level);
  MGLevelObject<std::shared_ptr<MGTwoLevelTransferNonNested<dim, VectorType>>>
                                       transfers(min_level, max_level);
  MGLevelObject<Operator<dim, Number>> operators(min_level, max_level);


  // set up levels
  double refine = 1.;
  for (auto l = min_level; l <= max_level; ++l)
    {
      auto &tria        = triangulations[l];
      auto &dof_handler = dof_handlers[l];
      auto &constraint  = constraints[l];
      auto &mapping     = mappings[l];
      auto &op          = operators[l];

      std::unique_ptr<FiniteElement<dim>> fe =
        std::make_unique<FE_Q<dim>>(fe_degree_fine);
      std::unique_ptr<Quadrature<dim>> quad =
        std::make_unique<QGauss<dim>>(fe_degree_fine + 1);

      // set up triangulation from gmsh files
      if (dim == 3)
        {
          GridGenerator::hyper_cube(tria);
          if (l == 0)
            tria.refine_global(l + 2);
          else
            tria.refine_global(l + 2);
          GridTools::distort_random(.25,
                                    tria,
                                    true,
                                    boost::random::mt19937::default_seed);
          deallog << "Number of cells = " << tria.n_active_cells() << std::endl;
          {
            GridOut       go;
            std::ofstream filename("level_" + std::to_string(l) + ".vtk");
            go.write_vtk(tria, filename);
          }
        }
      else
        {
          Assert(false, ExcImpossibleInDim(dim));
        }

      // set up dofhandler
      dof_handler.reinit(tria);
      dof_handler.distribute_dofs(*fe);
      deallog << "Number of DoFs = " << dof_handler.n_dofs() << std::endl;

      // set up constraints
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      constraint.reinit(locally_relevant_dofs);
      VectorTools::interpolate_boundary_values(
        mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraint);
      constraint.close();

      // set up operator
      op.reinit(mapping, dof_handler, *quad, constraint);
    }

  // set up transfer operator
  for (unsigned int l = min_level; l < max_level; ++l)
    {
      if constexpr(true)
      {
      transfers[l + 1] =
        std::make_shared<MGTwoLevelTransferNonNested<dim, VectorType>>();
      transfers[l + 1]->reinit(dof_handlers[l + 1],
                               dof_handlers[l],
                               mappings[l + 1],
                               mappings[l],
                               constraints[l + 1],
                               constraints[l]);
      }
      else if constexpr(false)
      {
      transfers[l + 1] =
        std::make_shared<MGTwoLevelTransfer<dim, VectorType>>();
      transfers[l + 1]->reinit(dof_handlers[l + 1],
                               dof_handlers[l],
                               constraints[l + 1],
                               constraints[l]);
      }
    }

  MGTransferGlobalCoarsening<dim, VectorType> transfer(
    transfers,
    [&](const auto l, auto &vec) { operators[l].initialize_dof_vector(vec); });


  GMGParameters mg_data; // TODO

  VectorType dst, src;
  operators[max_level].initialize_dof_vector(dst);
  operators[max_level].initialize_dof_vector(src);

  operators[max_level].rhs(src);

  ReductionControl solver_control(
    mg_data.maxiter, mg_data.abstol, mg_data.reltol, false, false);

  auto begin_mg = std::chrono::system_clock::now();
  mg_solve(solver_control,
           dst,
           src,
           mg_data,
           dof_handlers[max_level],
           operators[max_level],
           operators,
           transfer);
  auto end_mg = std::chrono::system_clock::now();
  deallog << "Elapsed time with Non-nested MG: "
          << (std::chrono::duration_cast<std::chrono::nanoseconds>(end_mg -
                                                                   begin_mg)
                .count()) /
               1e9
          << std::endl;
  deallog << dim << ' ' << fe_degree_fine << ' ' << n_refinements << ' '
          << "quad" << ' ' << solver_control.last_step() << std::endl;

  if(true)
  {
    
  auto                              begin = std::chrono::system_clock::now();
  TrilinosWrappers::PreconditionAMG preconditioner;
  TrilinosWrappers::PreconditionAMG::AdditionalData data;
  preconditioner.initialize(operators[n_refinements].get_system_matrix(), data);

  ReductionControl solver_control_amg(
    mg_data.maxiter, mg_data.abstol, mg_data.reltol, false, false);
  dealii::SolverCG<VectorType>(solver_control_amg)
    .solve(operators[n_refinements].get_system_matrix(),
           dst,
           src,
           preconditioner);
  auto end = std::chrono::system_clock::now();
  deallog << "Elapsed time with AMG: "
          << (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                .count()) /
               1e9
          << std::endl;
  deallog << dim << ' ' << fe_degree_fine << ' ' << n_refinements << ' '
          << "quad" << ' ' << solver_control_amg.last_step() << std::endl;
  }

  // if (n_refinements == 4)
  //   {
  //     DataOut<dim> data_out;

  //     data_out.attach_dof_handler(dof_handlers[n_refinements]);
  //     data_out.add_data_vector(dst, "soluton");

  //     data_out.build_patches();

  //     std::ofstream output("mg_non_nested_solution.vtk");
  //     data_out.write_vtk(output);
  //   }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  deallog.precision(8);

  constexpr unsigned int n_refinements = 3;
  // for (unsigned int degree = 1; degree <= 4; ++degree)
  // test<2>(n_refinements, 4);
  test<3>(n_refinements, 2);

  // {
  //   GridOut       go;
  //   std::ofstream filename("gmsh_test.vtk");
  //   go.write_vtk(tria, filename);
  // }
}
