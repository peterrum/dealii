/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/grid_generator.h>
#include <deal.II/simplex/quadrature_lib.h>

#include <fstream>
#include <iostream>

#include "../tests.h" 

using namespace dealii;

template <int dim>
class Step3
{
public:
  Step3();
  void
  run_1();
  void
  run_2();

private:
  void
  make_grid_1();
  void
  make_grid_2();
  void
  setup_system();
  void
  assemble_system();
  void
  solve();
  void
  output_results() const;

  Triangulation<dim, dim> in_triangulation, triangulation;
  unsigned int            fe_degree;
  Simplex::FE_P<dim>      fe;
  Simplex::FE_P<dim>      fe_mapping;

  DoFHandler<dim> dof_handler;
  MappingFE<dim>  mapping;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};

template <int dim>
Step3<dim>::Step3()
  : fe_degree(1)
  , fe(fe_degree)
  , fe_mapping(1)
  , dof_handler(triangulation)
  , mapping(fe_mapping)
{}

template <int dim>
void
Step3<dim>::make_grid_1()
{
  unsigned int n_refinements = 3;
  GridGenerator::subdivided_hyper_cube(in_triangulation,
                                       2 * std::pow(2, n_refinements));
  GridGenerator::convert_hypercube_to_simplex_mesh<dim, dim>(in_triangulation,
                                                             triangulation);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}

template <int dim>
void
Step3<dim>::make_grid_2()
{
  unsigned int n_refinements = 3;
  GridGenerator::subdivided_hyper_cube(in_triangulation, 2);
  GridGenerator::convert_hypercube_to_simplex_mesh<dim, dim>(in_triangulation,
                                                             triangulation);
  triangulation.refine_global(n_refinements);

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}

template <int dim>
void
Step3<dim>::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim>
void
Step3<dim>::assemble_system()
{
  Simplex::QGauss<dim> quadrature_formula(dim == 2 ? (fe_degree == 1 ? 3 : 7) :
                                                     (fe_degree == 1 ? 4 : 10));

  FEValues<dim> fe_values(mapping,
                          fe,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // re-compute values &gradients of shape func and determinant at quad -
      // points
      fe_values.reinit(cell);

      // reset local cell matrix and rhs
      cell_matrix = 0;
      cell_rhs    = 0;

      // integrate by gauss quad (matrix)
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx

          // integrate by gauss quad (rhs)
          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            1. *                                // f(x_q)
                            fe_values.JxW(q_index));            // dx
        }
      // get global indices
      cell->get_dof_indices(local_dof_indices);

      // assemble global matrix and rhs
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }

  // get list of dof at boundary and compute boundary values
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), boundary_values);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 1, Functions::ZeroFunction<dim>(), boundary_values);
  // apply BC
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}

template <int dim>
void
Step3<dim>::solve()
{
  // define stop criterion for CG
  SolverControl solver_control(1000, 1e-12);

  SolverCG<Vector<double>> solver(solver_control);
  // solve: write solution in solution vector
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}

template <int dim>
void
Step3<dim>::output_results() const
{
  // Write output to a file
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  // transfer data from the front- to the backend, transform into intermediate
  // format
  data_out.build_patches(mapping);

  // open file and write data
  // std::ofstream output("solution.vtk");
  // data_out.write_vtk(output);
  data_out.write_vtk(deallog.get_file_stream());
}

template <int dim>
void
Step3<dim>::run_1()
{
  make_grid_1();
  setup_system();
  assemble_system();
  solve();
  output_results();
}

template <int dim>
void
Step3<dim>::run_2()
{
  make_grid_2();
  setup_system();
  assemble_system();
  solve();
  output_results();
}


int
main()
{
  initlog(); 
  
  deallog.depth_console(2);

  // refine mesh before conversion to simplex mesh
  // Step3<2> laplace_problem_1;
  // laplace_problem_1.run_1();


  // convert to simplex mesh and then refine
  Step3<2> laplace_problem_2;
  laplace_problem_2.run_2();

  return 0;
}
