/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2022 - 2022 by the deal.II authors
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
 */

// Verify convergence rates for various simplex elements

#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <deal.II/numerics/vector_tools_project.h>

#include "../tests.h"

template <int dim>
class LinearFunction : public Function<dim>
{
public:
  LinearFunction()
    : Function<dim>(1)
  {}

  virtual double
  value(const Point<dim> &p,
        const unsigned int /*component*/ = 0) const override
  {
    return p[1];
  }
};

template <int dim>
void
test(const unsigned int degree)
{
  FE_SimplexP<dim> fe(degree);
  deallog << "FE = " << fe.get_name() << std::endl;
  QWitherdenVincentSimplex<dim> quadrature(4);

  double previous_error = 1.0;

  for (unsigned int r = 0; r < 3; ++r)
    {
      Triangulation<dim> tria_hex, tria_flat, tria;
#if 1
#  if 0
      GridGenerator::subdivided_hyper_cube_with_simplices(tria, 1);
#  else
      std::vector<Point<dim>> points;
      points.emplace_back(0, 0);
      points.emplace_back(1, 0);
      points.emplace_back(0, 1);
      points.emplace_back(1, 1);

      std::vector<CellData<dim>> cells;
      cells.emplace_back();
      cells.back().vertices = {0, 1, 2};
      cells.emplace_back();
      cells.back().vertices = {2, 3, 1};
      tria.create_triangulation(points, cells, SubCellData());
#  endif
#else
      // cannot reproduce error on just one cell
      GridGenerator::hyper_cube(tria_hex);
      tria_hex.refine_global(r + 1);
      GridGenerator::flatten_triangulation(tria_hex, tria_flat);
      GridGenerator::convert_hypercube_to_simplex_mesh(tria_flat, tria);
#endif
      deallog << "Number of cells = " << tria.n_active_cells() << std::endl;

      ReferenceCell   reference_cell = tria.begin_active()->reference_cell();
      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      Vector<double>      cell_errors(tria.n_active_cells());
      Vector<double>      solution(dof_handler.n_dofs());
      LinearFunction<dim> function;
      // Functions::CosineFunction<dim> function;
      AffineConstraints<double> constraints;
      constraints.close();
      const auto &mapping =
        reference_cell.template get_default_linear_mapping<dim>();
#if 0
      {
#  if 0
        VectorTools::project(mapping,
                             dof_handler,
                             constraints,
                             quadrature,
                             function,
                             solution);
#  else
        DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
        SparsityPattern        sp(dof_handler.n_dofs(), dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(dof_handler, dsp);
        sp.copy_from(dsp);

        SparseMatrix<double> mass(sp);
        Vector<double>       rhs(dof_handler.n_dofs());

        FEValues<dim>      fe_values(mapping,
                                fe,
                                quadrature,
                                update_quadrature_points | update_JxW_values |
                                  update_values);
        FullMatrix<double> cell_mass(fe.dofs_per_cell, fe.dofs_per_cell);
        Vector<double>     cell_rhs(fe.dofs_per_cell);
        std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);

        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            cell_mass = 0.0;
            cell_rhs  = 0.0;
            cell->get_dof_indices(cell_dofs);
            fe_values.reinit(cell);

            for (unsigned int qp = 0; qp < quadrature.size(); ++qp)
              for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                {
                  cell_rhs[i] +=
                    function.value(fe_values.quadrature_point(qp)) *
                    fe_values.shape_value(i, qp) * fe_values.JxW(qp);

                  for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                    cell_mass(i, j) += fe_values.shape_value(i, qp) *
                                       fe_values.shape_value(j, qp) *
                                       fe_values.JxW(qp);
                }

            constraints.distribute_local_to_global(
              cell_mass, cell_rhs, cell_dofs, mass, rhs);
          }

        SolverControl        solver_control;
        SolverCG<>           solver(solver_control);
        PreconditionIdentity prec;
        solver.solve(mass, solution, rhs, prec);
#  endif
      }
#else
      VectorTools::interpolate(mapping, dof_handler, function, solution);
#endif

#if 1
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        function,
                                        cell_errors,
                                        quadrature,
                                        VectorTools::Linfty_norm);
#endif
      std::vector<Point<dim>> support_points(dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(mapping,
                                           dof_handler,
                                           support_points);
      if (r == 0)
        {
          for (types::global_dof_index i = 0; i < dof_handler.n_dofs(); ++i)
            {
              deallog << "dof " << i << std::endl
                      << "  point = " << support_points[i] << std::endl
                      << "  value = " << solution[i] << std::endl;
            }
        }
      {
        Quadrature<dim> nodal_points(fe.get_unit_support_points());
        if (r == 0)
          {
            deallog << "nodal quadrature =" << std::endl;
            for (unsigned int qp = 0; qp < nodal_points.size(); ++qp)
              deallog << "  " << nodal_points.point(qp) << std::endl;
          }
        FEValues<dim> fe_values(mapping,
                                fe,
                                nodal_points,
                                update_quadrature_points | update_values);
        for (const auto &cell : dof_handler.active_cell_iterators())
          {
            fe_values.reinit(cell);
            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                AssertThrow(std::abs(fe_values.shape_value(i, j) -
                                     double(i == j)) < 1e-12,
                            ExcInternalError());

            deallog << "vertices =" << std::endl
                    << "  " << cell->vertex_index(0) << ": " << cell->vertex(0)
                    << std::endl
                    << "  " << cell->vertex_index(1) << ": " << cell->vertex(1)
                    << std::endl
                    << "  " << cell->vertex_index(2) << ": " << cell->vertex(2)
                    << std::endl;

            std::vector<types::global_dof_index> cell_dofs(fe.dofs_per_cell);
            cell->get_dof_indices(cell_dofs);
            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
              deallog << "DoF " << cell_dofs[i] << std::endl
                      << "  support point " << fe_values.quadrature_point(i)
                      << std::endl;
          }
      }

      const double max_error =
        *std::max_element(cell_errors.begin(), cell_errors.end());
      deallog << "max error = " << max_error << std::endl;
      if (max_error != 0.0)
        deallog << "ratio = " << previous_error / max_error << std::endl;
      previous_error = max_error;

      if (dim == 2)
        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          solution    = 0.0;
          solution[3] = 1.0;
          data_out.add_data_vector(solution, "u");
          data_out.build_patches(2);

          std::ofstream output("out-" + std::to_string(degree) + "-" +
                               std::to_string(r) + ".vtu");
          data_out.write_vtu(output);
        }
    }
}

int
main()
{
  initlog();

#if 0
  test<2>(1);
  test<2>(2);
#endif
  test<2>(3);

#if 0
  test<3>(1);
  test<3>(2);
  test<3>(3);
#endif
}
