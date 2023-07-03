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
    return p[0];
  }
};

template <int dim>
void
test(const unsigned int degree)
{
  FE_SimplexP<dim> fe(degree);
  deallog << "FE = " << fe.get_name() << std::endl;
  QGaussSimplex<dim> quadrature(degree + 1);

  double previous_error = 1.0;

  for (unsigned int r = 0; r < 6; ++r)
    {
      Triangulation<dim> tria_hex, tria_flat, tria;
#if 1
      // having two cells is nice for debugging
      // GridGenerator::subdivided_hyper_cube_with_simplices(tria, 1);

      const unsigned int face_no     = 0;
      const unsigned int orientation = r;

      Triangulation<3> dummy;
      GridGenerator::reference_cell(dummy, ReferenceCells::Tetrahedron);

      auto vertices = dummy.get_vertices();

      std::vector<CellData<3>> cells;

      {
        CellData<3> cell;
        cell.vertices    = {0, 1, 2, 3};
        cell.material_id = 0;
        cells.push_back(cell);
      }

      {
        const auto &face = dummy.begin()->face(face_no);
        const auto  permuted =
          ReferenceCell(ReferenceCells::Triangle)
            .permute_according_orientation(
              std::array<unsigned int, 3>{{face->vertex_index(0),
                                           face->vertex_index(1),
                                           face->vertex_index(2)}},
              orientation);

        auto direction =
          cross_product_3d(vertices[permuted[1]] - vertices[permuted[0]],
                           vertices[permuted[2]] - vertices[permuted[0]]);
        direction = direction / direction.norm();

        vertices.emplace_back(0.0, 0.0, direction[2]);

        CellData<3> cell;
        cell.vertices = {permuted[0], permuted[1], permuted[2], 4u};

        cell.material_id = 1;
        cells.push_back(cell);
      }

      tria.create_triangulation(vertices, cells, {});

      for (const auto &cell : tria.active_cell_iterators())
        {
          for (const auto l : cell->line_indices())
            std::cout << cell->line_orientation(l) << " ";
          std::cout << std::endl;
        }
      std::cout << std::endl;

#else
      GridGenerator::hyper_cube(tria_hex);
      tria_hex.refine_global(r + 1);
      GridGenerator::flatten_triangulation(tria_hex, tria_flat);
      GridGenerator::convert_hypercube_to_simplex_mesh(tria_flat, tria);
#endif
      deallog << "Orientation " << r << std::endl;
      deallog << "Number of cells = " << tria.n_active_cells() << std::endl;

      ReferenceCell   reference_cell = tria.begin_active()->reference_cell();
      DoFHandler<dim> dof_handler(tria);
      dof_handler.distribute_dofs(fe);

      Vector<double>            cell_errors(tria.n_active_cells());
      Vector<double>            solution(dof_handler.n_dofs());
      LinearFunction<dim>       function;
      AffineConstraints<double> constraints;
      constraints.close();
      const auto &mapping =
        reference_cell.template get_default_linear_mapping<dim>();



      FEValues<dim> fe_values(mapping,
                              fe,
                              quadrature,
                              update_values | update_gradients |
                                update_JxW_values);

      const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

      FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
      Vector<double>     cell_rhs(dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

      for (const auto &cell : dof_handler.active_cell_iterators())
        {
          cell->get_dof_indices(local_dof_indices);

          for (const auto i : local_dof_indices)
            std::cout << i << " ";
          std::cout << std::endl;

          fe_values.reinit(cell);

          cell_matrix = 0;
          cell_rhs    = 0;

          for (const unsigned int q_index :
               fe_values.quadrature_point_indices())
            {
              for (const unsigned int i : fe_values.dof_indices())
                for (const unsigned int j : fe_values.dof_indices())
                  cell_matrix(i, j) +=
                    (fe_values.shape_value(i, q_index) * // grad phi_i(x_q)
                     fe_values.shape_value(j, q_index) * // grad phi_j(x_q)
                     fe_values.JxW(q_index));            // dx

              for (const unsigned int i : fe_values.dof_indices())
                cell_rhs(i) +=
                  (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                   1. *                                // f(x_q)
                   fe_values.JxW(q_index));            // dx
            }

          std::cout << cell_matrix.frobenius_norm() << std::endl;
          std::cout << cell_rhs.l2_norm() << std::endl;
        }


#if false
      VectorTools::project(
        mapping, dof_handler, constraints, quadrature, function, solution);
#else
      VectorTools::interpolate(mapping, dof_handler, function, solution);
#endif

      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        solution,
                                        function,
                                        cell_errors,
                                        quadrature,
                                        VectorTools::Linfty_norm);
      std::vector<Point<dim>> support_points(dof_handler.n_dofs());
      DoFTools::map_dofs_to_support_points(mapping,
                                           dof_handler,
                                           support_points);
      const double max_error =
        *std::max_element(cell_errors.begin(), cell_errors.end());
      deallog << "max error = " << max_error << std::endl;
      if (max_error != 0.0)
        deallog << "ratio = " << previous_error / max_error << std::endl;
      previous_error = max_error;

#if 1
      if (dim == 3)
        {
          DataOut<dim> data_out;
          data_out.attach_dof_handler(dof_handler);
          // solution    = 0.0;
          // solution[3] = 1.0;
          data_out.add_data_vector(solution, "u");
          data_out.build_patches(2);

          std::ofstream output("out-" + std::to_string(degree) + "-" +
                               std::to_string(r) + ".vtu");
          data_out.write_vtu(output);
        }
#endif
      deallog << std::endl;
    }
}

int
main()
{
  initlog();

  // test<2>(1);
  // test<2>(2);
  // test<2>(3);

  test<3>(1);
  test<3>(2);
  test<3>(3);
}