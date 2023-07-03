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
      deallog << "Orientation " << r << std::endl;

      Triangulation<dim> tria_hex, tria_flat, tria;

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
              std::array<unsigned int, 3>{{0, 1, 2}}, orientation);

        for (const auto o : permuted)
          std::cout << o << " ";
        std::cout << std::endl;

        auto direction =
          cross_product_3d(vertices[permuted[1]] - vertices[permuted[0]],
                           vertices[permuted[2]] - vertices[permuted[0]]);
        direction = direction / direction.norm();

        std::cout << direction << std::endl;

        vertices.emplace_back(0.0, 0.0, direction[2]);

        CellData<3> cell;
        cell.vertices.resize(4);

        cell.vertices[permuted[0]] = face->vertex_index(0);
        cell.vertices[permuted[1]] = face->vertex_index(1);
        cell.vertices[permuted[2]] = face->vertex_index(2);
        cell.vertices[3]           = 4;

        cell.material_id = 1;
        cells.push_back(cell);
      }

      tria.create_triangulation(vertices, cells, {});

      for (const auto &cell : tria.active_cell_iterators())
        {
          for (const auto l : cell->face_indices())
            deallog << int(cell->combined_face_orientation(l)) << " ";
          deallog << std::endl;

          for (const auto l : cell->line_indices())
            deallog << cell->line_orientation(l) << " ";
          deallog << std::endl;
        }

      deallog << "Number of cells = " << tria.n_active_cells() << std::endl;
      deallog << std::endl;
    }
}

int
main()
{
  initlog();

  test<3>(1);
  test<3>(2);
  test<3>(3);
}