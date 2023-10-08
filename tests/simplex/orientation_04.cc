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

std::tuple<unsigned int, unsigned int>
create_triangulation(const std::vector<Point<3>>    &vertices_,
                     const std::vector<CellData<3>> &cell_data_,
                     const unsigned int              face_n,
                     const unsigned int              n_permuations,
                     Triangulation<3>               &tria)
{
  const ReferenceCell ref_cell  = ReferenceCells::Tetrahedron;
  auto                vertices  = vertices_;
  auto                cell_data = cell_data_;

  Point<3> extra_vertex;
  for (unsigned int i = 0; i < 3; ++i)
    extra_vertex += ref_cell.template vertex<3>(ref_cell.face_to_cell_vertices(
      face_n, i, ReferenceCell::default_combined_face_orientation()));

  extra_vertex /= 3.0;
  extra_vertex += ref_cell.template unit_normal_vectors<3>(face_n);

  vertices.push_back(extra_vertex);

  cell_data.emplace_back();
  cell_data.back().vertices.resize(0);
  for (unsigned int i = 0; i < 3; ++i)
    cell_data.back().vertices.push_back(ref_cell.face_to_cell_vertices(
      face_n, i, ref_cell.default_combined_face_orientation()));
  cell_data.back().vertices.push_back(ref_cell.n_vertices());
  std::sort(cell_data.back().vertices.begin(), cell_data.back().vertices.end());

  unsigned int permutation_n = 0;
  do
    {
      tria.clear();
      tria.create_triangulation(vertices, cell_data, SubCellData());
      ++permutation_n;
    }
  while ((permutation_n < n_permuations) &&
         std::next_permutation(cell_data.back().vertices.begin(),
                               cell_data.back().vertices.end()));

  const auto cell = tria.begin();

  const auto face = cell->face(face_n);

  auto ncell = tria.begin();
  ncell++;
  ncell->face(face_n);

  unsigned int nf = 0;
  for (; nf < ref_cell.n_faces(); ++nf)
    if (ncell->face(nf) == face)
      break;

  std::cout << ">>>>>>>>>>>>> " << n_permuations << " " << nf << " "
            << int(ncell->combined_face_orientation(nf)) << std::endl;

  return {nf, ncell->combined_face_orientation(nf)};
}

template <int dim>
void
test(const unsigned int degree)
{
  FE_SimplexP<dim> fe(degree);
  deallog << "FE = " << fe.get_name() << std::endl;
  QGaussSimplex<dim> quadrature(degree + 1);

  double previous_error = 1.0;

  for (unsigned int f = 0; f < 4; ++f)
    {
      for (unsigned int r = 0; r < 24; ++r)
        {
          unsigned int orientation = r;
          unsigned int face_no     = f;

          Triangulation<dim> tria;

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

          std::tie(face_no, orientation) =
            create_triangulation(vertices, cells, face_no, r, tria);

          deallog << "Orientation: "
                  << " (" << f << ", " << r << ") -> "
                  << " (" << face_no << ", " << orientation << ")" << std::endl
                  << " ";

          // for (unsigned int i0 = 0; i0 < 6; ++i0)
          //   {
          //     for (unsigned int i1 = 0; i1 < 6; ++i1)
          //       {
          //         for (const auto i2 : {false, true})
          //           {
          bool success = true;
          //
          //            const auto t = internal::bool_table[i0][i1];
          //
          //            internal::bool_table[i0][i1] = i2;

          auto cell = tria.begin();
          cell++;

          std::vector<unsigned int> verticess;

          for (const auto v : cell->vertex_indices())
            verticess.emplace_back(cell->vertex_index(v));

          std::cout << "-------------------------------------" << std::endl;

          std::cout << "X ";

          for (const auto i : verticess)
            std::cout << i << " ";
          std::cout << std::endl;

          for (unsigned int ll = 0; ll < 3; ++ll)
            {
              const unsigned int l =
                cell->reference_cell().face_to_cell_lines(face_no, ll, 1);

              const auto orientation_exp = cell->line_orientation(l);

              std::pair<unsigned int, unsigned int> p0;
              p0.first =
                verticess[cell->reference_cell().line_to_cell_vertices(l, 0)];
              p0.second =
                verticess[cell->reference_cell().line_to_cell_vertices(l, 1)];

              std::pair<unsigned int, unsigned int> p1;
              p1.first  = cell->line(l)->vertex_index(0);
              p1.second = cell->line(l)->vertex_index(1);

              if (orientation_exp == false)
                std::swap(p1.first, p1.second);

              success &= (p0 == p1);

              std::cout << "Y " << p0.first << " " << p0.second << " "
                        << p1.first << " " << p1.second << std::endl;
            }

          if (success)
            deallog << "x ";
          else
            deallog << "o ";

          //            internal::bool_table[i0][i1] = t;
          //          }
          //      }
          //  }
          deallog << std::endl;
        }
    }
}

int
main()
{
  initlog();

  // test<2>(1);
  // test<2>(2);
  // test<2>(3);

  // test<3>(1);
  test<3>(2);
  // test<3>(3);
}
