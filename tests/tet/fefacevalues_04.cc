// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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


// Test different triangulation iterators.

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_isoparametric.h>

#include <deal.II/grid/tria.h>

#include <deal.II/tet/fe_dgq.h>
#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/grid_generator.h>
#include <deal.II/tet/quadrature_lib.h>

#include <vector>

#include "../tests.h"

using namespace dealii;

void
test_3(const unsigned int degree)
{
  const int dim      = 3;
  const int spacedim = 3;

  Triangulation<dim, spacedim> tria;
  Tet::GridGenerator::subdivided_hyper_cube(tria, 4);

  DoFHandler<dim, spacedim> dof_handler(tria);

  Tet::FE_DGQ<dim> fe(degree);
  dof_handler.distribute_dofs(fe);

  Tet::QGauss<dim - 1> face_quad(dim == 2 ? (degree == 1 ? 2 : 3) :
                                            (degree == 1 ? 3 : 7));

  const Tet::FE_Q<dim>            fe_mapping(1);
  const MappingIsoparametric<dim> mapping(fe_mapping);

  const UpdateFlags flags =
    update_JxW_values | update_normal_vectors | update_quadrature_points;

  FEFaceValues<dim, spacedim> fe_face_values_1(mapping, fe, face_quad, flags);
  FEFaceValues<dim, spacedim> fe_face_values_2(mapping, fe, face_quad, flags);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto face : cell->face_indices())
        {
          if (cell->at_boundary(face))
            continue;

          fe_face_values_1.reinit(cell, face);
          fe_face_values_2.reinit(cell->neighbor(face),
                                  cell->neighbor_face_no(face));

          unsigned int face_index_1 = 0;
          unsigned int face_index_2 = 0;

          face_index_1 = face;
          face_index_2 = cell->neighbor_face_no(face);


          const auto process = [&](const auto &fu, const auto &comp) {
#if false
    deallog << face_index_1 << " : ";

    for (unsigned int q = 0; q < face_quad.size(); ++q)
      deallog << fu(fe_face_values_1, q) << " ";
    deallog << std::endl;

    deallog << face_index_2 << " : ";

    for (unsigned int q = 0; q < face_quad.size(); ++q)
      deallog << fu(fe_face_values_2, q) << " ";
    deallog << std::endl;
#endif
            for (unsigned int q = 0; q < face_quad.size(); ++q)
              {
                Assert(comp(fu(fe_face_values_1, q), fu(fe_face_values_2, q)),
                       ExcNotImplemented());
              }
          };

          process([](const auto &eval, const auto q) { return eval.JxW(q); },
                  [](const auto &a, const auto &b) {
                    return std::abs(a - b) < 10e-8;
                  });
          process([](const auto &eval,
                     const auto  q) { return eval.normal_vector(q); },
                  [](const auto &a, const auto &b) {
                    return (a + b).norm() < 10e-8;
                  });
          process([](const auto &eval,
                     const auto  q) { return eval.quadrature_point(q); },
                  [](const auto &a, const auto &b) {
                    return (a - b).norm() < 10e-8;
                  });
        }
    }
}

int
main()
{
  initlog();

  test_3(1 /*degree*/);
  test_3(2 /*degree*/);
}
