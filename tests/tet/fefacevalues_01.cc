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

#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/quadrature_lib.h>

#include <vector>

#include "../tests.h"

using namespace dealii;

void
test_2(const std::vector<unsigned int> &vertices_1,
       const std::vector<unsigned int> &vertices_2,
       const unsigned int               degree)
{
  const int dim      = 2;
  const int spacedim = 2;

  std::vector<Point<spacedim>> vertices;

  vertices.emplace_back(0.0, 0.0);
  vertices.emplace_back(1.0, 0.0);
  vertices.emplace_back(0.0, 1.0);
  vertices.emplace_back(1.0, 1.0);

  CellData<dim> cell_1(3);
  cell_1.vertices = vertices_1;

  CellData<dim> cell_2(3);
  cell_2.vertices = vertices_2;

  Triangulation<dim, spacedim> tria;

  tria.create_triangulation(vertices, {cell_1, cell_2}, SubCellData());

  DoFHandler<dim, spacedim> dof_handler(tria);

  Tet::FE_Q<dim> fe(degree);
  dof_handler.distribute_dofs(fe);

  Tet::QGauss<dim> quad(dim == 2 ? (degree == 1 ? 3 : 7) :
                                   (degree == 1 ? 4 : 10));

  Tet::QGauss<dim - 1> face_quad(dim == 2 ? (degree == 1 ? 2 : 3) :
                                            (degree == 1 ? 3 : 7));

  const Tet::FE_Q<dim>            fe_mapping(1);
  const MappingIsoparametric<dim> mapping(fe_mapping);

  const UpdateFlags flags =
    update_JxW_values | update_normal_vectors | update_quadrature_points;

  FEFaceValues<dim, spacedim> fe_face_values(mapping, fe, face_quad, flags);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto face : cell->face_indices())
        {
          if (cell->at_boundary(face))
            continue;

          fe_face_values.reinit(cell, face);

          deallog << face << " : ";

          for (unsigned int q = 0; q < face_quad.size(); ++q)
            deallog << fe_face_values.JxW(q) << " ";
          deallog << std::endl;
        }
    }

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto face : cell->face_indices())
        {
          if (cell->at_boundary(face))
            continue;

          fe_face_values.reinit(cell, face);

          deallog << face << " : ";

          for (unsigned int q = 0; q < face_quad.size(); ++q)
            deallog << fe_face_values.normal_vector(q) << " ";
          deallog << std::endl;
        }
    }

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto face : cell->face_indices())
        {
          if (cell->at_boundary(face))
            continue;

          fe_face_values.reinit(cell, face);

          deallog << face << " : ";

          for (unsigned int q = 0; q < face_quad.size(); ++q)
            deallog << fe_face_values.quadrature_point(q) << " ";
          deallog << std::endl;
        }
    }

  deallog << std::endl;
}

std::vector<unsigned int>
rotate(const std::vector<unsigned int> &input, const unsigned n_rotations)
{
  std::vector<unsigned int> temp = input;

  for (unsigned int n = 0; n < n_rotations; ++n)
    {
      unsigned int end = temp.back();

      for (unsigned i = temp.size() - 1; i != 0; --i)
        temp[i] = temp[i - 1];

      temp[0] = end;
    }

  return temp;
}

void
test_2(const unsigned int degree)
{
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      {
        deallog << "rotate: " << i << "/" << j << ":" << std::endl;
        test_2(rotate({0, 1, 2}, i), rotate({3, 2, 1}, j), degree);
      }
}

int
main()
{
  initlog();

  test_2(1 /*degree*/);
  test_2(2 /*degree*/);
}
