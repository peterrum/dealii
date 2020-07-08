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
test_3(const std::vector<unsigned int> &vertices_1,
       const unsigned int               degree,
       const unsigned int               v)
{
  const int dim      = 3;
  const int spacedim = 3;

  std::vector<Point<spacedim>> vertices;

  if (v == 0)
    {
      vertices.emplace_back(0.0, 0.0, 0.0);
      vertices.emplace_back(1.0, 0.0, 0.0);
      vertices.emplace_back(0.0, 1.0, 0.0);
      vertices.emplace_back(0.0, 0.0, 1.0);
    }
  else if (v == 1)
    {
      vertices.emplace_back(0.0, 0.0, 0.0);
      vertices.emplace_back(1.0, 1.0, 0.0);
      vertices.emplace_back(1.0, 0.0, 1.0);
      vertices.emplace_back(0.0, 1.0, 1.0);
    }
  else
    {
      Assert(false, ExcNotImplemented());
    }


  CellData<dim> cell_1(4);
  cell_1.vertices = vertices_1;

  Triangulation<dim, spacedim> tria;

  tria.create_triangulation(vertices, {cell_1}, SubCellData());

  DoFHandler<dim, spacedim> dof_handler(tria);

  Tet::FE_Q<dim> fe(degree);
  dof_handler.distribute_dofs(fe);

  Tet::QGauss<dim - 1> face_quad(dim == 2 ? (degree == 1 ? 2 : 3) :
                                            (degree == 1 ? 3 : 7));

  const Tet::FE_Q<dim>            fe_mapping(1);
  const MappingIsoparametric<dim> mapping(fe_mapping);

  const UpdateFlags flags =
    update_JxW_values | update_normal_vectors | update_quadrature_points;

  FEFaceValues<dim, spacedim> fe_face_values_1(mapping, fe, face_quad, flags);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto face : cell->face_indices())
        {
          fe_face_values_1.reinit(cell, face);

          for (unsigned int q = 0; q < face_quad.size(); ++q)
            deallog << fe_face_values_1.JxW(q) << " ";

          double area = 0.0;
          for (unsigned int q = 0; q < face_quad.size(); ++q)
            area += fe_face_values_1.JxW(q);
          deallog << " -> " << area;

          deallog << std::endl;
        }
    }
}

std::vector<std::vector<unsigned int>>
all_possible_permutations(const std::vector<unsigned int> &ref)
{
  std::vector<std::vector<unsigned int>> result;

  // basic
  result.emplace_back(
    std::vector<unsigned int>{ref[0], ref[1], ref[2], ref[3]});

  // rotate around 0
  result.emplace_back(
    std::vector<unsigned int>{ref[0], ref[3], ref[1], ref[2]});
  result.emplace_back(
    std::vector<unsigned int>{ref[0], ref[2], ref[3], ref[1]});

  // rotate around 1
  result.emplace_back(
    std::vector<unsigned int>{ref[3], ref[1], ref[0], ref[2]});
  result.emplace_back(
    std::vector<unsigned int>{ref[2], ref[1], ref[3], ref[0]});

  // rotate around 2
  result.emplace_back(
    std::vector<unsigned int>{ref[3], ref[0], ref[2], ref[1]});
  result.emplace_back(
    std::vector<unsigned int>{ref[1], ref[3], ref[2], ref[0]});

  // rotate around 3
  result.emplace_back(
    std::vector<unsigned int>{ref[2], ref[0], ref[1], ref[3]});
  result.emplace_back(
    std::vector<unsigned int>{ref[1], ref[2], ref[0], ref[3]});

  return result;
}

void
test_3(const unsigned int degree, const unsigned int v)
{
  unsigned int counter = 0;

  for (const auto i : all_possible_permutations({0, 1, 2, 3}))
    {
      deallog << "v-" << counter++ << " : ";
      deallog << "(" << i[0] << " " << i[1] << " " << i[2] << " " << i[3]
              << ")";
      deallog << std::endl;

      test_3(i, degree, v);
      deallog << std::endl;
    }
}

int
main()
{
  initlog();

  test_3(1 /*degree*/, 0 /*version*/); // 0.500000, 0.500000, 0.500000, 0.866025
  test_3(2 /*degree*/, 0 /*version*/);
  test_3(1 /*degree*/, 1 /*version*/); // 0.866025, 0.866025, 0.866025, 0.866025
  test_3(2 /*degree*/, 1 /*version*/);
}
