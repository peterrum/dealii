// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2020 by the deal.II authors
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


#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>

#include <iostream>
#include <utility>

#include "../tests.h"



/* The 2D case */
void generate_grid(Triangulation<2> &triangulation, int orientation)
{
  Point<2> vertices_1[] = {
    Point<2>(-1., -3.),
    Point<2>(+1., -3.),
    Point<2>(-1., -1.),
    Point<2>(+1., -1.),
    Point<2>(-1., +1.),
    Point<2>(+1., +1.),
    Point<2>(-1., +3.),
    Point<2>(+1., +3.),
  };
  std::vector<Point<2>> vertices(&vertices_1[0], &vertices_1[8]);

  std::vector<CellData<2>> cells(2, CellData<2>());

  /* cell 0 */
  int cell_vertices_0[GeometryInfo<2>::vertices_per_cell] = {0, 1, 2, 3};

  /* cell 1 */
  int cell_vertices_1[2][GeometryInfo<2>::vertices_per_cell] = {
    {4, 5, 6, 7},
    {7, 6, 5, 4},
  };

  for (const unsigned int j : GeometryInfo<2>::vertex_indices())
    {
      cells[0].vertices[j] = cell_vertices_0[j];
      cells[1].vertices[j] = cell_vertices_1[orientation][j];
    }

  triangulation.create_triangulation(vertices, cells, SubCellData());

  auto cell_input = cells.begin();

  for (const auto &cell : triangulation.active_cell_iterators())
    {
      for (unsigned int v = 0; v < cell_input->vertices.size(); ++v)
        deallog << cell->vertex_index(v) << " ";
      deallog << std::endl;

      for (const auto &vertex : cell_input->vertices)
        deallog << vertex << " ";
      deallog << std::endl;

      cell_input++;
    }
  deallog << std::endl;
}


/* The 3D case */
void generate_grid(Triangulation<3> &triangulation, int orientation)
{
  Point<3>              vertices_1[] = {Point<3>(-1., -1., -3.),
                           Point<3>(+1., -1., -3.),
                           Point<3>(-1., +1., -3.),
                           Point<3>(+1., +1., -3.),
                           Point<3>(-1., -1., -1.),
                           Point<3>(+1., -1., -1.),
                           Point<3>(-1., +1., -1.),
                           Point<3>(+1., +1., -1.),
                           Point<3>(-1., -1., +1.),
                           Point<3>(+1., -1., +1.),
                           Point<3>(-1., +1., +1.),
                           Point<3>(+1., +1., +1.),
                           Point<3>(-1., -1., +3.),
                           Point<3>(+1., -1., +3.),
                           Point<3>(-1., +1., +3.),
                           Point<3>(+1., +1., +3.)};
  std::vector<Point<3>> vertices(&vertices_1[0], &vertices_1[16]);

  std::vector<CellData<3>> cells(2, CellData<3>());

  /* cell 0 */
  int cell_vertices_0[GeometryInfo<3>::vertices_per_cell] = {
    0, 1, 2, 3, 4, 5, 6, 7};

  /* cell 1 */
  int cell_vertices_1[8][GeometryInfo<3>::vertices_per_cell] = {
    {8, 9, 10, 11, 12, 13, 14, 15},
    {9, 11, 8, 10, 13, 15, 12, 14},
    {11, 10, 9, 8, 15, 14, 13, 12},
    {10, 8, 11, 9, 14, 12, 15, 13},
    {13, 12, 15, 14, 9, 8, 11, 10},
    {12, 14, 13, 15, 8, 10, 9, 11},
    {14, 15, 12, 13, 10, 11, 8, 9},
    {15, 13, 14, 12, 11, 9, 10, 8},
  };

  for (const unsigned int j : GeometryInfo<3>::vertex_indices())
    {
      cells[0].vertices[j] = cell_vertices_0[j];
      cells[1].vertices[j] = cell_vertices_1[orientation][j];
    }


  triangulation.create_triangulation(vertices, cells, SubCellData());

  auto cell_input = cells.begin();

  for (const auto &cell : triangulation.active_cell_iterators())
    {
      for (unsigned int v = 0; v < cell_input->vertices.size(); ++v)
        deallog << cell->vertex_index(v) << " ";
      deallog << std::endl;

      for (const auto &vertex : cell_input->vertices)
        deallog << vertex << " ";
      deallog << std::endl;

      cell_input++;
    }
  deallog << std::endl;
}


int
main()
{
  initlog();

  deallog << "Test for 2D:" << std::endl << std::endl;

  for (int i = 0; i < 2; ++i)
    {
      // Generate a triangulation and match:
      Triangulation<2> triangulation;
      triangulation.set_use_arbitray_mesh(true);

      deallog << "Triangulation:" << i << std::endl;

      generate_grid(triangulation, i);
    }

  deallog << "Test for 3D:" << std::endl << std::endl;

  for (int i = 0; i < 8; ++i)
    {
      Triangulation<3> triangulation;
      triangulation.set_use_arbitray_mesh(true);

      deallog << "Triangulation:" << i << std::endl;

      generate_grid(triangulation, i);
    }

  return 0;
}
