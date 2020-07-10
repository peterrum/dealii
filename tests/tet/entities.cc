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

#include <deal.II/grid/tria.h>

#include <deal.II/tet/grid_generator.h>

#include "../tests.h"

using namespace dealii;

void
test_3(const std::vector<unsigned int> indices)
{
  const int dim      = 3;
  const int spacedim = 3;

  std::vector<Point<spacedim>> vertices(indices.size());

  Triangulation<dim, spacedim> tria;

  tria.set_use_arbitray_mesh(true);

  std::vector<CellData<dim>> cells;
  CellData<dim>              cell;
  cell.vertices = indices;

  cells.push_back(cell);

  SubCellData subcelldata;
  tria.create_triangulation(vertices, cells, subcelldata);

  std::vector<unsigned int> indices_;

  {
    auto cell  = tria.begin();
    auto ecell = tria.end();

    for (; cell != ecell; ++cell)
      for (auto i : cell->vertex_indices())
        indices_.push_back(cell->vertex_index(i));
  }

  for (auto i : indices)
    deallog << i << " ";

  if (indices != indices_)
    {
      deallog << " -> ";

      for (auto i : indices_)
        deallog << i << " ";
    }
  deallog << std::endl;
}

int
main()
{
  initlog();

  // Tetrahedron
  test_3({0, 1, 2, 3});

  // Pyramid
  test_3({0, 1, 2, 3, 4});
  test_3({0, 1, 2, 4, 3});

  // Wedge
  test_3({0, 1, 2, 3, 4, 5});

  // Hexahedron
  test_3({0, 1, 2, 3, 4, 5, 6, 7});
}
