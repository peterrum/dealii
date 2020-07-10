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

  std::vector<Point<spacedim>> vertices_all;
  vertices_all.emplace_back(0, 0, 0);
  vertices_all.emplace_back(1, 0, 0);
  vertices_all.emplace_back(0, 1, 0);
  vertices_all.emplace_back(1, 1, 0);
  vertices_all.emplace_back(0, 0, 1);
  vertices_all.emplace_back(1, 0, 1);
  vertices_all.emplace_back(0, 1, 1);
  vertices_all.emplace_back(1, 1, 1);

  std::vector<unsigned int> indices_ = indices;
  std::sort(indices_.begin(), indices_.end());

  std::vector<unsigned int> indicies__;

  for (auto i : indices)
    indicies__.push_back(
      std::distance(indices_.begin(),
                    std::find(indices_.begin(), indices_.end(), i)));

  for (auto i : indicies__)
    deallog << i << " ";
  deallog << std::endl;

  Triangulation<dim, spacedim> tria;

  std::vector<Point<spacedim>> vertices;
  std::vector<CellData<dim>>   cells;
  SubCellData                  subcelldata;

  for (const auto i : indices_)
    vertices.push_back(vertices_all[i]);

  CellData<dim> cell;
  cell.vertices = indicies__;

  cells.push_back(cell);

  tria.create_triangulation(vertices_all, cells, subcelldata);

  {
    auto cell  = tria.begin();
    auto ecell = tria.end();

    for (; cell != ecell; ++cell)
      {
        for (auto i : cell->vertex_indices())
          deallog << cell->vertex_index(i) << " ";
        deallog << std::endl;
      }
  }
  deallog << std::endl;
}

int
main()
{
  initlog();

  {
    deallog.push("3D-1");
    test_3({0, 1, 2, 4});
    deallog.pop();
  }
  {
    deallog.push("3D-2");
    test_3({1, 3, 2, 7});
    deallog.pop();
  }
  {
    deallog.push("3D-3");
    test_3({1, 4, 5, 7});
    deallog.pop();
  }
  {
    deallog.push("3D-4");
    test_3({2, 4, 7, 6});
    deallog.pop();
  }
  {
    deallog.push("3D-5");
    test_3({1, 2, 4, 7});
    deallog.pop();
  }
  //
  //  {
  //    deallog.push("3D-1");
  //    test_3({0,1,3,5});
  //    deallog.pop();
  //  }
  //  {
  //    deallog.push("3D-2");
  //    test_3({0,3,2,6});
  //    deallog.pop();
  //  }
  //  {
  //    deallog.push("3D-3");
  //    test_3({1,4,5,6});
  //    deallog.pop();
  //  }
  //  {
  //    deallog.push("3D-4");
  //    test_3({3,5,7,6});
  //    deallog.pop();
  //  }
  //  {
  //    deallog.push("3D-5");
  //    test_3({0,3,6,5});
  //    deallog.pop();
  //  }
}