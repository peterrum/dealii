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



// Test refinement of 3D simplex mesh.

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"

template <int dim>
void
print(Triangulation<dim> &tria, const std::string &label)
{
  GridOut grid_out;

  unsigned int counter = 0;

  for (const auto &cell : tria.active_cell_iterators())
    cell->set_material_id(counter++);

  std::ofstream out(label);
#if false
  grid_out.write_vtk(tria, out);
#else
  (void)label;
  grid_out.write_vtk(tria, deallog.get_file_stream());
#endif
}

void
test()
{
  Triangulation<3> tria;
  ReferenceCell::make_triangulation(ReferenceCell::Type::Tet, tria);

  print(tria, "tria.0.vtk");
  tria.refine_global();
  print(tria, "tria.1.vtk");
}

int
main()
{
  initlog();
  test();
}
