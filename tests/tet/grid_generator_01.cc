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


// Test PolynomialsTet on quadrature points returned by QGaussTet.

#include <deal.II/grid/tria.h>

#include <deal.II/tet/grid_generator.h>

#include "../tests.h"

using namespace dealii;

template <int dim>
void
test()
{
  Triangulation<dim> tria;
  Tet::GridGenerator::subdivided_hyper_rectangle(tria,
                                                 {1, 1},
                                                 {0.0, 0.0},
                                                 {1.0, 1.0});

  deallog << tria.n_quads() << std::endl;
  deallog << tria.n_lines() << std::endl;
  deallog << tria.n_vertices() << std::endl;


  {
    deallog << std::endl << "CELLS" << std::endl;
    auto cell  = tria.begin();
    auto ecell = tria.end();

    for (; cell != ecell; ++cell)
      deallog << cell->index() << " " << cell->id() << std::endl;
  }

  {
    deallog << std::endl << "FACES" << std::endl;
    auto face  = tria.begin_face();
    auto eface = tria.end_face();

    for (; face != eface; ++face)
      deallog << face->index() << std::endl;
  }

  {
    deallog << std::endl << "CELLS -> FACES" << std::endl;
    auto cell  = tria.begin();
    auto ecell = tria.end();

    for (; cell != ecell; ++cell)
      {
        deallog << cell->index() << " " << cell->id() << std::endl;
        for (unsigned int i = 0; i < 3 /* TODO */; ++i)
          deallog << "  " << cell->line_index(i) << "  "
                  << cell->line(i)->index() << std::endl;
      }
  }

  {
    deallog << std::endl << "FACES -> VERTICES" << std::endl;
    auto face  = tria.begin_face();
    auto eface = tria.end_face();

    for (; face != eface; ++face)
      {
        deallog << face->index() << std::endl;
        for (unsigned int i = 0; i < 2 /* TODO */; ++i)
          deallog << "  " << face->vertex_index(i) << std::endl;
      }
  }


  {
    deallog << std::endl << "CELLS -> NEIGHBORS" << std::endl;
    auto cell  = tria.begin();
    auto ecell = tria.end();

    for (; cell != ecell; ++cell)
      {
        deallog << cell->index() << " " << cell->id() << std::endl;
        for (unsigned int i = 0; i < 3 /* TODO */; ++i)
          {
            deallog << "  " << cell->at_boundary(i) << " ";
            if (!cell->at_boundary(i))
              deallog << "  " << cell->neighbor_index(i) << "  "
                      << cell->neighbor(i)->index();
            deallog << std::endl;
          }
      }
  }
}

int
main()
{
  initlog();

  {
    deallog.push("2D");
    test<2>();
    deallog.pop();
  }
}