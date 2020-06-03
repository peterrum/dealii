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
test_2()
{
  const int dim = 2;

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

  // TODO: CELLS -> VERTICES

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

void
test_3()
{
  const int dim = 3;

  Triangulation<dim> tria;
  Tet::GridGenerator::subdivided_hyper_rectangle(tria,
                                                 {1, 1, 1},
                                                 {0.0, 0.0, 0.0},
                                                 {1.0, 1.0, 1.0});

  deallog << tria.n_cells() << std::endl;
  deallog << tria.n_hexs() << std::endl;
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

  // NOTE: loop over lines is not possible!

  {
    deallog << std::endl << "CELLS -> FACES" << std::endl;
    auto cell  = tria.begin();
    auto ecell = tria.end();

    for (; cell != ecell; ++cell)
      {
        deallog << cell->index() << " " << cell->id() << std::endl;
        for (unsigned int i = 0; i < 4 /* TODO */; ++i)
          deallog << "  " << cell->face_index(i) << "  "
                  << cell->face(i)->index() << std::endl;
      }
  }

  // TODO: CELLS -> LINES
  // TODO: CELLS -> VERTICES

  {
    deallog << std::endl << "FACES -> LINES" << std::endl;
    auto face  = tria.begin_face();
    auto eface = tria.end_face();

    for (; face != eface; ++face)
      {
        deallog << face->index() << std::endl;
        for (unsigned int i = 0; i < 3 /* TODO */; ++i)
          deallog << "  " << face->line_index(i) << "  "
                  << face->line(i)->index() << std::endl;
      }
  }

  // TODO: FACES -> VERTICES

  {
    deallog << std::endl << "CELLS -> NEIGHBORS" << std::endl;
    auto cell  = tria.begin();
    auto ecell = tria.end();

    for (; cell != ecell; ++cell)
      {
        deallog << cell->index() << " " << cell->id() << std::endl;
        for (unsigned int i = 0; i < 4 /* TODO */; ++i)
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
    test_2();
    deallog.pop();
  }

  {
    deallog.push("3D");
    test_3();
    deallog.pop();
  }
}