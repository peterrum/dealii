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


// Test Simplex::PGauss: output its quadrature points and weights.


#include <deal.II/base/qprojector.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/mapping_fe.h>

#include <deal.II/hp/q_collection.h>

#include "../tests.h"

using namespace dealii;

template <int dim>
void
test()
{}

template <>
void
test<2>()
{
  const unsigned int dim = 2;

  const hp::QCollection<dim - 1> quad_ref(QGauss<dim - 1>(1),
                                          QGauss<dim - 1>(2),
                                          QGauss<dim - 1>(3),
                                          QGauss<dim - 1>(4));

  const auto quad =
    QProjector<dim>::project_to_all_faces(ReferenceCell::Type::Quad, quad_ref);

  const auto print = [&](const unsigned int face_no) {
    deallog << "face_no=" << face_no << ":" << std::endl;
    for (unsigned int
           q = 0,
           i = QProjector<dim>::DataSetDescriptor::face(
             ReferenceCell::Type::Quad, face_no, false, false, false, quad_ref);
         q < quad_ref[face_no].size();
         ++q, ++i)
      {
        deallog << quad.point(i) << " ";
        deallog << quad.weight(i) << " ";
        deallog << std::endl;
      }
    deallog << std::endl;
  };

  for (unsigned int i = 0; i < 4 /*TODO*/; ++i)
    print(i);
}

int
main()
{
  initlog();

  test<2>();
}
