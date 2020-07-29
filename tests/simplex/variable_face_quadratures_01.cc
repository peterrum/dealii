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

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>

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

  // test: QProjector::project_to_all_faces
  {
    const hp::QCollection<dim - 1> quad_ref(QGauss<dim - 1>(1),
                                            QGauss<dim - 1>(2),
                                            QGauss<dim - 1>(3),
                                            QGauss<dim - 1>(4));

    const auto quad =
      QProjector<dim>::project_to_all_faces(ReferenceCell::Type::Quad,
                                            quad_ref);

    const auto print = [&](const unsigned int face_no) {
      deallog << "face_no=" << face_no << ":" << std::endl;
      for (unsigned int q = 0,
                        i = QProjector<dim>::DataSetDescriptor::face(
                          ReferenceCell::Type::Quad,
                          face_no,
                          false,
                          false,
                          false,
                          quad_ref);
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

    deallog << std::endl;
  }

  // test: Mapping
  {
    const hp::QCollection<dim - 1> quad_ref(QGauss<dim - 1>(1),
                                            QGauss<dim - 1>(2),
                                            QGauss<dim - 1>(3),
                                            QGauss<dim - 1>(4));

    MappingFE<dim> mapping(FE_Q<dim>(1));
    FE_Q<dim> fe(3);

    const UpdateFlags flags = mapping.requires_update_flags(
      update_values | update_quadrature_points | update_JxW_values);

    auto data_ref = mapping.get_face_data(flags, quad_ref);

    internal::FEValuesImplementation::MappingRelatedData<dim> data;
    data.initialize(quad_ref.max_n_quadrature_points(), flags);

    Triangulation<dim> tria;
    GridGenerator::hyper_cube(tria);

    for (const auto &cell : tria.active_cell_iterators())
      for (const auto face_no : cell->face_indices())
        {
          mapping.fill_fe_face_values(cell, face_no, quad_ref, *data_ref, data);

          deallog << "face_no=" << face_no << ":" << std::endl;
          for (unsigned int q = 0; q < quad_ref[face_no].size(); ++q)
            {
              deallog << data.quadrature_points[q] << " ";
              deallog << data.JxW_values[q] << " ";
              deallog << std::endl;
            }
          deallog << std::endl;
        }
    
  internal::FEValuesImplementation::FiniteElementRelatedData<dim> data_fe;
  data_fe.initialize(quad_ref.max_n_quadrature_points(), fe, flags);

  auto data_fe_ref =fe.get_face_data(flags, mapping, quad_ref, data_fe);
   
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto face_no : cell->face_indices())
      {
        fe.fill_fe_face_values(cell, face_no, quad_ref, mapping, *data_ref, data, *data_fe_ref, data_fe);
      }
    
    
  }
}

int
main()
{
  initlog();

  test<2>();
}
