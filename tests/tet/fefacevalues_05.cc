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

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/tet/fe_dgq.h>
#include <deal.II/tet/fe_q.h>
#include <deal.II/tet/grid_generator.h>
#include <deal.II/tet/quadrature_lib.h>

#include <vector>

#include "../tests.h"

using namespace dealii;

template <int dim>
class Fu : public Function<dim>
{
public:
  Fu(const unsigned int c)
    : c(c)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override
  {
    (void)component;
    return p[c];
  }

private:
  const unsigned int c;
};

void
test_3(const unsigned int degree)
{
  const int dim      = 3;
  const int spacedim = 3;

  Triangulation<dim, spacedim> tria;
  Tet::GridGenerator::subdivided_hyper_cube(tria, 2);

  DoFHandler<dim, spacedim> dof_handler(tria);

  Tet::FE_DGQ<dim> fe(degree);
  dof_handler.distribute_dofs(fe);

  Tet::QGauss<dim - 1> face_quad(dim == 2 ? (degree == 1 ? 2 : 3) :
                                            (degree == 1 ? 3 : 7));

  const Tet::FE_Q<dim>            fe_mapping(1);
  const MappingIsoparametric<dim> mapping(fe_mapping);

  const UpdateFlags flags = update_JxW_values | update_normal_vectors |
                            update_quadrature_points | update_values |
                            update_gradients;

  FEFaceValues<dim, spacedim> fe_face_values_1(mapping, fe, face_quad, flags);
  FEFaceValues<dim, spacedim> fe_face_values_2(mapping, fe, face_quad, flags);

  Vector<double> global_vector(dof_handler.n_dofs());

  VectorTools::interpolate(mapping, dof_handler, Fu<dim>(0), global_vector);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto face : cell->face_indices())
        {
          if (cell->at_boundary(face))
            continue;

          fe_face_values_1.reinit(cell, face);
          fe_face_values_2.reinit(cell->neighbor(face),
                                  cell->neighbor_face_no(face));

          unsigned int face_index_1 = 0;
          unsigned int face_index_2 = 0;

          face_index_1 = face;
          face_index_2 = cell->neighbor_face_no(face);

          {
            std::vector<double> local_vector_1(face_quad.size());
            std::vector<double> local_vector_2(face_quad.size());

            fe_face_values_1.get_function_values(global_vector, local_vector_1);
            fe_face_values_2.get_function_values(global_vector, local_vector_2);

            for (auto i : local_vector_1)
              deallog << i << " ";
            deallog << std::endl;

            for (auto i : local_vector_2)
              deallog << i << " ";
            deallog << std::endl;

            deallog << ((local_vector_1 == local_vector_2) ? 1 : 0)
                    << std::endl;
          }

          {
            std::vector<Tensor<1, spacedim, double>> local_vector_1(
              face_quad.size());
            std::vector<Tensor<1, spacedim, double>> local_vector_2(
              face_quad.size());

            fe_face_values_1.get_function_gradients(global_vector,
                                                    local_vector_1);
            fe_face_values_2.get_function_gradients(global_vector,
                                                    local_vector_2);

            for (auto i : local_vector_1)
              deallog << i << " ";
            deallog << std::endl;

            for (auto i : local_vector_2)
              deallog << i << " ";
            deallog << std::endl;

            deallog << ((local_vector_1 == local_vector_2) ? 1 : 0)
                    << std::endl;
          }

          {
            for (auto i : fe_face_values_1.get_normal_vectors())
              deallog << i << " ";
            deallog << std::endl;

            for (auto i : fe_face_values_2.get_normal_vectors())
              deallog << i << " ";
            deallog << std::endl << std::endl;
          }

          {
            for (auto i : fe_face_values_1.get_JxW_values())
              deallog << i << " ";
            deallog << std::endl;

            for (auto i : fe_face_values_2.get_JxW_values())
              deallog << i << " ";
            deallog << std::endl << std::endl;
          }
        }
    }

  {
    Vector<double> global_vector_0(dof_handler.n_dofs());
    Vector<double> global_vector_1(dof_handler.n_dofs());
    Vector<double> global_vector_2(dof_handler.n_dofs());
    VectorTools::interpolate(mapping, dof_handler, Fu<dim>(0), global_vector_0);
    VectorTools::interpolate(mapping, dof_handler, Fu<dim>(1), global_vector_1);
    VectorTools::interpolate(mapping, dof_handler, Fu<dim>(2), global_vector_2);



    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        for (const auto face : cell->face_indices())
          {
            fe_face_values_1.reinit(cell, face);

            std::vector<double> local_vector_0(face_quad.size());
            std::vector<double> local_vector_1(face_quad.size());
            std::vector<double> local_vector_2(face_quad.size());

            fe_face_values_1.get_function_values(global_vector_0,
                                                 local_vector_0);
            fe_face_values_1.get_function_values(global_vector_1,
                                                 local_vector_1);
            fe_face_values_1.get_function_values(global_vector_2,
                                                 local_vector_2);

            for (unsigned int i = 0; i < face_quad.size(); ++i)
              deallog << "(" << local_vector_0[i] << "|" << local_vector_1[i]
                      << "|" << local_vector_2[i] << ") ";
            deallog << std::endl;

            for (auto i : fe_face_values_1.get_quadrature_points())
              deallog << "(" << i[0] << "|" << i[1] << "|" << i[2] << ") ";
            deallog << std::endl;
          }
      }
  }
}

int
main()
{
  initlog();

  test_3(1 /*degree*/);
  // test_3(2 /*degree*/);
}
