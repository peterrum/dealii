// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2021 by the deal.II authors
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

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"


// Test FEFaceEvaluation::read_dof_values() and
// FEFaceEvaluation::gather_evaluate() for ECL for two cells.
//
// @note Since this program assumes that both cells are within the same
//   macro cell, this test is only run if vectorization is enabled.

template <int dim, int fe_degree, int n_points, typename Number>
void
test(const unsigned int n_refinements)
{
  using VectorizedArrayType = VectorizedArray<Number>;

  using VectorType = LinearAlgebra::Vector<Number>;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(n_refinements);

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  MappingQ<dim> mapping(1);
  QGauss<1>     quad(n_points);

  AffineConstraints<Number> constraint;

  using MF = MatrixFree<dim, Number, VectorizedArrayType>;

  typename MF::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_values | update_quadrature_points;

  MF matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraint, quad, additional_data);

  VectorType src, dst;

  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst);

  std::vector<std::tuple<unsigned int, unsigned int, unsigned int>> indices;

  for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
    {
      for (unsigned int i = 0;
           i < matrix_free.n_active_entries_per_cell_batch(cell);
           ++i)
        indices.emplace_back(indices.size(), cell, i);
    }

  std::sort(indices.begin(), indices.end(), [](const auto &a, const auto &b) {
    if (std::get<2>(a) != std::get<2>(b))
      return std::get<2>(a) < std::get<2>(b);

    return std::get<1>(a) < std::get<1>(b);
  });

  std::vector<std::vector<Point<dim>>> quadrature_points_ref;

  {
    FEEvaluation<dim, fe_degree, n_points, 1, Number, VectorizedArrayType> phi(
      matrix_free);

    for (unsigned int cell = 0; cell < matrix_free.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);

        for (unsigned int v = 0;
             v < matrix_free.n_active_entries_per_cell_batch(cell);
             ++v)
          {
            std::vector<Point<dim>> points;

            for (unsigned int i = 0; i < phi.n_q_points; ++i)
              {
                auto temp_v = phi.quadrature_point(i);

                Point<dim> temp;
                for (unsigned int d = 0; d < dim; ++d)
                  temp[d] = temp_v[d][v];

                points.emplace_back(temp);
              }

            quadrature_points_ref.emplace_back(points);
          }
      }
  }



  {
    FEEvaluation<dim, fe_degree, n_points, 1, Number, VectorizedArrayType> phi(
      matrix_free);

    for (unsigned int v = 0; v < indices.size();
         v += VectorizedArrayType::size())
      {
        std::array<unsigned int, VectorizedArrayType::size()> indices_;

        indices_.fill(numbers::invalid_unsigned_int);

        const unsigned int n_lanes_filled =
          std::min(v + VectorizedArrayType::size(), indices.size()) - v;

        for (unsigned int i = v, c = 0; i < v + n_lanes_filled; ++i, ++c)
          indices_[c] = std::get<1>(indices[i]) * VectorizedArrayType::size() +
                        std::get<2>(indices[i]);

        for (const auto i : indices_)
          std::cout << i << " ";
        std::cout << std::endl;

        phi.reinit(indices_);


        for (unsigned int i = v, c = 0; i < v + n_lanes_filled; ++i, ++c)
          {
            std::vector<Point<dim>> points;

            for (unsigned int i = 0; i < phi.n_q_points; ++i)
              {
                auto temp_v = phi.quadrature_point(i);

                Point<dim> temp;
                for (unsigned int d = 0; d < dim; ++d)
                  temp[d] = temp_v[d][c];

                points.emplace_back(temp);
              }

            Assert(points == quadrature_points_ref[std::get<0>(indices[i])],
                   ExcInternalError());
          }
      }
  }

  deallog << "OK!" << std::endl;
}

int
main()
{
  initlog();
  test<2, 1, 2, double>(2);
}
