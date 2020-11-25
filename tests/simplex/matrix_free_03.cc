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

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/grid_generator.h>
#include <deal.II/simplex/quadrature_lib.h>

#include "../tests.h"

using namespace dealii;

namespace util
{
  /**
   * Extract communicator of @p mesh.
   */
  template <typename MeshType>
  MPI_Comm
  get_mpi_comm(const MeshType &mesh)
  {
    const auto *tria_parallel = dynamic_cast<
      const parallel::TriangulationBase<MeshType::dimension,
                                        MeshType::space_dimension> *>(
      &(mesh.get_triangulation()));

    return tria_parallel != nullptr ? tria_parallel->get_communicator() :
                                      MPI_COMM_SELF;
  }

  template <int dim, int spacedim>
  void
  create_reentrant_corner(Triangulation<dim, spacedim> &tria)
  {
    const unsigned int n_refinements = 1;

    std::vector<unsigned int> repetitions(dim, 2);
    Point<dim>                bottom_left, top_right;
    for (unsigned int d = 0; d < dim; ++d)
      {
        bottom_left[d] = -1.;
        top_right[d]   = 1.;
      }
    std::vector<int> cells_to_remove(dim, 1);
    cells_to_remove[0] = -1;

    GridGenerator::subdivided_hyper_L(
      tria, repetitions, bottom_left, top_right, cells_to_remove);

    tria.refine_global(n_refinements);
  }

} // namespace util



template <int dim>
class RightHandSideFunction : public Function<dim>
{
public:
  RightHandSideFunction(const unsigned int component)
    : Function<dim>(1)
    , component(component)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int = 0) const
  {
    return p[component];
  }

  const unsigned int component;
};


int
main(int argc, char **argv)
{
  initlog();

  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim    = 3; // only 2D is working for now
  const unsigned int degree = 1;

  // create mesh, select relevant FEM ingredients, and set up DoFHandler
  Triangulation<dim> tria;

#if 1
  GridGenerator::subdivided_hyper_cube_with_simplices(tria,
                                                      1 /*dim == 2 ? 16 : 8*/);

  Simplex::FE_DGP<dim> fe(degree);
  Simplex::QGauss<dim> quad(degree + 1);
  MappingFE<dim>       mapping(Simplex::FE_P<dim>(1));
#else
  GridGenerator::subdivided_hyper_cube(tria, 1 /*dim == 2 ? 16 : 8*/);

  FE_DGQ<dim>    fe(degree);
  QGauss<dim>    quad(degree + 1);
  MappingFE<dim> mapping(FE_Q<dim>(1));

#endif

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Create constraint matrix
  AffineConstraints<double> constraints;
  constraints.close();

  // initialize MatrixFree
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags = update_gradients | update_values;
  additional_data.mapping_update_flags_boundary_faces =
    update_gradients | update_values | update_quadrature_points;
  additional_data.mapping_update_flags_inner_faces =
    update_gradients | update_values | update_quadrature_points;

  MatrixFree<dim, double> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quad, additional_data);

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  for (unsigned int i = 0; i < dim; ++i)
    {
      VectorType x, b;
      matrix_free.initialize_dof_vector(x);
      matrix_free.initialize_dof_vector(b);

      VectorTools::interpolate(mapping,
                               dof_handler,
                               RightHandSideFunction<dim>(i),
                               b);

      const auto print = [](const auto &val, const auto lanes) {
        for (unsigned int i = 0; i < lanes; ++i)
          deallog << val[i] << " ";
        deallog << std::endl;
      };


      matrix_free.template loop<VectorType, VectorType>(
        [&](const auto &, auto &, const auto &, const auto) {},
        [&](const auto &, auto &, const auto &src, const auto cells) {
          FEFaceEvaluation<dim, -1, 0, 1, double> phi_m(matrix_free, true);
          FEFaceEvaluation<dim, -1, 0, 1, double> phi_p(matrix_free, false);
          for (unsigned int cell = cells.first; cell < cells.second; ++cell)
            {
              phi_m.reinit(cell);
              phi_m.gather_evaluate(src, true, true);

              phi_p.reinit(cell);
              phi_p.gather_evaluate(src, true, true);

              const unsigned int n_lanes =
                matrix_free.n_active_entries_per_face_batch(cell);

              if (true)
                for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
                  {
                    deallog << "I ";
                    print(phi_m.quadrature_point(q)[i], n_lanes);
                    deallog << "I ";
                    print(phi_m.get_value(q), n_lanes);
                    deallog << "I ";
                    print(phi_p.quadrature_point(q)[i], n_lanes);
                    deallog << "I ";
                    print(phi_p.get_value(q), n_lanes);
                    deallog << "I " << std::endl;
                  }

              if (false)
                for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
                  {
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        deallog << "I " << phi_m.get_gradient(q)[d]
                                << std::endl;
                        deallog << "I " << phi_p.get_gradient(q)[d]
                                << std::endl;
                        // deallog << "I " <<  std::endl;
                      }
                    deallog << "I " << std::endl;
                  }
            }
        },
        [&](const auto &, auto &, const auto &src, const auto cells) {
          FEFaceEvaluation<dim, -1, 0, 1, double> phi_m(matrix_free, true);
          for (unsigned int cell = cells.first; cell < cells.second; ++cell)
            {
              phi_m.reinit(cell);
              phi_m.gather_evaluate(src, true, true);

              const unsigned int n_lanes =
                matrix_free.n_active_entries_per_face_batch(cell);

              if (true)
                for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
                  {
                    deallog << "B ";
                    print(phi_m.quadrature_point(q)[i], n_lanes);
                    deallog << "B ";
                    print(phi_m.get_value(q), n_lanes);
                    deallog << "B " << std::endl;
                  }

              if (false)
                for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
                  {
                    for (unsigned int d = 0; d < dim; ++d)
                      {
                        deallog << "B " << phi_m.get_gradient(q)[d]
                                << std::endl;
                        // deallog << "B "  << phi_m.begin_gradients ()[q + d *
                        // phi_m.n_q_points] << std::endl; deallog << "B "  <<
                        // std::endl;
                      }
                    deallog << "B " << std::endl;
                  }
            }
        },
        x,
        b,
        true);

      deallog << std::endl << std::endl << std::endl;
    }
}
