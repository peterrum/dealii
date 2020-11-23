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
  RightHandSideFunction(const unsigned int n_components)
    : Function<dim>(n_components)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const
  {
    return p[component];
  }
};


int
main(int argc, char **argv)
{
  initlog();

  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim    = 2; // only 2D is working for now
  const unsigned int degree = 1;

  // create mesh, select relevant FEM ingredients, and set up DoFHandler
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, dim == 2 ? 16 : 8);

  Simplex::FE_P<dim>   fe(degree);
  Simplex::QGauss<dim> quad(degree + 1);
  MappingFE<dim>       mapping(Simplex::FE_P<dim>(1));

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Create constraint matrix
  AffineConstraints<double> constraints;
  constraints.close();

  // initialize MatrixFree
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags = update_gradients | update_values;
  additional_data.mapping_update_flags_boundary_faces =
    update_gradients | update_values;
  additional_data.mapping_update_flags_inner_faces =
    update_gradients | update_values;

  MatrixFree<dim, double> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quad, additional_data);

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  VectorType x, b;
  matrix_free.initialize_dof_vector(x);
  matrix_free.initialize_dof_vector(b);



  VectorTools::interpolate(mapping,
                           dof_handler,
                           RightHandSideFunction<dim>(1),
                           b);


  matrix_free.template loop<VectorType, VectorType>(
    [&](const auto &, auto &, const auto &, const auto) {},
    [&](const auto &, auto &, const auto &src, const auto cells) {
      FEFaceEvaluation<dim, -1, 0, 1, double> phi_m(matrix_free, true);
      FEFaceEvaluation<dim, -1, 0, 1, double> phi_p(matrix_free, false);
      for (unsigned int cell = cells.first; cell < cells.second; ++cell)
        {
          phi_m.reinit(cell);
          phi_m.gather_evaluate(src, true, false);

          phi_p.reinit(cell);
          phi_p.gather_evaluate(src, true, false);

          for (unsigned int q = 0; q < phi_m.n_q_points; ++q)
            deallog << phi_m.get_value(q) << " " << phi_p.get_value(q)
                    << std::endl;
        }
    },
    [&](const auto &, auto &, const auto &, const auto) {},
    x,
    b,
    true);
}