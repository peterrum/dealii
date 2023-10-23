/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 *
 * Authors: Johannes Heinz, TU Wien, 2023
 *          Peter Munch, University of Augsburg, 2023
 */

// @sect3{Include files}
//
// The program starts with including all the relevant header files.
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/numerics/data_out.h>

#include <iostream>
#include <fstream>

// The following header file provides the class FERemoteEvaluation, which allows
// to access values and/or gradients at remote triangulations similar to FEEvaluation.
#include <deal.II/matrix_free/fe_remote_evaluation.h>

// We pack everything that is specific for this program into a namespace
// of its own.
namespace Step89
{
  using namespace dealii;

  template<int dim, typename Number, typename VectorizedArrayType>
  class AcousticConservationEquation
  {

    using VectorType  = dealii::LinearAlgebra::distributed::Vector<Number>;
  using This = AcousticConservationEquation<dim, Number>;

  public:
    void
  evaluate(MatrixFree<dim, Number> const &   matrix_free,VectorType & dst, VectorType const & src) const
    {
      matrix_free.loop(&This::cell_loop,
                    &This::face_loop,
                    &This::boundary_face_loop,
                    this,
                    dst,
                    src,
                    true,
                    dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values,
                    dealii::MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

  private:
void
cell_loop(dealii::MatrixFree<dim, Number,VectorizedArrayType> const & matrix_free,
                                         VectorType &              dst,
                                         VectorType const &                 src,
                                         std::pair<unsigned int, unsigned int> const &  cell_range) const
{
  FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>pressure(matrix_free, 0,0,0);
  FEEvaluation<dim, -1, 0, dim, Number, VectorizedArrayType>velocity(matrix_free,0,0,1);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    velocity.reinit(cell);
    pressure.reinit(cell);

    pressure.gather_evaluate(src.block(0), dealii::EvaluationFlags::gradients);
    velocity.gather_evaluate(src.block(1), dealii::EvaluationFlags::gradients);

    do_cell_integral_strong(pressure, velocity);

    for(unsigned int q = 0; q < pressure.n_q_points; ++q)
    {
      pressure.submit_value(rho*c*c * velocity.get_divergence(q), q);
      velocity.submit_value(1.0/rho * pressure.get_gradient(q), q);
    }

    pressure.integrate_scatter(dealii::EvaluationFlags::values, dst);
    velocity.integrate_scatter(dealii::EvaluationFlags::values, dst);

  }
}

    void
face_loop(dealii::MatrixFree<dim, Number,VectorizedArrayType> const & matrix_free,
                                         VectorType &              dst,
                                         VectorType const &                 src,
                                         std::pair<unsigned int, unsigned int> const &  face_range) const
{
  FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>pressure_m(matrix_free, true,0,0,0);
  FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>pressure_p(matrix_free, false,0,0,0);
  FEFaceEvaluation<dim, -1, 0, dim, Number, VectorizedArrayType>velocity_m(matrix_free,true,0,0,1);
  FEFaceEvaluation<dim, -1, 0, dim, Number, VectorizedArrayType>velocity_p(matrix_free,false,0,0,1);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      velocity_m.reinit(face);
      velocity_p.reinit(face);

      pressure_m.reinit(face);
      pressure_p.reinit(face);

        for(unsigned int q : pressure_m.quadrature_point_indices())
  {
   const auto& pm = pressure_m.get_value(q);
   const auto&pp = pressure_p.get_value(q);
  const auto& um = velocity_m.get_value(q);
   const auto& up = velocity_p.get_value(q);
  const auto& n  = pressure_m.normal_vector(q);

  const auto & flux_momentum = 0.5 * (pm + pp) + 0.5*tau*(um-up)*n;
      velocity_m.submit_value(1.0/rho * (flux_momentum - pm) * n, q);
      velocity_p.submit_value(1.0/rho * (flux_momentum - pp) * (-n), q);
  
  const auto & flux_mass = 0.5 * (um + up) + 0.5*gamma*(pm - pp)*n;
      pressure_m.submit_value(rho*c*c * (flux_mass - um) * n, q);
      pressure_p.submit_value(rho*c*c * (flux_mass - up) * (-n), q);
  }

            pressure_m.gather_evaluate(src.block(0), dealii::EvaluationFlags::values);
      pressure_p.gather_evaluate(src.block(0), dealii::EvaluationFlags::values);

            velocity_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
      velocity_p.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
}

    void
boundary_face_loop(dealii::MatrixFree<dim, Number,VectorizedArrayType> const & matrix_free,
                                         VectorType &              dst,
                                         VectorType const &                 src,
                                         std::pair<unsigned int, unsigned int> const &  face_range) const
{
  FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType>pressure_m(matrix_free, true,0,0,0);
  FEFaceEvaluation<dim, -1, 0, dim, Number, VectorizedArrayType>velocity_m(matrix_free,true,0,0,1);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      velocity_m.reinit(face);

      pressure_m.reinit(face);

        for(unsigned int q : pressure_m.quadrature_point_indices())
  {
   const auto& pm = pressure_m.get_value(q);
   const auto&pp = -pm;//homogenous boundary conditions
  const auto& um = velocity_m.get_value(q);
   const auto& up = up;
  const auto& n  = pressure_m.normal_vector(q);

  const auto & flux_momentum = 0.5 * (pm + pp) + 0.5*tau*(um-up)*n;
      velocity_m.submit_value(1.0/rho * (flux_momentum - pm) * n, q);
      velocity_p.submit_value(1.0/rho * (flux_momentum - pp) * (-n), q);
  
  const auto & flux_mass = 0.5 * (um + up) + 0.5*gamma*(pm - pp)*n;
      pressure_m.submit_value(rho*c*c * (flux_mass - um) * n, q);
      pressure_p.submit_value(rho*c*c * (flux_mass - up) * (-n), q);
  }
        
            pressure_m.gather_evaluate(src.block(0), dealii::EvaluationFlags::values);
            velocity_m.integrate_scatter(dealii::EvaluationFlags::values, dst);
    }
}

    
  };


  // @sect3{Point-to-point interpolation}
  //
  // Description
  void point_to_point_interpolation()
  {
    constexpr unsigned int dim       = 2;
    constexpr unsigned int fe_degree = 3;

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);
  }


  // @sect3{Nitsche-type mortaring}
  //
  // Description
  void nitsche_type_mortaring()
  {

  }
  
} // namespace Step87


// @sect3{Driver}
//
// Finally, the driver executes the different versions of handling non-matching interfaces.

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  std::cout.precision(5);

  Step87::point_to_point_interpolation();
  Step87::nitsche_type_mortaring();
  Step87::inhomogenous_material();
  
  return 0;
}
