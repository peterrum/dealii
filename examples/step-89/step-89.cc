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
 *          Marco Feder, SISSA, 2023
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
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
// The following header file provides the class FERemoteEvaluation, which allows
// to access values and/or gradients at remote triangulations similar to
// FEEvaluation.
// #include <deal.II/matrix_free/fe_remote_evaluation.h>
#include "fe_remote_evaluation.h"

#include <iostream>

// We pack everything that is specific for this program into a namespace
// of its own.
namespace Step89
{
  using namespace dealii;

  template <typename FERemoteEvaluationCommunicatorType>
  class AcousticConservationEquation
  {
  public:
    AcousticConservationEquation(
      const FERemoteEvaluationCommunicatorType &remote_comm,
      const double                              density,
      const double                              speed_of_sound)
      : remote_communicator(remote_comm)
      , rho(density)
      , c(speed_of_sound)
      , tau(0.5 * rho * c)
      , gamma(0.5 / (rho * c))
    {}

    template <int dim, typename Number, typename VectorType>
    void evaluate(const MatrixFree<dim, Number> &matrix_free,
                  VectorType                    &dst,
                  const VectorType              &src) const
    {
      matrix_free.loop(&AcousticConservationEquation::cell_loop,
                       &AcousticConservationEquation::face_loop,
                       &AcousticConservationEquation::boundary_face_loop,
                       this,
                       dst,
                       src,
                       true,
                       MatrixFree<dim, Number>::DataAccessOnFaces::values,
                       MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

  private:
    template <int dim, typename Number, typename VectorType>
    void
    cell_loop(const MatrixFree<dim, Number>               &matrix_free,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, 1, Number>   pressure(matrix_free, 0, 0, 0);
      FEEvaluation<dim, -1, 0, dim, Number> velocity(matrix_free, 0, 0, 1);

      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          velocity.reinit(cell);
          pressure.reinit(cell);

          pressure.gather_evaluate(src, EvaluationFlags::gradients);
          velocity.gather_evaluate(src, EvaluationFlags::gradients);

          for (unsigned int q = 0; q < pressure.n_q_points; ++q)
            {
              pressure.submit_value(rho * c * c * velocity.get_divergence(q),
                                    q);
              velocity.submit_value(1.0 / rho * pressure.get_gradient(q), q);
            }

          pressure.integrate_scatter(EvaluationFlags::values, dst);
          velocity.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    template <int dim, typename Number, typename VectorType>
    void
    face_loop(const MatrixFree<dim, Number>               &matrix_free,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, 1, Number> pressure_m(
        matrix_free, true, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, 1, Number> pressure_p(
        matrix_free, false, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, dim, Number> velocity_m(
        matrix_free, true, 0, 0, 1);
      FEFaceEvaluation<dim, -1, 0, dim, Number> velocity_p(
        matrix_free, false, 0, 0, 1);

      for (unsigned int face = face_range.first; face < face_range.second;
           face++)
        {
          velocity_m.reinit(face);
          velocity_p.reinit(face);

          pressure_m.reinit(face);
          pressure_p.reinit(face);

          pressure_m.gather_evaluate(src, EvaluationFlags::values);
          pressure_p.gather_evaluate(src, EvaluationFlags::values);

          velocity_m.gather_evaluate(src, EvaluationFlags::values);
          velocity_p.gather_evaluate(src, EvaluationFlags::values);

          for (unsigned int q : pressure_m.quadrature_point_indices())
            {
              const auto &n  = pressure_m.normal_vector(q);
              const auto &pm = pressure_m.get_value(q);
              const auto &um = velocity_m.get_value(q);
              const auto &pp = pressure_p.get_value(q);
              const auto &up = velocity_p.get_value(q);

              const auto flux_momentum =
                0.5 * (pm + pp) + 0.5 * tau * (um - up) * n;
              velocity_m.submit_value(1.0 / rho * (flux_momentum - pm) * n, q);
              velocity_p.submit_value(1.0 / rho * (flux_momentum - pp) * (-n),
                                      q);

              const auto flux_mass =
                0.5 * (um + up) + 0.5 * gamma * (pm - pp) * n;
              pressure_m.submit_value(rho * c * c * (flux_mass - um) * n, q);
              pressure_p.submit_value(rho * c * c * (flux_mass - up) * (-n), q);
            }


          pressure_m.integrate_scatter(EvaluationFlags::values, dst);
          pressure_p.integrate_scatter(EvaluationFlags::values, dst);
          velocity_m.integrate_scatter(EvaluationFlags::values, dst);
          velocity_p.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    template <int dim, typename Number, typename VectorType>
    void boundary_face_loop(
      const MatrixFree<dim, Number>               &matrix_free,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      // @PETER/@MARCO: Doing it here is problematic if a lot of remote values
      // are used sind memory is allocated every time this function is
      // called. On the other hand this way we can get rid of defineing FERemoteEvaluation
      // mutable. Handing in a cache during construction of FERemoteEvaluation objects would
      // still require the cache to be mutable :/. This is similar to FEPointEval but, but here
      // much less memory has to be allocated
      
      FERemoteEvaluation<FERemoteEvaluationCommunicatorType, 1> pressure_r(
        remote_communicator,
        matrix_free.get_dof_handler(),
        VectorTools::EvaluationFlags::avg,
        0 /*first selected comp*/);
      FERemoteEvaluation<FERemoteEvaluationCommunicatorType, dim> velocity_r(
        remote_communicator,
        matrix_free.get_dof_handler(),
        VectorTools::EvaluationFlags::avg,
        1 /*first selected comp*/);

      // @PETER/@MARCO: having the call here is nice in my opinion
      pressure_r.gather_evaluate(src, EvaluationFlags::values);
      velocity_r.gather_evaluate(src, EvaluationFlags::values);

      FEFaceEvaluation<dim, -1, 0, 1, Number> pressure_m(
        matrix_free, true, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, dim, Number> velocity_m(
        matrix_free, true, 0, 0, 1);

      for (unsigned int face = face_range.first; face < face_range.second;
           face++)
        {
          velocity_m.reinit(face);
          pressure_m.reinit(face);

          pressure_m.gather_evaluate(src, EvaluationFlags::values);
          velocity_m.gather_evaluate(src, EvaluationFlags::values);

          // TOOD: 90 should not be hard coded
          if (matrix_free.get_boundary_id(face) > 90) 
            {
              for (unsigned int q : pressure_m.quadrature_point_indices())
                {
                  const auto &n  = pressure_m.normal_vector(q);
                  const auto &pm = pressure_m.get_value(q);
                  const auto &um = velocity_m.get_value(q);

                  // @PETER/@MARCO: this interface should definitely stay like this
                  velocity_r.reinit(face);
                  pressure_r.reinit(face);
                  const auto &pp = pressure_r.get_value(q);
                  const auto &up = velocity_r.get_value(q);

                  const auto &flux_momentum =
                    0.5 * (pm + pp) + 0.5 * tau * (um - up) * n;
                  velocity_m.submit_value(1.0 / rho * (flux_momentum - pm) * n,
                                          q);

                  const auto &flux_mass =
                    0.5 * (um + up) + 0.5 * gamma * (pm - pp) * n;
                  pressure_m.submit_value(rho * c * c * (flux_mass - um) * n,
                                          q);
                }
            }
          else
            {
              for (unsigned int q : pressure_m.quadrature_point_indices())
                {
                  const auto &n  = pressure_m.normal_vector(q);
                  const auto &pm = pressure_m.get_value(q);
                  const auto &um = velocity_m.get_value(q);

                  // homogenous boundary conditions
                  const auto &pp = -pm;
                  const auto &up = um;

                  const auto &flux_momentum =
                    0.5 * (pm + pp) + 0.5 * tau * (um - up) * n;
                  velocity_m.submit_value(1.0 / rho * (flux_momentum - pm) * n,
                                          q);

                  const auto &flux_mass =
                    0.5 * (um + up) + 0.5 * gamma * (pm - pp) * n;
                  pressure_m.submit_value(rho * c * c * (flux_mass - um) * n,
                                          q);
                }
            }

          pressure_m.integrate_scatter(EvaluationFlags::values, dst);
          velocity_m.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    const FERemoteEvaluationCommunicatorType &remote_communicator;
    const double                              rho;
    const double                              c;
    const double                              tau;
    const double                              gamma;
  };



  class InverseMassOperator
  {
  public:
    template <int dim, typename Number, typename VectorType>
    void apply(const dealii::MatrixFree<dim, Number> &matrix_free,
               VectorType                            &dst,
               const VectorType                      &src) const
    {
      dst.zero_out_ghost_values();
      matrix_free.cell_loop(&InverseMassOperator::cell_loop, this, dst, src);
    }

  private:
    template <int dim, typename Number, typename VectorType>
    void
    cell_loop(const dealii::MatrixFree<dim, Number>       &matrix_free,
              VectorType                                  &dst,
              const VectorType                            &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, dim + 1, Number> phi(matrix_free);
      MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim + 1, Number>
        minv(phi);

      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(src);
          minv.apply(phi.begin_dof_values(), phi.begin_dof_values());
          phi.set_dof_values(dst);
        }
    }
  };

  template <int dim,
            typename Number,
            typename FERemoteEvaluationCommunicatorType>
  class SpatialOperator
  {
  public:
    SpatialOperator(const MatrixFree<dim, Number>            &matrix_free_in,
                    const FERemoteEvaluationCommunicatorType &remote_comm,
                    const double                              density,
                    const double                              speed_of_sound)
      : matrix_free(matrix_free_in)
      , remote_communicator(remote_comm)
      , rho(Number{density})
      , c(Number{speed_of_sound})
    {}

    template <typename VectorType>
    void evaluate(VectorType &dst, const VectorType &src) const
    {
      AcousticConservationEquation(remote_communicator, rho, c)
        .evaluate(matrix_free, dst, src);
      dst *= Number{-1.0};
      InverseMassOperator().apply(matrix_free, dst, dst);
    }

  private:
    const MatrixFree<dim, Number>            &matrix_free;
    const FERemoteEvaluationCommunicatorType &remote_communicator;
    const Number                              rho;
    const Number                              c;
  };

  struct RungeKutta2
  {
    template <typename VectorType, typename Operator>
    static void perform_time_step(const Operator   &pde_operator,
                                  const double      dt,
                                  VectorType       &dst,
                                  const VectorType &src)
    {
      VectorType k1 = src;

      // stage 1
      pde_operator.evaluate(k1, src);

      // stage 2
      k1.sadd(0.5 * dt, 1.0, src);
      pde_operator.evaluate(dst, k1);
      dst.sadd(dt, 1.0, src);
    }
  };

  template <int dim,
            typename Number,

            typename VectorType>
  void
  set_initial_condition_vibrating_membrane(MatrixFree<dim, Number> matrix_free,
                                           const double            modes,
                                           VectorType             &dst)
  {
    class InitialSolution : public Function<dim>
    {
    public:
      InitialSolution(const double modes)
        : Function<dim>(dim + 1, 0.0)
        , M(modes)
      {
        static_assert(dim == 2, "Only implemented for dim==2");
      }

      double value(const Point<dim> &p, const unsigned int comp) const final
      {
        if (comp == 0)
          return std::sin(M * numbers::PI * p[0]) *
                 std::sin(M * numbers::PI * p[1]);
        else
          return 0.0;
      }

    private:
      const double M;
    };

    VectorTools::interpolate(*matrix_free.get_mapping_info().mapping,
                             matrix_free.get_dof_handler(),
                             InitialSolution(modes),
                             dst);
  }

  double
  compute_dt_cfl(const double hmin, const unsigned int degree, const double c)
  {
    return hmin / (std::pow(degree, 1.5) * c);
  }

  // @sect3{Point-to-point interpolation}
  //
  // Description
  void point_to_point_interpolation(const unsigned int refinements,
                                    const unsigned int degree)
  {
    using Number      = double;
    using VectorType  = LinearAlgebra::distributed::Vector<Number>;
    constexpr int dim = 2;

    const double density        = 1.0;
    const double speed_of_sound = 1.0;
    const double modes          = 120.0;
    const double length         = 0.1;

    const unsigned int subdiv_left  = 1;
    const unsigned int subdiv_right = 3;

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    Triangulation<dim>                        tria_left;
    Triangulation<dim>                        tria_right;
    GridGenerator::subdivided_hyper_rectangle(tria_left,
                                              {subdiv_left, 2 * subdiv_left},
                                              {0.0, 0.0},
                                              {0.5 * length, length});
    for (const auto &face : tria_left.active_face_iterators())
      if (face->at_boundary())
        {
          face->set_boundary_id(0);
          if (face->center()[0] > 0.5 * length - 1e-6)
            face->set_boundary_id(99);
        }
    
    GridGenerator::subdivided_hyper_rectangle(tria_right,
                                              {subdiv_right, 2 * subdiv_right},
                                              {0.5 * length, 0.0},
                                              {length, length});
    for (const auto &face : tria_right.active_face_iterators())
      if (face->at_boundary())
        {
          face->set_boundary_id(0);
          if (face->center()[0] < 0.5 * length + 1e-6)
            face->set_boundary_id(98);
        }
    
    GridGenerator::merge_triangulations(
      tria_left, tria_right, tria, 0., false, true);
    tria.refine_global(refinements);

    const MappingQ1<dim> mapping;
    const FESystem<dim>  fe_dgq(FE_DGQ<dim>(degree), dim + 1);
    const QGauss<dim>    quad(degree + 1);

    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe_dgq);

    AffineConstraints<Number> constraints;
    constraints.close();

    MatrixFree<dim, Number> matrix_free;

    typename MatrixFree<dim, Number>::AdditionalData data;
    data.mapping_update_flags = update_gradients | update_values;
    data.mapping_update_flags_inner_faces =
      update_quadrature_points | update_values;
    data.mapping_update_flags_boundary_faces =
      data.mapping_update_flags_inner_faces;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);


    double dt =
      0.1 * compute_dt_cfl(0.5 * length /
                             ((double)std::max(subdiv_left, subdiv_right)) /
                             std::pow(2, refinements),
                           degree,
                           speed_of_sound);

    VectorType solution;
    matrix_free.initialize_dof_vector(solution);
    set_initial_condition_vibrating_membrane(matrix_free, modes, solution);

    VectorType solution_temp;
    matrix_free.initialize_dof_vector(solution_temp);

    FEFaceEvaluation<dim, -1, 0> fe_eval(matrix_free);
    FERemoteEvaluationCommunicator<FEFaceEvaluation<dim, -1, 0>, true>
                                                       remote_communicator;

    // @PETER and @MARCO: As discussedit migh be better to leave the
    // intialization up to the user: the code of initialize_face_pairs would
    // end up here and the classes can be used in a more cutomizable way. What do you think?
    std::vector<std::pair<unsigned int, unsigned int>> face_pairs;
    face_pairs.push_back(std::make_pair(99, 98));
    face_pairs.push_back(std::make_pair(98, 99));
    remote_communicator.initialize_face_pairs(face_pairs, fe_eval);

    SpatialOperator<
      dim,
      Number,
      FERemoteEvaluationCommunicator<FEFaceEvaluation<dim, -1, 0>, true>>
      acoustic_operator(matrix_free,
                        remote_communicator,
                        density,
                        speed_of_sound);

    const double end_time = 2.0 / (modes * std::sqrt(dim) * speed_of_sound);
    double       time     = 0.0;
    unsigned int timestep = 0;

    while (time < end_time)
      {
        pcout << time << std::endl;
        std::swap(solution, solution_temp);
        time += dt;
        timestep++;
        RungeKutta2::perform_time_step(acoustic_operator,
                                       dt,
                                       solution,
                                       solution_temp);

        if (timestep % 10 == 0)
          {
            DataOut<dim>          data_out;
            DataOutBase::VtkFlags flags;
            flags.write_higher_order_cells = true;
            data_out.set_flags(flags);

            std::vector<
              DataComponentInterpretation::DataComponentInterpretation>
              interpretation(
                dim + 1,
                DataComponentInterpretation::component_is_part_of_vector);
            std::vector<std::string> names(dim + 1, "U");

            interpretation[0] =
              DataComponentInterpretation::component_is_scalar;
            names[0] = "P";

            data_out.add_data_vector(dof_handler,
                                     solution,
                                     names,
                                     interpretation);

            data_out.build_patches(mapping,
                                   degree,
                                   DataOut<dim>::curved_inner_cells);
            data_out.write_vtu_in_parallel("example_1" +
                                             std::to_string(timestep) + ".vtu",
                                           MPI_COMM_WORLD);
          }
      }
  }


  // // @sect3{Nitsche-type mortaring}
  // //
  // // Description
  // void nitsche_type_mortaring()
  // {}

} // namespace Step89


// @sect3{Driver}
//
// Finally, the driver executes the different versions of handling non-matching
// interfaces.

int main(int argc, char *argv[])
{
  dealii::Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  std::cout.precision(5);

  Step89::point_to_point_interpolation(4, 5);
  // Step89::nitsche_type_mortaring();
  // Step89::inhomogenous_material();

  return 0;
}
