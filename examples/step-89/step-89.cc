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


#include <iostream>
#include <fstream>

// The following header file provides the class FERemoteEvaluation, which allows
// to access values and/or gradients at remote triangulations similar to
// FEEvaluation.
// #include <deal.II/matrix_free/fe_remote_evaluation.h>

// We pack everything that is specific for this program into a namespace
// of its own.
namespace Step89
{
  using namespace dealii;

  template <int dim, typename Number, typename VectorizedArrayType>
  class AcousticConservationEquation
  {
    using This = AcousticConservationEquation<dim, Number, VectorizedArrayType>;

  public:
    AcousticConservationEquation(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      const double                                        density,
      const double                                        speed_of_sound)
      : mf(matrix_free)
      , rho(density)
      , c(speed_of_sound)
      , tau(0.5 * rho * c)
      , gamma(0.5 / (rho * c))
    {}

    template <typename VectorType>
    void evaluate(VectorType &dst, const VectorType &src) const
    {
      mf.loop(
        &This::cell_loop,
        &This::face_loop,
        &This::boundary_face_loop,
        this,
        dst,
        src,
        true,
        MatrixFree<dim, Number, VectorizedArrayType>::DataAccessOnFaces::values,
        MatrixFree<dim, Number, VectorizedArrayType>::DataAccessOnFaces::
          values);
    }

  private:
    template <typename VectorType>
    void
    cell_loop(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
              VectorType                                         &dst,
              const VectorType                                   &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> pressure(
        matrix_free, 0, 0, 0);
      FEEvaluation<dim, -1, 0, dim, Number, VectorizedArrayType> velocity(
        matrix_free, 0, 0, 1);

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

    template <typename VectorType>
    void
    face_loop(const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
              VectorType                                         &dst,
              const VectorType                                   &src,
              const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> pressure_m(
        matrix_free, true, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> pressure_p(
        matrix_free, false, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, dim, Number, VectorizedArrayType> velocity_m(
        matrix_free, true, 0, 0, 1);
      FEFaceEvaluation<dim, -1, 0, dim, Number, VectorizedArrayType> velocity_p(
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

              // homogenous boundary conditions
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

    template <typename VectorType>
    void boundary_face_loop(
      const MatrixFree<dim, Number, VectorizedArrayType> &matrix_free,
      VectorType                                         &dst,
      const VectorType                                   &src,
      const std::pair<unsigned int, unsigned int>        &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> pressure_m(
        matrix_free, true, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, dim, Number, VectorizedArrayType> velocity_m(
        matrix_free, true, 0, 0, 1);

      for (unsigned int face = face_range.first; face < face_range.second;
           face++)
        {
          velocity_m.reinit(face);
          pressure_m.reinit(face);

          pressure_m.gather_evaluate(src, EvaluationFlags::values);
          velocity_m.gather_evaluate(src, EvaluationFlags::values);

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
              velocity_m.submit_value(1.0 / rho * (flux_momentum - pm) * n, q);

              const auto &flux_mass =
                0.5 * (um + up) + 0.5 * gamma * (pm - pp) * n;
              pressure_m.submit_value(rho * c * c * (flux_mass - um) * n, q);
            }

          pressure_m.integrate_scatter(EvaluationFlags::values, dst);
          velocity_m.integrate_scatter(EvaluationFlags::values, dst);
        }
    }

    const MatrixFree<dim, Number, VectorizedArrayType> &mf;
    const double                                        rho;
    const double                                        c;
    const double                                        tau;
    const double                                        gamma;
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
      k1.sadd(-0.5 * dt, 1.0, src);
      pde_operator.evaluate(dst, k1);
      dst.sadd(-dt, 1.0, src);
    }
  };

  template <int dim,
            typename Number,
            typename VectorizedArrayType,
            typename VectorType>
  void set_initial_condition_vibrating_membrane(
    MatrixFree<dim, Number, VectorizedArrayType> matrix_free,
    const double                                 modes,
    VectorType                                  &dst)
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
    using Number              = double;
    using VectorType          = LinearAlgebra::distributed::Vector<Number>;
    using VectorizedArrayType = VectorizedArray<Number>;
    constexpr int dim         = 2;

    const double density        = 1.0;
    const double speed_of_sound = 1.0;
    const double modes          = 120.0;
    const double length         = 0.1;

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
    GridGenerator::hyper_cube(tria, 0.0, length);
    tria.refine_global(refinements);

    const MappingQ1<dim> mapping;
    const FESystem<dim>  fe_dgq(FE_DGQ<dim>(degree), dim + 1);
    const QGauss<dim>    quad(degree + 1);

    DoFHandler<dim> dof_handler(tria);
    dof_handler.distribute_dofs(fe_dgq);

    AffineConstraints<Number> constraints;
    constraints.close();

    MatrixFree<dim, Number, VectorizedArrayType> matrix_free;

    typename MatrixFree<dim, Number, VectorizedArrayType>::AdditionalData data;
    data.mapping_update_flags             = update_gradients | update_values;
    data.mapping_update_flags_inner_faces = update_values;
    data.mapping_update_flags_boundary_faces =
      data.mapping_update_flags_inner_faces;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);


    double dt =
      0.1 *
      compute_dt_cfl(length / std::pow(2, refinements), degree, speed_of_sound);

    VectorType solution;
    matrix_free.initialize_dof_vector(solution);
    set_initial_condition_vibrating_membrane(matrix_free, modes, solution);

    VectorType solution_temp;
    matrix_free.initialize_dof_vector(solution_temp);

    AcousticConservationEquation<dim, Number, VectorizedArrayType>
      acoustic_operator(matrix_free, density, speed_of_sound);

    const double end_time = 0.4;
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
        DataOut<dim>          data_out;
        DataOutBase::VtkFlags flags;
        flags.write_higher_order_cells = true;
        data_out.set_flags(flags);

        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          interpretation(
            dim + 1, DataComponentInterpretation::component_is_part_of_vector);
        std::vector<std::string> names(dim + 1, "U");

        interpretation[0] = DataComponentInterpretation::component_is_scalar;
        names[0]          = "P";

        data_out.add_data_vector(dof_handler, solution, names, interpretation);

        data_out.build_patches(mapping,
                               degree,
                               DataOut<dim>::curved_inner_cells);
        data_out.write_vtu_in_parallel("example_1" + std::to_string(timestep) +
                                         ".vtu",
                                       MPI_COMM_WORLD);
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

  Step89::point_to_point_interpolation(2, 3);
  // Step89::nitsche_type_mortaring();
  // Step89::inhomogenous_material();

  return 0;
}
