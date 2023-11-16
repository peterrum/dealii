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
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>


#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/non_matching/mapping_info.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
// The following header file provides the class FERemoteEvaluation, which allows
// to access values and/or gradients at remote triangulations similar to
// FEEvaluation.
// #include <deal.II/matrix_free/fe_remote_evaluation.h>
#include "fe_remote_evaluation.h"



// We pack everything that is specific for this program into a namespace
// of its own.
namespace Step89
{
  using namespace dealii;
  // TODO: dont declare it here
  using FERemoteEvaluationCommunicatorType =
    FERemoteEvaluationCommunicator<2, true, true>;
  using FERemoteEvaluationCommunicatorTypeMortar =
    FERemoteEvaluationCommunicator<2, true, false>;
  enum class CouplingType
  {
    P2P,
    Mortaring
  };
  CouplingType coupling_type = CouplingType::Mortaring;


  namespace NonMatchingHelpers
  {
    bool is_non_matching_face(
      const std::set<types::boundary_id> &non_matching_face_ids,
      const types::boundary_id            face_id)
    {
      return non_matching_face_ids.find(face_id) != non_matching_face_ids.end();
    }
  } // namespace NonMatchingHelpers

  template <int dim, typename Number>
  class AcousticConservationEquation
  {
    class HomogenousBCEvalP
    {
    public:
      HomogenousBCEvalP(
        const FEFaceEvaluation<dim, -1, 0, 1, Number> &pressure_m)
        : pressure_m(pressure_m)
      {}

      typename FEFaceEvaluation<dim, -1, 0, 1, Number>::value_type
      get_value(const unsigned int q) const
      {
        return pressure_m.get_value(q);
      }

    private:
      const FEFaceEvaluation<dim, -1, 0, 1, Number> &pressure_m;
    };

    class HomogenousBCEvalU
    {
    public:
      HomogenousBCEvalU(
        const FEFaceEvaluation<dim, -1, 0, dim, Number> &velocity_m)
        : velocity_m(velocity_m)
      {}

      typename FEFaceEvaluation<dim, -1, 0, dim, Number>::value_type
      get_value(const unsigned int q) const
      {
        return -velocity_m.get_value(q);
      }

    private:
      const FEFaceEvaluation<dim, -1, 0, dim, Number> &velocity_m;
    };


  public:
    AcousticConservationEquation(
      const FERemoteEvaluationCommunicatorType       &remote_comm,
      const FERemoteEvaluationCommunicatorTypeMortar &remote_comm_mortar,
      NonMatching::MappingInfo<dim, dim, Number>     &nm_info,
      const std::set<types::boundary_id>             &non_matching_face_ids,
      const double                                    density,
      const double                                    speed_of_sound)
      : remote_communicator(remote_comm)
      , remote_communicator_mortar(remote_comm_mortar)
      , nm_mapping_info(nm_info)
      , remote_face_ids(non_matching_face_ids)
      , rho(density)
      , c(speed_of_sound)
      , tau(0.5 * rho * c)
      , gamma(0.5 / (rho * c))
    {}

    template <typename VectorType>
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
    template <typename VectorType>
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

    template <typename VectorType>
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
              const auto n  = pressure_m.normal_vector(q);
              const auto pm = pressure_m.get_value(q);
              const auto um = velocity_m.get_value(q);
              const auto pp = pressure_p.get_value(q);
              const auto up = velocity_p.get_value(q);

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


    template <typename InternalFaceIntegratorPressure,
              typename InternalFaceIntegratorVelocity,
              typename ExternalFaceIntegratorPressure,
              typename ExternalFaceIntegratorVelocity>
    void
    perform_face_int(InternalFaceIntegratorPressure       &pressure_m,
                     InternalFaceIntegratorVelocity       &velocity_m,
                     const ExternalFaceIntegratorPressure &pressure_p,
                     const ExternalFaceIntegratorVelocity &velocity_p) const
    {
      for (unsigned int q : pressure_m.quadrature_point_indices())
        {
          const auto n  = pressure_m.normal_vector(q);
          const auto pm = pressure_m.get_value(q);
          const auto um = velocity_m.get_value(q);

          const auto pp = pressure_p.get_value(q);
          const auto up = velocity_p.get_value(q);

          const auto flux_momentum =
            0.5 * (pm + pp) + 0.5 * tau * (um - up) * n;
          velocity_m.submit_value(1.0 / rho * (flux_momentum - pm) * n, q);

          const auto flux_mass = 0.5 * (um + up) + 0.5 * gamma * (pm - pp) * n;
          pressure_m.submit_value(rho * c * c * (flux_mass - um) * n, q);
        }
    }

    template <typename VectorType>
    void boundary_face_loop(
      const MatrixFree<dim, Number>               &matrix_free,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      const auto &dh = matrix_free.get_dof_handler();
      const auto &fe = dh.get_fe();
      // remote evalutators
      FEFaceRemoteEvaluation<dim, 1, Number>   pressure_r(remote_communicator,
                                                        dh,
                                                        0);
      FEFaceRemoteEvaluation<dim, dim, Number> velocity_r(remote_communicator,
                                                          dh,
                                                          1);

      if (coupling_type == CouplingType::P2P)
        {
          pressure_r.gather_evaluate(src, EvaluationFlags::values);
          velocity_r.gather_evaluate(src, EvaluationFlags::values);
        }
      // standard face evaluators
      FEFaceEvaluation<dim, -1, 0, 1, Number> pressure_m(
        matrix_free, true, 0, 0, 0);
      FEFaceEvaluation<dim, -1, 0, dim, Number> velocity_m(
        matrix_free, true, 0, 0, 1);

      // classes which return the correct bc values
      HomogenousBCEvalP pressure_hbc(pressure_m);
      HomogenousBCEvalU velocity_hbc(velocity_m);

      // mortaring
      FEPointEvaluation<1, dim, dim, Number> pressure_m_mortar(nm_mapping_info,
                                                               fe,
                                                               0);
      FEPointEvaluation<dim, dim, dim, Number> velocity_m_mortar(
        nm_mapping_info, fe, 1);
      std::vector<Number> point_values(fe.dofs_per_cell);

      FEFaceRemotePointEvaluation<dim, 1, Number> pressure_r_mortar(
        remote_communicator_mortar, dh, 0);
      FEFaceRemotePointEvaluation<dim, dim, Number> velocity_r_mortar(
        remote_communicator_mortar, dh, 1);
      if (coupling_type == CouplingType::Mortaring)
        {
          pressure_r_mortar.gather_evaluate(src, EvaluationFlags::values);
          velocity_r_mortar.gather_evaluate(src, EvaluationFlags::values);
        }

      for (unsigned int face = face_range.first; face < face_range.second;
           face++)
        {
          if (NonMatchingHelpers::is_non_matching_face(
                remote_face_ids, matrix_free.get_boundary_id(face)))
            {
              if (coupling_type == CouplingType::Mortaring)
                {
                  for (unsigned int v = 0;
                       v < matrix_free.n_active_entries_per_face_batch(face);
                       ++v)
                    {
                      const auto [cell, f] =
                        matrix_free.get_face_iterator(face, v, true);

                      velocity_m_mortar.reinit(cell->active_cell_index(), f);
                      pressure_m_mortar.reinit(cell->active_cell_index(), f);

                      cell->get_dof_values(src,
                                           point_values.begin(),
                                           point_values.end());
                      velocity_m_mortar.evaluate(point_values,
                                                 EvaluationFlags::values);
                      pressure_m_mortar.evaluate(point_values,
                                                 EvaluationFlags::values);

                      velocity_r_mortar.reinit(cell->active_cell_index(), f);
                      pressure_r_mortar.reinit(cell->active_cell_index(), f);

                      perform_face_int(pressure_m_mortar,
                                       velocity_m_mortar,
                                       pressure_r_mortar,
                                       velocity_r_mortar);

                      // First zero out buffer via sum_into_values
                      velocity_m_mortar.integrate(point_values,
                                                  EvaluationFlags::values,
                                                  /*sum_into_values=*/false);
                      // Don't zero out values again to keep integrated values
                      pressure_m_mortar.integrate(point_values,
                                                  EvaluationFlags::values,
                                                  /*sum_into_values=*/true);

                      cell->distribute_local_to_global(point_values.begin(),
                                                       point_values.end(),
                                                       dst);
                    }
                }
              else
                {
                  velocity_m.reinit(face);
                  pressure_m.reinit(face);

                  pressure_m.gather_evaluate(src, EvaluationFlags::values);
                  velocity_m.gather_evaluate(src, EvaluationFlags::values);

                  velocity_r.reinit(face);
                  pressure_r.reinit(face);

                  perform_face_int(pressure_m,
                                   velocity_m,
                                   pressure_r,
                                   velocity_r);
                  pressure_m.integrate_scatter(EvaluationFlags::values, dst);
                  velocity_m.integrate_scatter(EvaluationFlags::values, dst);
                }
            }
          else
            {
              velocity_m.reinit(face);
              pressure_m.reinit(face);

              pressure_m.gather_evaluate(src, EvaluationFlags::values);
              velocity_m.gather_evaluate(src, EvaluationFlags::values);
              perform_face_int(pressure_m,
                               velocity_m,
                               pressure_hbc,
                               velocity_hbc);
              pressure_m.integrate_scatter(EvaluationFlags::values, dst);
              velocity_m.integrate_scatter(EvaluationFlags::values, dst);
            }
        }
    }


    const FERemoteEvaluationCommunicatorType       &remote_communicator;
    const FERemoteEvaluationCommunicatorTypeMortar &remote_communicator_mortar;
    NonMatching::MappingInfo<dim, dim, Number>     &nm_mapping_info;

    const std::set<types::boundary_id> remote_face_ids;
    const double                       rho;
    const double                       c;
    const double                       tau;
    const double                       gamma;
  };



  class InverseMassOperator
  {
  public:
    template <int dim, typename Number, typename VectorType>
    void apply(const MatrixFree<dim, Number> &matrix_free,
               VectorType                    &dst,
               const VectorType              &src) const
    {
      dst.zero_out_ghost_values();
      matrix_free.cell_loop(&InverseMassOperator::cell_loop, this, dst, src);
    }

  private:
    template <int dim, typename Number, typename VectorType>
    void
    cell_loop(const MatrixFree<dim, Number>               &matrix_free,
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

  template <int dim, typename Number>
  class SpatialOperator
  {
  public:
    SpatialOperator(
      const MatrixFree<dim, Number>                  &matrix_free_in,
      const FERemoteEvaluationCommunicatorType       &remote_comm,
      const FERemoteEvaluationCommunicatorTypeMortar &remote_comm_mortar,
      NonMatching::MappingInfo<dim, dim, Number>     &nm_info,
      const std::set<types::boundary_id>             &non_matching_face_ids,
      const double                                    density,
      const double                                    speed_of_sound)
      : matrix_free(matrix_free_in)
      , remote_communicator(remote_comm)
      , remote_communicator_mortar(remote_comm_mortar)
      , nm_mapping_info(nm_info)
      , remote_face_ids(non_matching_face_ids)
      , rho(Number{density})
      , c(Number{speed_of_sound})
    {}

    template <typename VectorType>
    void evaluate(VectorType &dst, const VectorType &src) const
    {
      AcousticConservationEquation<dim, Number>(remote_communicator,
                                                remote_communicator_mortar,
                                                nm_mapping_info,
                                                remote_face_ids,
                                                rho,
                                                c)
        .evaluate(matrix_free, dst, src);
      dst *= Number{-1.0};
      InverseMassOperator().apply(matrix_free, dst, dst);
    }

  private:
    const MatrixFree<dim, Number>                  &matrix_free;
    const FERemoteEvaluationCommunicatorType       &remote_communicator;
    const FERemoteEvaluationCommunicatorTypeMortar &remote_communicator_mortar;
    NonMatching::MappingInfo<dim, dim, Number>     &nm_mapping_info;

    const std::set<types::boundary_id> &remote_face_ids;
    const Number                        rho;
    const Number                        c;
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
    const double modes          = 10.0;
    const double length         = 0.1;

    const unsigned int subdiv_left  = 1;
    const unsigned int subdiv_right = 3;

    ConditionalOStream pcout(std::cout,
                             Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==
                               0);

    // store non-matching face pair
    std::pair<types::boundary_id, types::boundary_id> non_matching_face_pair =
      std::make_pair(98, 99);


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
            face->set_boundary_id(non_matching_face_pair.first);
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
            face->set_boundary_id(non_matching_face_pair.second);
        }


    GridGenerator::merge_triangulations(
      tria_left, tria_right, tria, 0., false, true);
    tria.refine_global(refinements);

    const MappingQ1<dim>  mapping;
    const FESystem<dim>   fe_dgq(FE_DGQ<dim>(degree), dim + 1);
    const unsigned int    n_quadrature_pnts = degree + 1;
    const QGauss<dim>     quad(n_quadrature_pnts);
    const QGauss<dim - 1> face_quad(degree + 1);

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


    std::set<types::boundary_id> non_matching_faces = {
      non_matching_face_pair.first, non_matching_face_pair.second};

    // TODO: Whats the problem with MPI
    // TODO: P2P Version
    FERemoteEvaluationCommunicatorType       remote_communicator;
    FERemoteEvaluationCommunicatorTypeMortar remote_communicator_mortar;
    typename NonMatching::MappingInfo<dim, dim, Number>::AdditionalData
      additional_data;
    additional_data.use_global_weights = true;
    NonMatching::MappingInfo<dim, dim, Number> nm_mapping_info(
      mapping,
      update_values | update_JxW_values | update_normal_vectors |
        update_quadrature_points,
      additional_data);

    if (coupling_type == CouplingType::P2P)
      {
        for (const auto &nm_face : non_matching_faces)
          {
            FEFaceValues<dim> phi(mapping,
                                  fe_dgq,
                                  face_quad,
                                  update_quadrature_points);

            std::vector<std::pair<typename Triangulation<dim>::cell_iterator,
                                  unsigned int>>
                                    cell_face_pairs;
            std::vector<Point<dim>> points;
            // get number of quadrature points per face
            std::vector<unsigned int> n_q_points;
            for (const auto &cell : tria.active_cell_iterators())
              {
                std::vector<Quadrature<dim>> face_quads(cell->n_faces());

                for (unsigned int f = 0; f < cell->n_faces(); ++f)
                  if (cell->face(f)->at_boundary() &&
                      cell->face(f)->boundary_id() == nm_face)
                    {
                      phi.reinit(cell, f);
                      cell_face_pairs.push_back(std::make_pair(cell, f));
                      n_q_points.push_back(phi.n_quadrature_points);
                      for (unsigned int q = 0; q < phi.n_quadrature_points; ++q)
                        {
                          points.push_back(phi.quadrature_point(q));
                        }
                    }
              }

            auto rpe =
              std::make_shared<Utilities::MPI::RemotePointEvaluation<dim>>(
                1.0e-9, false, 0, [&]() {
                  // only search points at cells that are not connected to
                  // nm_face
                  std::vector<bool> mask(tria.n_vertices(), false);

                  for (const auto &face : tria.active_face_iterators())
                    if (face->at_boundary() && face->boundary_id() != nm_face)
                      for (const auto v : face->vertex_indices())
                        mask[face->vertex_index(v)] = true;

                  return mask;
                });

            rpe->reinit(points, tria, mapping);
            Assert(rpe->all_points_found(),
                   ExcMessage("Not all remote points found."));


            remote_communicator.add_faces(matrix_free,
                                          rpe,
                                          cell_face_pairs,
                                          n_q_points);
          }
      }
    else if (coupling_type == CouplingType::Mortaring)
      {
        std::vector<std::vector<Quadrature<dim - 1>>> global_quadrature_vector;
        for (const auto &cell : tria.active_cell_iterators())
          global_quadrature_vector.emplace_back(
            std::vector<Quadrature<dim - 1>>(cell->n_faces()));

        std::vector<std::pair<
          std::shared_ptr<Utilities::MPI::RemotePointEvaluation<dim>>,
          std::vector<std::pair<typename Triangulation<dim>::cell_iterator,
                                unsigned int>>>>
          comm_objects;

        for (const auto &nm_face : non_matching_faces)
          {
            // 1) compute cell face pairs
            std::vector<std::pair<typename Triangulation<dim>::cell_iterator,
                                  unsigned int>>
              cell_face_pairs;

            for (const auto &cell : tria.active_cell_iterators())
              if (cell->is_locally_owned())
                for (unsigned int f = 0; f < cell->n_faces(); ++f)
                  if (cell->face(f)->at_boundary() &&
                      cell->face(f)->boundary_id() == nm_face)
                    cell_face_pairs.emplace_back(std::make_pair(cell, f));

            // 2) create RPE
            // create bounding boxes to search in
            std::vector<BoundingBox<dim>> local_boxes;
            for (const auto &cell : tria.active_cell_iterators())
              if (cell->is_locally_owned())
                local_boxes.emplace_back(mapping.get_bounding_box(cell));

            // create r-tree of bounding boxes
            const auto local_tree = pack_rtree(local_boxes);

            // compress r-tree to a minimal set of bounding boxes
            std::vector<std::vector<BoundingBox<dim>>> global_bboxes(1);
            global_bboxes[0] = extract_rtree_level(local_tree, 0);

            const GridTools::Cache<dim, dim> cache(tria, mapping);

            // build intersection requests. Intersection requests
            // correspond to vertices at faces.
            std::vector<std::vector<Point<dim>>> intersection_requests;
            for (const auto &[cell, f] : cell_face_pairs)
              {
                std::vector<Point<dim>> vertices(cell->face(f)->n_vertices());
                std::copy_n(mapping.get_vertices(cell, f).begin(),
                            cell->face(f)->n_vertices(),
                            vertices.begin());
                intersection_requests.emplace_back(vertices);
              }

            // compute intersection data
            auto intersection_data =
              GridTools::internal::distributed_compute_intersection_locations<
                dim - 1>(
                cache,
                intersection_requests,
                global_bboxes,
                [&]() {
                  std::vector<bool> mask(tria.n_vertices(), false);

                  for (const auto &face : tria.active_face_iterators())
                    if (face->at_boundary() && face->boundary_id() != nm_face)
                      for (const auto v : face->vertex_indices())
                        mask[face->vertex_index(v)] = true;

                  return mask;
                }(),
                1.0e-9);

            // convert to rpe
            auto rpe =
              std::make_shared<Utilities::MPI::RemotePointEvaluation<dim>>();
            rpe->reinit(
              intersection_data
                .template convert_to_distributed_compute_point_locations_internal<
                  dim>(n_quadrature_pnts, tria, mapping),
              tria,
              mapping);

            // TODO: Most of the following is currently done twice in
            //  convert_to_distributed_compute_point_locations_internal.
            //  We have to adapt
            //  convert_to_distributed_compute_point_locations_internal to be
            //  able to retrieve relevant information.

            // TODO: NonMatchingMappingInfo should be able to work with
            //  Quadrature<dim> instead <dim-1>. Currently we are constructing
            //  dim-1 from dim and inside MappingInfo it is converted back.

            // 3) fill quadrature vector.
            for (unsigned int i = 0; i < intersection_requests.size(); ++i)
              {
                const auto &[cell, f] = cell_face_pairs[i];

                const unsigned int begin = intersection_data.recv_ptrs[i];
                const unsigned int end   = intersection_data.recv_ptrs[i + 1];

                std::vector<typename GridTools::internal::
                              DistributedComputeIntersectionLocationsInternal<
                                dim - 1,
                                dim>::IntersectionType>
                  found_intersections(end - begin);

                unsigned int c = 0;
                for (unsigned int ptr = begin; ptr < end; ++ptr, ++c)
                  found_intersections[c] =
                    std::get<2>(intersection_data.recv_components[ptr]);

                const auto quad = QGaussSimplex<dim - 1>(n_quadrature_pnts)
                                    .mapped_quadrature(found_intersections);

                std::vector<Point<dim - 1>> face_points(quad.size());
                for (uint q = 0; q < quad.size(); ++q)
                  {
                    face_points[q] =
                      mapping.project_real_point_to_unit_point_on_face(
                        cell, f, quad.point(q));
                  }

                Assert(global_quadrature_vector[cell->active_cell_index()][f]
                           .size() == 0,
                       ExcMessage(
                         "Quadrature for given face already provided."));

                global_quadrature_vector[cell->active_cell_index()][f] =
                  Quadrature<dim - 1>(face_points, quad.get_weights());
              }
            comm_objects.push_back(std::make_pair(rpe, cell_face_pairs));
          }

        remote_communicator_mortar.reinit_faces(comm_objects,
                                                matrix_free.get_dof_handler()
                                                  .get_triangulation()
                                                  .active_cell_iterators(),
                                                global_quadrature_vector);

        nm_mapping_info.reinit_faces(matrix_free.get_dof_handler()
                                       .get_triangulation()
                                       .active_cell_iterators(),
                                     global_quadrature_vector);
      }
    else
      AssertThrow(false, ExcMessage("CouplingType not implemented"));

    pcout << "setup..." << std::endl;
    SpatialOperator<dim, Number> acoustic_operator(matrix_free,
                                                   remote_communicator,
                                                   remote_communicator_mortar,
                                                   nm_mapping_info,
                                                   non_matching_faces,
                                                   density,
                                                   speed_of_sound);

    const double end_time = 2.0 / (modes * std::sqrt(dim) * speed_of_sound);
    double       time     = 0.0;
    unsigned int timestep = 0;

    while (time < end_time)
      {
        pcout << "time" << time << std::endl;
        std::swap(solution, solution_temp);
        time += dt;
        timestep++;
        RungeKutta2::perform_time_step(acoustic_operator,
                                       dt,
                                       solution,
                                       solution_temp);

        // if (timestep % 1000 == 0)
        {
          DataOut<dim>          data_out;
          DataOutBase::VtkFlags flags;
          flags.write_higher_order_cells = true;
          data_out.set_flags(flags);

          std::vector<DataComponentInterpretation::DataComponentInterpretation>
            interpretation(
              dim + 1,
              DataComponentInterpretation::component_is_part_of_vector);
          std::vector<std::string> names(dim + 1, "U");

          interpretation[0] = DataComponentInterpretation::component_is_scalar;
          names[0]          = "P";

          data_out.add_data_vector(dof_handler,
                                   solution,
                                   names,
                                   interpretation);

          data_out.build_patches(mapping,
                                 degree,
                                 DataOut<dim>::curved_inner_cells);
          data_out.write_vtu_in_parallel("Aexample_1" +
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

  Step89::point_to_point_interpolation(2, 1);
  // Step89::nitsche_type_mortaring();
  // Step89::inhomogenous_material();

  return 0;
}
