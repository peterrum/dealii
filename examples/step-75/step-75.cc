/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2020 by the deal.II authors
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
 * Author: Marc Fehling, Colorado State University, 2020
 *         Peter Munch, Technical University of Munich and Helmholtz-Zentrum
 *                      Geesthacht, 2020
 *         Wolfgang Bangerth, Colorado State University, 2020
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

// uncomment the following \#define if you have PETSc and Trilinos installed
// and you prefer using Trilinos in this example:
// @code
// #define FORCE_USE_OF_TRILINOS
// @endcode

// This will either import PETSc or TrilinosWrappers into the namespace
// LA. Note that we are defining the macro USE_PETSC_LA so that we can detect
// if we are using PETSc (see solve() for an example where this is necessary)
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>


#include <deal.II/distributed/error_predictor.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/refinement.h>

#include <deal.II/fe/fe_series.h>

#include <deal.II/numerics/smoothness_estimator.h>

#include <deal.II/base/geometric_utilities.h>
#include <deal.II/base/mpi.h>


#include <fstream>
#include <memory>
#include <iostream>

// TODO
// includes need cleanup

namespace Step75
{
  using namespace dealii;

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution(const double alpha = 2. / 3.)
      : Function<dim>()
      , alpha(alpha)
    {
      Assert(dim > 1, ExcNotImplemented());
      Assert(alpha > 0, ExcLowerRange(alpha, 0));
    }

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
             const unsigned int component = 0) const override;

    virtual double laplacian(const Point<dim> & p,
                             const unsigned int component = 0) const override;

  private:
    const double alpha;
  };



  template <int dim>
  double Solution<dim>::value(const Point<dim> &p,
                              const unsigned int /*component*/) const
  {
    const std::array<double, dim> p_sphere =
      GeometricUtilities::Coordinates::to_spherical(p);

    return std::pow(p_sphere[0], alpha) * std::sin(alpha * p_sphere[1]);
  }



  template <int dim>
  Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p,
                                         const unsigned int /*component*/) const
  {
    const std::array<double, dim> p_sphere =
      GeometricUtilities::Coordinates::to_spherical(p);

    std::array<double, dim> ret_sphere;
    // only for polar coordinates
    const double fac = alpha * std::pow(p_sphere[0], alpha - 1);
    ret_sphere[0]    = fac * std::sin(alpha * p_sphere[1]);
    ret_sphere[1]    = fac * std::cos(alpha * p_sphere[1]);

    // transform back to cartesian coordinates
    // by considering polar unit vectors
    Tensor<1, dim> ret;
    ret[0] = ret_sphere[0] * std::cos(p_sphere[1]) -
             ret_sphere[1] * std::sin(p_sphere[1]);
    ret[1] = ret_sphere[0] * std::sin(p_sphere[1]) +
             ret_sphere[1] * std::cos(p_sphere[1]);
    return ret;
  }



  template <int dim>
  double Solution<dim>::laplacian(const Point<dim> & /*p*/,
                                  const unsigned int /*component*/) const
  {
    return 0.;
  }



  enum AdaptationType
  {
    h,
    hpLegendre,
    hpFourier,
    hpHistory
  };



  enum PreconditionerType
  {
    AMG,
    pAMG,
    pGMG
  };



  enum SolverType
  {
    Matrix,
    MatrixFree,
    MatrixFreeCUDA
  };



  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem(AdaptationType     adaptation_type,
                   PreconditionerType preconditioner_type,
                   SolverType         solver_type);

    void run(const unsigned int n_cycles);

  private:
    void create_coarse_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void compute_errors();
    void flag_adaptation();
    void decide_hp();
    void limit_levels();
    void output_results(const unsigned int cycle) const;

    MPI_Comm mpi_communicator;

    const AdaptationType     adaptation_type;
    const PreconditionerType preconditioner_type;
    const SolverType         solver_type;

    parallel::distributed::Triangulation<dim> triangulation;
    const unsigned int                        min_level, max_level;

    DoFHandler<dim>          dof_handler;
    hp::FECollection<dim>    fe_collection;
    hp::QCollection<dim>     quadrature_collection;
    hp::QCollection<dim - 1> face_quadrature_collection;
    const unsigned int       min_degree, max_degree;

    std::unique_ptr<hp::FEValues<dim>>       fe_values_collection;
    std::unique_ptr<FESeries::Legendre<dim>> legendre;
    std::unique_ptr<FESeries::Fourier<dim>>  fourier;

    Vector<float>                              predicted_error_per_cell;
    parallel::distributed::ErrorPredictor<dim> error_predictor;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LA::MPI::SparseMatrix system_matrix;
    LA::MPI::Vector       locally_relevant_solution;
    LA::MPI::Vector       system_rhs;

    Vector<float> estimated_error_per_cell, hp_decision_indicators;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };



  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem(AdaptationType     adaptation_type,
                                      PreconditionerType preconditioner_type,
                                      SolverType         solver_type)
    : mpi_communicator(MPI_COMM_WORLD)
    , adaptation_type(adaptation_type)
    , preconditioner_type(preconditioner_type)
    , solver_type(solver_type)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , min_level(5)
    , max_level(dim <= 2 ? 10 : 8)
    , dof_handler(triangulation)
    , min_degree(2)
    , max_degree(dim <= 2 ? 7 : 5)
    , error_predictor(dof_handler)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {
    Assert(preconditioner_type == PreconditionerType::AMG, ExcNotImplemented());
    Assert(solver_type == SolverType::Matrix, ExcNotImplemented());

    TimerOutput::Scope t(computing_timer, "init");

    Assert(min_level <= max_level,
           ExcMessage(
             "Triangulation level limits have been incorrectly set up."));
    Assert(min_degree <= max_degree,
           ExcMessage("FECollection degrees have been incorrectly set up."));

    for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
      {
        fe_collection.push_back(FE_Q<dim>(degree));
        quadrature_collection.push_back(QGauss<dim>(degree + 1));
        face_quadrature_collection.push_back(QGauss<dim - 1>(degree + 1));
      }

    fe_values_collection =
      std::make_unique<hp::FEValues<dim>>(fe_collection,
                                          quadrature_collection,
                                          update_gradients |
                                            update_quadrature_points |
                                            update_JxW_values);
    fe_values_collection->precalculate_fe_values();

    switch (adaptation_type)
      {
        case AdaptationType::hpLegendre:
          legendre = std::make_unique<FESeries::Legendre<dim>>(
            SmoothnessEstimator::Legendre::default_fe_series(fe_collection));
          legendre->precalculate_all_transformation_matrices();
          break;

        case AdaptationType::hpFourier:
          fourier = std::make_unique<FESeries::Fourier<dim>>(
            SmoothnessEstimator::Fourier::default_fe_series(fe_collection));
          fourier->precalculate_all_transformation_matrices();
          break;

        default:
          break;
      }
  }



  template <int dim>
  void LaplaceProblem<dim>::create_coarse_grid()
  {
    TimerOutput::Scope t(computing_timer, "coarse grid");

    std::vector<unsigned int> repetitions(dim, 2);
    Point<dim>                bottom_left, top_right;
    for (unsigned int d = 0; d < dim; ++d)
      {
        bottom_left[d] = -1.;
        top_right[d]   = 1.;
      }

    std::vector<int> cells_to_remove(dim, 1);
    cells_to_remove[0] = -1;

    // TODO
    // expand domain by 1 cell in z direction for 3d case

    GridGenerator::subdivided_hyper_L(
      triangulation, repetitions, bottom_left, top_right, cells_to_remove);
  }



  template <int dim>
  void LaplaceProblem<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe_collection);

    locally_owned_dofs = dof_handler.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    locally_relevant_solution.reinit(locally_owned_dofs,
                                     locally_relevant_dofs,
                                     mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Solution<dim>(),
                                             constraints);

#ifdef DEBUG
    // We have not dealt with chains of constraints on ghost cells yet.
    // Thus, we are content with verifying their consistency for now.
    IndexSet locally_active_dofs;
    DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);
    AssertThrow(constraints.is_consistent_in_parallel(
                  Utilities::MPI::all_gather(mpi_communicator,
                                             dof_handler.locally_owned_dofs()),
                  locally_active_dofs,
                  mpi_communicator,
                  /*verbose=*/true),
                ExcMessage(
                  "AffineConstraints object contains inconsistencies!"));
#endif
    constraints.close();

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               dof_handler.locally_owned_dofs(),
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }



  template <int dim>
  void LaplaceProblem<dim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assemble");

    FullMatrix<double> cell_matrix;
    Vector<double>     cell_rhs;

    std::vector<types::global_dof_index> local_dof_indices;

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;

          cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_matrix = 0;

          cell_rhs.reinit(dofs_per_cell);
          cell_rhs = 0;

          fe_values_collection->reinit(cell);

          const FEValues<dim> &fe_values =
            fe_values_collection->get_present_fe_values();

          for (unsigned int q_point = 0;
               q_point < fe_values.n_quadrature_points;
               ++q_point)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q_point) * // grad phi_i(x_q)
                     fe_values.shape_grad(j, q_point) * // grad phi_j(x_q)
                     fe_values.JxW(q_point));           // dx
              }

          local_dof_indices.resize(dofs_per_cell);
          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }



  template <int dim>
  void LaplaceProblem<dim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    SolverControl solver_control(system_rhs.size(),
                                 1e-12 * system_rhs.l2_norm());
#ifdef USE_PETSC_LA
    LA::SolverCG cg(solver_control, mpi_communicator);
#else
    LA::SolverCG cg(solver_control);
#endif

    LA::MPI::PreconditionAMG                 preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
    data.elliptic              = true;
    data.higher_order_elements = true;
#endif
    preconditioner.initialize(system_matrix, data);

    cg.solve(system_matrix,
             completely_distributed_solution,
             system_rhs,
             preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(completely_distributed_solution);

    locally_relevant_solution = completely_distributed_solution;
  }



  template <int dim>
  void LaplaceProblem<dim>::compute_errors()
  {
    TimerOutput::Scope t(computing_timer, "compute errors");

    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      locally_relevant_solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      quadrature_collection,
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      locally_relevant_solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      quadrature_collection,
                                      VectorTools::H1_norm);
    const double H1_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_norm);

    pcout << "L2 error: " << L2_error << std::endl
          << "H1 error: " << H1_error << std::endl;

    // TODO
    // Store errors in Convergence table
  }



  template <int dim>
  void LaplaceProblem<dim>::flag_adaptation()
  {
    TimerOutput::Scope t(computing_timer, "flag adaptation");

    estimated_error_per_cell.grow_or_shrink(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      face_quadrature_collection,
      std::map<types::boundary_id, const Function<dim> *>(),
      locally_relevant_solution,
      estimated_error_per_cell,
      /*component_mask=*/ComponentMask(),
      /*coefficients=*/nullptr,
      /*n_threads=*/numbers::invalid_unsigned_int,
      /*subdomain_id=*/numbers::invalid_subdomain_id,
      /*material_id=*/numbers::invalid_material_id,
      /*strategy=*/
      KellyErrorEstimator<dim>::Strategy::face_diameter_over_twice_max_degree);

    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, estimated_error_per_cell, 0.3, 0.03);
  }



  template <int dim>
  void LaplaceProblem<dim>::decide_hp()
  {
    TimerOutput::Scope t(computing_timer, "decide hp");

    hp_decision_indicators.grow_or_shrink(triangulation.n_active_cells());

    switch (adaptation_type)
      {
        case AdaptationType::hpLegendre:
          SmoothnessEstimator::Legendre::coefficient_decay(
            *legendre,
            dof_handler,
            locally_relevant_solution,
            hp_decision_indicators,
            /*regression_strategy=*/VectorTools::Linfty_norm,
            /*smallest_abs_coefficient=*/1e-10,
            /*only_flagged_cells=*/true);
          break;

        case AdaptationType::hpFourier:
          SmoothnessEstimator::Fourier::coefficient_decay(
            *fourier,
            dof_handler,
            locally_relevant_solution,
            hp_decision_indicators,
            /*regression_strategy=*/VectorTools::Linfty_norm,
            /*smallest_abs_coefficient=*/1e-10,
            /*only_flagged_cells=*/true);
          break;

        case AdaptationType::hpHistory:
          {
            for (unsigned int i = 0; i < triangulation.n_active_cells(); ++i)
              hp_decision_indicators(i) =
                predicted_error_per_cell(i) - estimated_error_per_cell(i);

            const float global_minimum = Utilities::MPI::min(
              *std::min_element(hp_decision_indicators.begin(),
                                hp_decision_indicators.end()),
              mpi_communicator);
            if (global_minimum < 0)
              for (auto &indicator : hp_decision_indicators)
                indicator -= global_minimum;
          }
          break;

        default:
          Assert(false, ExcNotImplemented());
          break;
      }

    hp::Refinement::p_adaptivity_fixed_number(dof_handler,
                                              hp_decision_indicators,
                                              0.9,
                                              0.9);
    hp::Refinement::choose_p_over_h(dof_handler);
  }



  template <int dim>
  void LaplaceProblem<dim>::limit_levels()
  {
    Assert(triangulation.n_levels() >= min_level + 1 &&
             triangulation.n_levels() <= max_level + 1,
           ExcInternalError());

    if (triangulation.n_levels() > max_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_level))
        cell->clear_refine_flag();

    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_level))
      cell->clear_coarsen_flag();
  }



  template <int dim>
  void LaplaceProblem<dim>::output_results(const unsigned int cycle) const
  {
    Vector<float> fe_degrees(triangulation.n_active_cells());
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        fe_degrees(cell->active_cell_index()) =
          fe_collection[cell->active_fe_index()].degree;

    Vector<float> subdomain(triangulation.n_active_cells());
    for (auto &subd : subdomain)
      subd = triangulation.locally_owned_subdomain();

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(locally_relevant_solution, "solution");
    data_out.add_data_vector(fe_degrees, "fe_degree");
    data_out.add_data_vector(subdomain, "subdomain");
    // data_out.add_data_vector(estimated_error_per_cell, "error");
    // data_out.add_data_vector(hp_decision_indicators, "hp_indicator");
    data_out.build_patches();

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2, 8);
  }



  template <int dim>
  void LaplaceProblem<dim>::run(const unsigned int n_cycles)
  {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    for (int cycle = adaptation_type != AdaptationType::hpHistory ? 0 : -1;
         cycle < (int)n_cycles;
         ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (adaptation_type != AdaptationType::hpHistory)
          {
            if (cycle == 0)
              {
                create_coarse_grid();
                triangulation.refine_global(min_level);
              }
            else
              {
                flag_adaptation();
                if (adaptation_type != AdaptationType::h)
                  decide_hp();
                limit_levels();

                triangulation.execute_coarsening_and_refinement();
              }
          }
        else
          {
            if (cycle == -1)
              {
                create_coarse_grid();
                triangulation.refine_global(min_level - 1);
              }
            else
              {
                if (cycle == 0)
                  {
                    estimated_error_per_cell.grow_or_shrink(
                      triangulation.n_active_cells());

                    KellyErrorEstimator<dim>::estimate(
                      dof_handler,
                      face_quadrature_collection,
                      std::map<types::boundary_id, const Function<dim> *>(),
                      locally_relevant_solution,
                      estimated_error_per_cell,
                      /*component_mask=*/ComponentMask(),
                      /*coefficients=*/nullptr,
                      /*n_threads=*/numbers::invalid_unsigned_int,
                      /*subdomain_id=*/numbers::invalid_subdomain_id,
                      /*material_id=*/numbers::invalid_material_id,
                      /*strategy=*/
                      KellyErrorEstimator<
                        dim>::Strategy::face_diameter_over_twice_max_degree);

                    for (const auto &cell :
                         triangulation.active_cell_iterators())
                      if (cell->is_locally_owned())
                        cell->set_refine_flag();
                  }
                else
                  {
                    flag_adaptation();
                    decide_hp();
                    limit_levels();
                  }

                error_predictor.prepare_for_coarsening_and_refinement(
                  estimated_error_per_cell,
                  /*gamma_p=*/std::sqrt(0.4),
                  /*gamma_h=*/2.,
                  /*gamma_n=*/1.);

                triangulation.execute_coarsening_and_refinement();

                predicted_error_per_cell.grow_or_shrink(
                  triangulation.n_active_cells());
                error_predictor.unpack(predicted_error_per_cell);
              }
          }

        setup_system();

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        assemble_system();
        solve();
        compute_errors();

        if (cycle >= 0 &&
            Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(cycle);
          }

        computing_timer.print_summary();
        computing_timer.reset();

        pcout << std::endl;
      }
  }
} // namespace Step75



int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step75;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      LaplaceProblem<2> laplace_problem_2d(AdaptationType::hpLegendre,
                                           PreconditionerType::AMG,
                                           SolverType::Matrix);
      laplace_problem_2d.run(/*n_cycles=*/8);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
