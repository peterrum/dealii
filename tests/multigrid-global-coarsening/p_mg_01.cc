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


/**
 * TODO.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_precondition.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/simplex/fe_lib.h>
#include <deal.II/simplex/grid_generator.h>

#include <vector>

#include "test.h"

using namespace dealii;

namespace MGTransferGlobalCoarseningTools
{
  enum class PolynomialSequenceType
  {
    bisect,
    decrease_by_one,
    go_to_one
  };

  unsigned int
  generate_level_degree(const unsigned int            previous_fe_degree,
                        const PolynomialSequenceType &p_sequence)
  {
    switch (p_sequence)
      {
        case PolynomialSequenceType::bisect:
          return std::max(previous_fe_degree / 2, 1u);
        case PolynomialSequenceType::decrease_by_one:
          return std::max(previous_fe_degree - 1, 1u);
        case PolynomialSequenceType::go_to_one:
          return 1u;
        default:
          Assert(false, StandardExceptions::ExcNotImplemented());
          return 1u;
      }
  }

  std::vector<unsigned int>
  create_p_sequence(const unsigned int            degree,
                    const PolynomialSequenceType &p_sequence)
  {
    std::vector<unsigned int> degrees;
    degrees.push_back(degree);

    unsigned int previous_fe_degree = degree;
    while (previous_fe_degree > 1)
      {
        const unsigned int level_degree =
          generate_level_degree(previous_fe_degree, p_sequence);

        degrees.push_back(level_degree);
        previous_fe_degree = level_degree;
      }

    std::reverse(degrees.begin(), degrees.end());

    return degrees;
  }

} // namespace MGTransferGlobalCoarseningTools

template <int dim_, typename Number>
class Operator : public Subscriptor
{
public:
  using value_type = Number;
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  static const int dim = dim_;

  virtual types::global_dof_index
  m() const;

  Number
  el(unsigned int, unsigned int) const;

  virtual void
  initialize_dof_vector(VectorType &vec) const;

  virtual void
  vmult(VectorType &dst, const VectorType &src) const;

  void
  Tvmult(VectorType &dst, const VectorType &src) const;

  void
  compute_inverse_diagonal(VectorType &diagonal) const;

  virtual const TrilinosWrappers::SparseMatrix &
  get_system_matrix() const;

private:
};

struct GMGParameters
{
  struct CoarseSolverParameters
  {
    std::string  type            = "cg_with_amg"; // "cg";
    unsigned int maxiter         = 10000;
    double       abstol          = 1e-20;
    double       reltol          = 1e-4;
    unsigned int smoother_sweeps = 1;
    unsigned int n_cycles        = 1;
    std::string  smoother_type   = "ILU";
  };

  struct SmootherParameters
  {
    std::string  type                = "chebyshev";
    double       smoothing_range     = 20;
    unsigned int degree              = 5;
    unsigned int eig_cg_n_iterations = 20;
  };

  SmootherParameters     smoother;
  CoarseSolverParameters coarse_solver;

  unsigned int maxiter = 10000;
  double       abstol  = 1e-20;
  double       reltol  = 1e-4;
};

template <typename VectorType,
          int dim,
          typename SystemMatrixType,
          typename LevelMatrixType,
          typename MGTransferType>
static void
mg_solve(SolverControl &                       solver_control,
         VectorType &                          dst,
         const VectorType &                    src,
         const GMGParameters &                 mg_data,
         const DoFHandler<dim> &               dof,
         const SystemMatrixType &              fine_matrix,
         const MGLevelObject<LevelMatrixType> &mg_matrices,
         const MGTransferType &                mg_transfer)
{
  AssertThrow(mg_data.smoother.type == "chebyshev", ExcNotImplemented());

  const unsigned int min_level = mg_matrices.min_level();
  const unsigned int max_level = mg_matrices.max_level();

  using Number                     = typename VectorType::value_type;
  using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
  using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                             VectorType,
                                             SmootherPreconditionerType>;
  using PreconditionerType = PreconditionMG<dim, VectorType, MGTransferType>;

  // Initialize level operators.
  mg::Matrix<VectorType> mg_matrix(mg_matrices);

  // Initialize smoothers.
  MGLevelObject<typename SmootherType::AdditionalData> smoother_data(min_level,
                                                                     max_level);

  for (unsigned int level = min_level; level <= max_level; level++)
    {
      smoother_data[level].preconditioner =
        std::make_shared<SmootherPreconditionerType>();
      mg_matrices[level].compute_inverse_diagonal(
        smoother_data[level].preconditioner->get_vector());
      smoother_data[level].smoothing_range = mg_data.smoother.smoothing_range;
      smoother_data[level].degree          = mg_data.smoother.degree;
      smoother_data[level].eig_cg_n_iterations =
        mg_data.smoother.eig_cg_n_iterations;
    }

  MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType> mg_smoother;
  mg_smoother.initialize(mg_matrices, smoother_data);

  // Initialize coarse-grid solver.
  ReductionControl     coarse_grid_solver_control(mg_data.coarse_solver.maxiter,
                                              mg_data.coarse_solver.abstol,
                                              mg_data.coarse_solver.reltol,
                                              false,
                                              false);
  SolverCG<VectorType> coarse_grid_solver(coarse_grid_solver_control);

  PreconditionIdentity precondition_identity;
  PreconditionChebyshev<LevelMatrixType, VectorType, DiagonalMatrix<VectorType>>
    precondition_chebyshev;

#ifdef DEAL_II_WITH_TRILINOS
  TrilinosWrappers::PreconditionAMG precondition_amg;
#endif

  std::unique_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

  if (mg_data.coarse_solver.type == "cg")
    {
      // CG with identity matrix as preconditioner

      mg_coarse =
        std::make_unique<MGCoarseGridIterativeSolver<VectorType,
                                                     SolverCG<VectorType>,
                                                     LevelMatrixType,
                                                     PreconditionIdentity>>(
          coarse_grid_solver, mg_matrices[min_level], precondition_identity);
    }
  else if (mg_data.coarse_solver.type == "cg_with_chebyshev")
    {
      // CG with Chebyshev as preconditioner

      typename SmootherType::AdditionalData smoother_data;

      smoother_data.preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>();
      mg_matrices[min_level].compute_inverse_diagonal(
        smoother_data.preconditioner->get_vector());
      smoother_data.smoothing_range     = mg_data.smoother.smoothing_range;
      smoother_data.degree              = mg_data.smoother.degree;
      smoother_data.eig_cg_n_iterations = mg_data.smoother.eig_cg_n_iterations;

      precondition_chebyshev.initialize(mg_matrices[min_level], smoother_data);

      mg_coarse = std::make_unique<
        MGCoarseGridIterativeSolver<VectorType,
                                    SolverCG<VectorType>,
                                    LevelMatrixType,
                                    decltype(precondition_chebyshev)>>(
        coarse_grid_solver, mg_matrices[min_level], precondition_chebyshev);
    }
  else if (mg_data.coarse_solver.type == "cg_with_amg")
    {
      // CG with AMG as preconditioner

#ifdef DEAL_II_WITH_TRILINOS
      TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
      amg_data.smoother_sweeps = mg_data.coarse_solver.smoother_sweeps;
      amg_data.n_cycles        = mg_data.coarse_solver.n_cycles;
      amg_data.smoother_type   = mg_data.coarse_solver.smoother_type.c_str();

      // CG with AMG as preconditioner
      precondition_amg.initialize(mg_matrices[min_level].get_system_matrix(),
                                  amg_data);

      mg_coarse = std::make_unique<
        MGCoarseGridIterativeSolver<VectorType,
                                    SolverCG<VectorType>,
                                    LevelMatrixType,
                                    decltype(precondition_amg)>>(
        coarse_grid_solver, mg_matrices[min_level], precondition_amg);
#else
      AssertThrow(false, ExcNotImplemented());
#endif
    }
  else
    {
      AssertThrow(false, ExcNotImplemented());
    }

  // Create multigrid object.
  Multigrid<VectorType> mg(
    mg_matrix, *mg_coarse, mg_transfer, mg_smoother, mg_smoother);

  // Convert it to a preconditioner.
  PreconditionerType preconditioner(dof, mg, mg_transfer);

  // Finally, solve.
  SolverCG<VectorType>(solver_control)
    .solve(fine_matrix, dst, src, preconditioner);
}

template <int dim, typename Number = double>
void
test(int fe_degree_fine)
{
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, 2);
  tria.refine_global(2);

  const auto level_degrees = MGTransferGlobalCoarseningTools::create_p_sequence(
    fe_degree_fine,
    MGTransferGlobalCoarseningTools::PolynomialSequenceType::bisect);

  const unsigned int min_level = 0;
  const unsigned int max_level = level_degrees.size() - 1;

  MGLevelObject<DoFHandler<dim>> dof_handlers(min_level, max_level, tria);
  MGLevelObject<AffineConstraints<Number>> constraints(min_level, max_level);
  MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers(min_level,
                                                               max_level);
  MGLevelObject<Operator<dim, Number>> operators(min_level, max_level);

  // set up levels
  for (auto l = min_level; l <= max_level; ++l)
    {
      auto &dof_handler = dof_handlers[l];
      auto &constraint  = constraints[l];
      auto &op          = operators[l];

      const FE_Q<dim> fe(level_degrees[l]);

      // set up dofhandler
      dof_handler.distribute_dofs(fe);

      // set up constraints
      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      constraint.reinit(locally_relevant_dofs);
      DoFTools::make_hanging_node_constraints(dof_handler, constraint);
      constraint.close();

      // set up operator
    }

  // set up transfer operator
  for (unsigned int l = min_level; l < max_level; ++l)
    transfers[l + 1].reinit_polynomial_transfer(dof_handlers[l + 1],
                                                dof_handlers[l],
                                                constraints[l + 1],
                                                constraints[l]);

  MGTransferGlobalCoarsening<Operator<dim, Number>, VectorType> transfer(
    operators, transfers);

  GMGParameters mg_data;  // TODO
  VectorType    dst, src; // TODO

  ReductionControl solver_control(
    mg_data.maxiter, mg_data.abstol, mg_data.reltol, false, false);

  mg_solve(solver_control,
           dst,
           src,
           mg_data,
           dof_handlers[max_level],
           operators[max_level],
           operators,
           transfer);
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  deallog.precision(8);

  test<2>(2);
}
