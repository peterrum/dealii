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

#ifndef dealii_mg_solver_h
#define dealii_mg_solver_h


#include <deal.II/base/config.h>

#include <deal.II/base/mg_level_object.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <vector>

DEAL_II_NAMESPACE_OPEN



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

  MGTransferGlobalCoarseningTools::PolynomialSequenceType p_sequence =
    MGTransferGlobalCoarseningTools::PolynomialSequenceType::
      decrease_by_one;            // TODO: move
  bool perform_h_transfer = true; // TODO: move
};



template <int dim_, typename number>
class LaplaceOperator : public Subscriptor
{
public:
  static const int dim = dim_;
  using value_type     = number;
  using VectorType     = LinearAlgebra::distributed::Vector<number>;

  // Return number of rows of the matrix. Since we are dealing with a
  // symmetrical matrix, the returned value is the same as the number of
  // columns.
  virtual types::global_dof_index
  m() const = 0;

  // Access a particular element in the matrix. This function is neither
  // needed nor implemented, however, is required to compile the program.
  virtual number
  el(unsigned int, unsigned int) const;

  // Allocate memory for a distributed vector.
  virtual void
  initialize_dof_vector(VectorType &vec) const = 0;

  // Perform an operator application on the vector @p src.
  virtual void
  vmult(VectorType &dst, const VectorType &src) const = 0;

  // Perform the transposed operator evaluation. Since we are considering
  // symmetric matrices, this function is identical to the above function.
  void
  Tvmult(VectorType &dst, const VectorType &src) const;

  // Compute the inverse of the diagonal of the vector and store it into the
  // provided vector. The inverse diagonal is used below in a Chebyshev
  // smoother.
  virtual void
  compute_inverse_diagonal(VectorType &diagonal) const = 0;

  // Return the actual system matrix, which can be used in any matrix-based
  // solvers (like AMG).
  virtual const TrilinosWrappers::SparseMatrix &
  get_system_matrix() const = 0;
};



template <int dim_, typename number>
void
LaplaceOperator<dim_, number>::Tvmult(VectorType &      dst,
                                      const VectorType &src) const
{
  this->vmult(dst, src);
}



template <int dim_, typename number>
number
LaplaceOperator<dim_, number>::el(unsigned int, unsigned int) const
{
  Assert(false, ExcNotImplemented());
  return 0;
}

template <typename VectorType,
          int dim,
          typename SystemMatrixType,
          typename LevelMatrixType,
          typename MGTransferType>
static void
mg_solve(SolverControl &                                        solver_control,
         VectorType &                                           dst,
         const VectorType &                                     src,
         const GMGParameters &                                  mg_data,
         const DoFHandler<dim> &                                dof,
         const SystemMatrixType &                               fine_matrix,
         const MGLevelObject<std::unique_ptr<LevelMatrixType>> &mg_matrices,
         const MGTransferType &                                 mg_transfer)
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
      mg_matrices[level]->compute_inverse_diagonal(
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
          coarse_grid_solver, *mg_matrices[min_level], precondition_identity);
    }
  else if (mg_data.coarse_solver.type == "cg_with_chebyshev")
    {
      // CG with Chebyshev as preconditioner

      typename SmootherType::AdditionalData smoother_data;

      smoother_data.preconditioner =
        std::make_shared<DiagonalMatrix<VectorType>>();
      mg_matrices[min_level]->compute_inverse_diagonal(
        smoother_data.preconditioner->get_vector());
      smoother_data.smoothing_range     = mg_data.smoother.smoothing_range;
      smoother_data.degree              = mg_data.smoother.degree;
      smoother_data.eig_cg_n_iterations = mg_data.smoother.eig_cg_n_iterations;

      precondition_chebyshev.initialize(*mg_matrices[min_level], smoother_data);

      mg_coarse = std::make_unique<
        MGCoarseGridIterativeSolver<VectorType,
                                    SolverCG<VectorType>,
                                    LevelMatrixType,
                                    decltype(precondition_chebyshev)>>(
        coarse_grid_solver, *mg_matrices[min_level], precondition_chebyshev);
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
      precondition_amg.initialize(mg_matrices[min_level]->get_system_matrix(),
                                  amg_data);

      mg_coarse = std::make_unique<
        MGCoarseGridIterativeSolver<VectorType,
                                    SolverCG<VectorType>,
                                    LevelMatrixType,
                                    decltype(precondition_amg)>>(
        coarse_grid_solver, *mg_matrices[min_level], precondition_amg);
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

#ifndef DOXYGEN



#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

#endif
