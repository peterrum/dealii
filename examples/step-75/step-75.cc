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


// @sect4{Include files}
//
// Include files
// TODO: need cleanup
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/generic_linear_algebra.h>

// uncomment the following \#define if you have PETSc and Trilinos installed
// and you prefer using Trilinos in this example:
// @code
#define FORCE_USE_OF_TRILINOS
// @endcode

// This will either import PETSc or TrilinosWrappers into the namespace
// LA. Note that we are defining the macro USE_PETSC_LA so that we can detect
// if we are using PETSc (see solve() for an example where this is necessary)
namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
} // namespace LA

#include <deal.II/lac/la_parallel_vector.h>


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
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parameter_acceptor.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_solver.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <fstream>
#include <memory>
#include <iostream>

namespace Step75
{
  using namespace dealii;

  // @sect4{Utility functions}

  // Helper functions for this tutorial.
  template <typename MeshType>
  MPI_Comm get_mpi_comm(const MeshType &mesh)
  {
    const auto *tria_parallel = dynamic_cast<
      const parallel::TriangulationBase<MeshType::dimension,
                                        MeshType::space_dimension> *>(
      &(mesh.get_triangulation()));

    return tria_parallel != nullptr ? tria_parallel->get_communicator() :
                                      MPI_COMM_SELF;
  }



  // @sect3{The <code>Parameter</code> class implementation}

  // Parameter class.

  // forward declarations
  template <int dim>
  class LaplaceProblem;

  template <int dim>
  class AdaptationStrategy;

  class SolverAMG;
  class SolverGMG;



  class AdaptationParameters : public ParameterAcceptor
  {
  public:
    AdaptationParameters();

  private:
    std::string  type;
    unsigned int min_level, max_level;
    unsigned int min_degree, max_degree;

    // double refine_fraction, coarsen_fraction;
    // double hp_refine_fraction, hp_coarsen_fraction;

    template <int dim>
    friend class AdaptationStrategy;
    template <int dim>
    friend class LaplaceProblem;
  };


  AdaptationParameters::AdaptationParameters()
    : ParameterAcceptor("adaptation")
  {
    type = "hp_Legendre";
    add_parameter("type", type);

    min_level = 5;
    add_parameter("minlevel", min_level);

    max_level = 10;
    add_parameter("maxlevel", max_level);

    min_degree = 2;
    add_parameter("mindegree", min_degree);

    max_degree = 7;
    add_parameter("maxdegree", max_degree);
  }



  class OperatorParameters : public ParameterAcceptor
  {
  public:
    OperatorParameters();

  private:
    std::string type;

    template <int dim>
    friend class LaplaceProblem;
  };

  OperatorParameters::OperatorParameters()
    : ParameterAcceptor("operator")
  {
    type = "MatrixFree";
    add_parameter("type", type);
  }



  class SolverParameters : public ParameterAcceptor
  {
  public:
    SolverParameters();

  private:
    std::string type;

    friend class SolverAMG;
    friend class SolverGMG;
    template <int dim>
    friend class LaplaceProblem;
  };

  SolverParameters::SolverParameters()
    : ParameterAcceptor("solver")
  {
    type = "GMG";
    add_parameter("type", type);
  }



  class ProblemParameters : public ParameterAcceptor
  {
  public:
    ProblemParameters();

  private:
    unsigned int dim;
    unsigned int n_cycles;

    AdaptationParameters prm_adaptation;
    OperatorParameters   prm_operator;
    SolverParameters     prm_solver;

    template <int dim>
    friend class LaplaceProblem;
  };

  ProblemParameters::ProblemParameters()
    : ParameterAcceptor("problem")
  {
    dim = 2;
    add_parameter("dimension", dim);

    n_cycles = 8;
    add_parameter("ncycles", n_cycles);
  }



  // @sect3{The <code>Solution</code> class template}

  // Analytic solution for the scenario described above.
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



  // @sect3{Matrix-free Laplace operator}

  // A matrix-free implementation of the Laplace operator.
  template <int dim, typename number>
  class LaplaceOperatorMatrixFree : public MGSolverOperatorBase<dim, number>
  {
  public:
    using typename MGSolverOperatorBase<dim, number>::VectorType;

    // An alias to the FEEvaluation class. Please note that, in contrast to
    // other tutorials, the template arguments `degree` is set to -1 and
    // `number of quadrature in 1D` to 0. In this case, FEEvaluation selects
    // dynamically the correct degree and number of quadrature points. The
    // need for dynamical decisions within FEEvaluation and possibly the
    // lack of knowledge of matrix sizes during sum factorization might lead
    // to a performance drop (up to 50%) compared to a templated approach,
    // however, allows us to write here simple code without the need to
    // explicitly deal with FEEvaluation instances with different template
    // arguments, e.g., via jump tables.
    using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, number>;

    // Initialize the internal MatrixFree instance and compute the system
    // right-hand-side vector
    void reinit(const hp::MappingCollection<dim> &mapping,
                const DoFHandler<dim> &           dof_handler,
                const hp::QCollection<dim> &      quad,
                const AffineConstraints<number> & constraints,
                VectorType &                      system_rhs);

    // Since we do not have a matrix, query the DoFHandler for the number of
    // degrees of freedom.
    types::global_dof_index m() const override;

    // Delegate the task to MatrixFree.
    void initialize_dof_vector(VectorType &vec) const override;

    // Perform an operator evaluation by looping with the help of MatrixFree
    // over all cells and evaluating the effect of the cell integrals (see also:
    // do_cell_integral_local() and do_cell_integral_global()).
    void vmult(VectorType &dst, const VectorType &src) const override;

    // Perform the transposed operator evaluation. Since we are considering
    // symmetric matrices, this function is identical to the above function.
    void Tvmult(VectorType &dst, const VectorType &src) const override;

    // Since we do not have a system matrix, we cannot loop over the the
    // diagonal entries of the matrix. Instead, we compute the diagonal by
    // performing a sequence of operator evaluations to unit basis vectors.
    // For this purpose, an optimized function from the MatrixFreeTools
    // namespace is used.
    void compute_inverse_diagonal(VectorType &diagonal) const override;

    // In the default case, no system matrix is set up during initialization
    // of this class. As a consequence, it has to be computed here. Just like
    // in the case of compute_inverse_diagonal(), the matrix entries are
    // obtained via sequence of operator evaluations. For this purpose, an
    // optimized function from the MatrixFreeTools namespace is used.
    const TrilinosWrappers::SparseMatrix &get_system_matrix() const override;

  private:
    // Perform cell integral on a cell batch without gathering and scattering
    // the values. This function is needed for the MatrixFreeTools functions
    // since these functions operate directly on the buffers of FEEvaluation.
    void do_cell_integral_local(FECellIntegrator &integrator) const;

    // Same as above but with access to the global vectors.
    void do_cell_integral_global(FECellIntegrator &integrator,
                                 VectorType &      dst,
                                 const VectorType &src) const;

    // This function loops over all cell batches within a cell-batch range and
    // calls the above function.
    void do_cell_integral_range(
      const MatrixFree<dim, number> &              matrix_free,
      VectorType &                                 dst,
      const VectorType &                           src,
      const std::pair<unsigned int, unsigned int> &range) const;

    // MatrixFree object.
    MatrixFree<dim, number> matrix_free;

    // Constraints potentially needed for the computation of the system matrix.
    AffineConstraints<number> constraints;

    // System matrix. In the default case, this matrix is empty. However, once
    // get_system_matrix() is called, this matrix is filled.
    mutable TrilinosWrappers::SparseMatrix system_matrix;
  };



  template <int dim, typename number>
  void LaplaceOperatorMatrixFree<dim, number>::reinit(
    const hp::MappingCollection<dim> &mapping,
    const DoFHandler<dim> &           dof_handler,
    const hp::QCollection<dim> &      quad,
    const AffineConstraints<number> & constraints,
    VectorType &                      system_rhs)
  {
    // Clear internal data structures (if operator is reused).
    this->system_matrix.clear();

    // Copy the constrains, since they might be needed for computation of the
    // system matrix later on.
    this->constraints.copy_from(constraints);

    // Set up MatrixFree. At the quadrature points, we only need to evaluate
    // the gradient of the solution and test with the gradient of the shape
    // functions so that we only need to set the flag `update_gradients`.
    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_gradients;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);

    // Compute the right-hand side vector. For this purpose, we set up a second
    // MatrixFree instance that uses a modified ConstraintMatrix not containing
    // the constraints due to Dirichlet-boundary conditions. This modified
    // operator is applied to a vector with only the Dirichlet values set. The
    // result is the negative right-hand-side vector.
    {
      AffineConstraints<number> constraints_without_dbc;

      IndexSet locally_relevant_dofs;
      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);
      constraints_without_dbc.reinit(locally_relevant_dofs);

      DoFTools::make_hanging_node_constraints(dof_handler,
                                              constraints_without_dbc);
      constraints_without_dbc.close();

      VectorType b, x;

      this->initialize_dof_vector(system_rhs);
      this->initialize_dof_vector(b);
      this->initialize_dof_vector(x);

      MatrixFree<dim, number> matrix_free;
      matrix_free.reinit(
        mapping, dof_handler, constraints_without_dbc, quad, data);

      constraints.distribute(x);

      matrix_free.cell_loop(&LaplaceOperatorMatrixFree::do_cell_integral_range,
                            this,
                            b,
                            x);

      constraints.set_zero(b);

      system_rhs -= b;
    }
  }



  template <int dim, typename number>
  types::global_dof_index LaplaceOperatorMatrixFree<dim, number>::m() const
  {
    return matrix_free.get_dof_handler().n_dofs();
  }



  template <int dim, typename number>
  void LaplaceOperatorMatrixFree<dim, number>::initialize_dof_vector(
    VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }



  template <int dim, typename number>
  void
  LaplaceOperatorMatrixFree<dim, number>::vmult(VectorType &      dst,
                                                const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &LaplaceOperatorMatrixFree::do_cell_integral_range, this, dst, src, true);
  }



  template <int dim, typename number>
  void
  LaplaceOperatorMatrixFree<dim, number>::Tvmult(VectorType &      dst,
                                                 const VectorType &src) const
  {
    this->vmult(dst, src);
  }



  template <int dim, typename number>
  void LaplaceOperatorMatrixFree<dim, number>::compute_inverse_diagonal(
    VectorType &diagonal) const
  {
    // compute diagonal
    MatrixFreeTools::compute_diagonal(
      matrix_free,
      diagonal,
      &LaplaceOperatorMatrixFree::do_cell_integral_local,
      this);

    // and invert it
    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }



  template <int dim, typename number>
  const TrilinosWrappers::SparseMatrix &
  LaplaceOperatorMatrixFree<dim, number>::get_system_matrix() const
  {
    // Check if matrix has already been set up.
    if (system_matrix.m() == 0 && system_matrix.n() == 0)
      {
        // Set up sparsity pattern of system matrix.
        const auto &dof_handler = this->matrix_free.get_dof_handler();

        TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(),
                                              get_mpi_comm(dof_handler));

        DoFTools::make_sparsity_pattern(dof_handler, dsp, this->constraints);

        dsp.compress();
        system_matrix.reinit(dsp);

        // Assemble system matrix.
        MatrixFreeTools::compute_matrix(
          matrix_free,
          constraints,
          system_matrix,
          &LaplaceOperatorMatrixFree::do_cell_integral_local,
          this);
      }

    return this->system_matrix;
  }



  template <int dim, typename number>
  void LaplaceOperatorMatrixFree<dim, number>::do_cell_integral_local(
    FECellIntegrator &integrator) const
  {
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate(EvaluationFlags::gradients);
  }



  template <int dim, typename number>
  void LaplaceOperatorMatrixFree<dim, number>::do_cell_integral_global(
    FECellIntegrator &integrator,
    VectorType &      dst,
    const VectorType &src) const
  {
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
  }



  template <int dim, typename number>
  void LaplaceOperatorMatrixFree<dim, number>::do_cell_integral_range(
    const MatrixFree<dim, number> &              matrix_free,
    VectorType &                                 dst,
    const VectorType &                           src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);

    for (unsigned cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);

        do_cell_integral_global(integrator, dst, src);
      }
  }



  // @sect3{Solver and preconditioner}

  // @sect4{Conjugate-gradient solver preconditioned by a algebraic multigrid approach}

  template <typename VectorType, typename Operator>
  void solve_with_amg(SolverControl &   solver_control,
                      const Operator &  system_matrix,
                      VectorType &      dst,
                      const VectorType &src)
  {
    LA::MPI::PreconditionAMG::AdditionalData data;
    data.elliptic              = true;
    data.higher_order_elements = true;

    LA::MPI::PreconditionAMG preconditioner;
    preconditioner.initialize(system_matrix.get_system_matrix(), data);

    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);
    cg.solve(system_matrix, dst, src, preconditioner);
  }



  // @sect4{Conjugate-gradient solver preconditioned by hybrid polynomial-global-coarsening multigrid approach}

  template <typename VectorType, typename Operator, int dim>
  void solve_with_gmg(SolverControl &                  solver_control,
                      const Operator &                 system_matrix,
                      VectorType &                     dst,
                      const VectorType &               src,
                      const hp::MappingCollection<dim> mapping_collection,
                      const DoFHandler<dim> &          dof_handler,
                      const hp::QCollection<dim> &     quadrature_collection)
  {
    const GMGParameters mg_data; // TODO -> MF

    // Create a DoFHandler and operator for each multigrid level defined
    // by p-coarsening, as well as, create transfer operators.
    MGLevelObject<DoFHandler<dim>> dof_handlers;
    MGLevelObject<std::unique_ptr<
      MGSolverOperatorBase<dim, typename VectorType::value_type>>>
                                                       operators;
    MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;

    MGLevelObject<std::shared_ptr<Triangulation<dim>>>
      coarse_grid_triangulations;

    if (mg_data.perform_h_transfer)
      MGTransferGlobalCoarseningTools::create_global_coarsening_sequence(
        coarse_grid_triangulations, dof_handler.get_triangulation());

    const unsigned int n_h_levels = coarse_grid_triangulations.max_level() -
                                    coarse_grid_triangulations.min_level();

    // Determine the number of levels.
    const auto get_max_active_fe_index = [&](const auto &dof_handler) {
      unsigned int min = 0;

      for (auto &cell : dof_handler.active_cell_iterators())
        {
          if (cell->is_locally_owned())
            min = std::max(min, cell->active_fe_index());
        }

      return Utilities::MPI::max(min, MPI_COMM_WORLD);
    };

    const unsigned int n_p_levels =
      MGTransferGlobalCoarseningTools::create_p_sequence(
        get_max_active_fe_index(dof_handler) + 1, mg_data.p_sequence)
        .size();

    unsigned int minlevel   = 0;
    unsigned int minlevel_p = n_h_levels;
    unsigned int maxlevel   = n_h_levels + n_p_levels - 1;

    // Allocate memory for all levels.
    dof_handlers.resize(minlevel, maxlevel);
    operators.resize(minlevel, maxlevel);
    transfers.resize(minlevel, maxlevel);

    // Loop from max to min level and set up DoFHandler with coarser mesh...
    for (unsigned int l = 0; l < n_h_levels; ++l)
      {
        dof_handlers[l].reinit(*coarse_grid_triangulations[l]);
        dof_handlers[l].distribute_dofs(dof_handler.get_fe_collection());
      }

    // ... with lower polynomial degrees
    for (unsigned int i = 0, l = maxlevel; i < n_p_levels; ++i, --l)
      {
        dof_handlers[l].reinit(dof_handler.get_triangulation());

        if (l == maxlevel) // finest level
          {
            auto &dof_handler_mg = dof_handlers[l];

            auto cell_other = dof_handler.begin_active();
            for (auto &cell : dof_handler_mg.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  cell->set_active_fe_index(cell_other->active_fe_index());
                cell_other++;
              }
          }
        else // coarse level
          {
            auto &dof_handler_fine   = dof_handlers[l + 1];
            auto &dof_handler_coarse = dof_handlers[l + 0];

            auto cell_other = dof_handler_fine.begin_active();
            for (auto &cell : dof_handler_coarse.active_cell_iterators())
              {
                if (cell->is_locally_owned())
                  cell->set_active_fe_index(
                    MGTransferGlobalCoarseningTools::generate_level_degree(
                      cell_other->active_fe_index() + 1, mg_data.p_sequence) -
                    1);
                cell_other++;
              }
          }

        dof_handlers[l].distribute_dofs(dof_handler.get_fe_collection());
      }

    // Create data structures on each multigrid level.
    MGLevelObject<AffineConstraints<typename VectorType::value_type>>
      constraints(minlevel, maxlevel);

    for (unsigned int level = minlevel; level <= maxlevel; level++)
      {
        const auto &dof_handler = dof_handlers[level];
        auto &      constraint  = constraints[level];

        // ... constraints (with homogenous Dirichlet BC)
        {
          IndexSet locally_relevant_dofs;
          DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                  locally_relevant_dofs);
          constraint.reinit(locally_relevant_dofs);


          DoFTools::make_hanging_node_constraints(dof_handler, constraint);
          VectorTools::interpolate_boundary_values(
            mapping_collection,
            dof_handler,
            0,
            Functions::ZeroFunction<dim>(),
            constraint);
          constraint.close();
        }

        // ... operator (just like on the finest level)
        {
          VectorType dummy;

          auto temp = std::make_unique<Operator>();
          temp->reinit(mapping_collection,
                       dof_handler,
                       quadrature_collection,
                       constraint,
                       dummy);

          operators[level] = std::move(temp);
        }
      }

    // Set up intergrid operators.
    for (unsigned int level = minlevel; level < minlevel_p; ++level)
      transfers[level + 1].reinit_geometric_transfer(dof_handlers[level + 1],
                                                     dof_handlers[level],
                                                     constraints[level + 1],
                                                     constraints[level]);

    for (unsigned int level = minlevel_p; level < maxlevel; ++level)
      transfers[level + 1].reinit_polynomial_transfer(dof_handlers[level + 1],
                                                      dof_handlers[level],
                                                      constraints[level + 1],
                                                      constraints[level]);

    // Collect transfer operators within a single operator as needed by
    // the Multigrid solver class.
    MGTransferGlobalCoarsening<
      MGSolverOperatorBase<dim, typename VectorType::value_type>,
      VectorType>
      transfer(operators, transfers);

    // Proceed to solve the problem with multigrid.
    mg_solve(solver_control,
             dst,
             src,
             mg_data,
             dof_handler,
             system_matrix,
             operators,
             transfer);
  }



  // @sect3{The <code>AdaptationStrategy</code> class template}

  // Different strategies to perform hp adaptation.
  template <int dim>
  class AdaptationStrategy
  {
  public:
    AdaptationStrategy(const AdaptationParameters &               prm,
                       DoFHandler<dim> &                          dof_handler,
                       parallel::distributed::Triangulation<dim> &triangulation)
      : prm(prm)
      , dof_handler(dof_handler)
      , triangulation(triangulation)
    {
      for (unsigned int degree = prm.min_degree; degree <= prm.max_degree;
           ++degree)
        face_quadrature_collection.push_back(QGauss<dim - 1>(degree + 1));
    };

    virtual void
    adapt_resolution(const LinearAlgebra::distributed::Vector<double>
                       &locally_relevant_solution);

  protected:
    void flag_adaptation(const LinearAlgebra::distributed::Vector<double>
                           &locally_relevant_solution);
    virtual void decide_hp(const LinearAlgebra::distributed::Vector<double>
                             &locally_relevant_solution);
    void         limit_levels();
    virtual void execute_refinement();

    const AdaptationParameters &prm;

    DoFHandler<dim> &                          dof_handler;
    parallel::distributed::Triangulation<dim> &triangulation;

    hp::QCollection<dim - 1> face_quadrature_collection;

    Vector<float> estimated_error_per_cell;
  };



  template <int dim>
  void AdaptationStrategy<dim>::adapt_resolution(
    const LinearAlgebra::distributed::Vector<double> &locally_relevant_solution)
  {
    flag_adaptation(locally_relevant_solution);
    decide_hp(locally_relevant_solution);
    limit_levels();
    execute_refinement();
  }



  template <int dim>
  void AdaptationStrategy<dim>::flag_adaptation(
    const LinearAlgebra::distributed::Vector<double> &locally_relevant_solution)
  {
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
  void AdaptationStrategy<dim>::decide_hp(
    const LinearAlgebra::distributed::Vector<double>
      & /*locally_relevant_solution*/)
  {}



  template <int dim>
  void AdaptationStrategy<dim>::limit_levels()
  {
    Assert(triangulation.n_levels() >= prm.min_level + 1 &&
             triangulation.n_levels() <= prm.max_level + 1,
           ExcInternalError());

    if (triangulation.n_levels() > prm.max_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(prm.max_level))
        cell->clear_refine_flag();

    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(prm.min_level))
      cell->clear_coarsen_flag();
  }



  template <int dim>
  void AdaptationStrategy<dim>::execute_refinement()
  {
    triangulation.execute_coarsening_and_refinement();
  }



  template <int dim>
  class hpLegendreStrategy : public AdaptationStrategy<dim>
  {
  public:
    hpLegendreStrategy(const AdaptationParameters &               prm,
                       DoFHandler<dim> &                          dof_handler,
                       parallel::distributed::Triangulation<dim> &triangulation,
                       hp::FECollection<dim> &                    fe_collection)
      : AdaptationStrategy<dim>(prm, dof_handler, triangulation)
      , legendre(
          SmoothnessEstimator::Legendre::default_fe_series(fe_collection))
    {
      legendre.precalculate_all_transformation_matrices();
    };

  private:
    void decide_hp(const LinearAlgebra::distributed::Vector<double>
                     &locally_relevant_solution) override;

    FESeries::Legendre<dim> legendre;

    Vector<float> hp_decision_indicators;
  };



  template <int dim>
  void hpLegendreStrategy<dim>::decide_hp(
    const LinearAlgebra::distributed::Vector<double> &locally_relevant_solution)
  {
    hp_decision_indicators.grow_or_shrink(this->triangulation.n_active_cells());

    SmoothnessEstimator::Legendre::coefficient_decay(
      legendre,
      this->dof_handler,
      locally_relevant_solution,
      hp_decision_indicators,
      /*regression_strategy=*/VectorTools::Linfty_norm,
      /*smallest_abs_coefficient=*/1e-10,
      /*only_flagged_cells=*/true);

    hp::Refinement::p_adaptivity_fixed_number(this->dof_handler,
                                              hp_decision_indicators,
                                              0.9,
                                              0.9);
    hp::Refinement::choose_p_over_h(this->dof_handler);
  }



  template <int dim>
  class hpFourierStrategy : public AdaptationStrategy<dim>
  {
  public:
    hpFourierStrategy(const AdaptationParameters &               prm,
                      DoFHandler<dim> &                          dof_handler,
                      parallel::distributed::Triangulation<dim> &triangulation,
                      hp::FECollection<dim> &                    fe_collection)
      : AdaptationStrategy<dim>(prm, dof_handler, triangulation)
      , fourier(SmoothnessEstimator::Fourier::default_fe_series(fe_collection))
    {
      fourier.precalculate_all_transformation_matrices();
    };
    void decide_hp(const LinearAlgebra::distributed::Vector<double>
                     &locally_relevant_solution) override;

  private:
    FESeries::Fourier<dim> fourier;

    Vector<float> hp_decision_indicators;
  };



  template <int dim>
  void hpFourierStrategy<dim>::decide_hp(
    const LinearAlgebra::distributed::Vector<double> &locally_relevant_solution)
  {
    hp_decision_indicators.grow_or_shrink(this->triangulation.n_active_cells());

    SmoothnessEstimator::Fourier::coefficient_decay(
      fourier,
      this->dof_handler,
      locally_relevant_solution,
      hp_decision_indicators,
      /*regression_strategy=*/VectorTools::Linfty_norm,
      /*smallest_abs_coefficient=*/1e-10,
      /*only_flagged_cells=*/true);

    hp::Refinement::p_adaptivity_fixed_number(this->dof_handler,
                                              hp_decision_indicators,
                                              0.9,
                                              0.9);
    hp::Refinement::choose_p_over_h(this->dof_handler);
  }



  template <int dim>
  class hpHistoryStrategy : public AdaptationStrategy<dim>
  {
  public:
    hpHistoryStrategy(const AdaptationParameters &               prm,
                      DoFHandler<dim> &                          dof_handler,
                      parallel::distributed::Triangulation<dim> &triangulation)
      : AdaptationStrategy<dim>(prm, dof_handler, triangulation)
      , error_predictor(dof_handler)
      , init_step(true){};

    void adapt_resolution(const LinearAlgebra::distributed::Vector<double>
                            &locally_relevant_solution) override;

  private:
    void decide_hp(const LinearAlgebra::distributed::Vector<double>
                     &locally_relevant_solution) override;
    void execute_refinement() override;


    parallel::distributed::ErrorPredictor<dim> error_predictor;

    bool          init_step;
    Vector<float> hp_decision_indicators;
    Vector<float> predicted_error_per_cell;
  };


  template <int dim>
  void hpHistoryStrategy<dim>::adapt_resolution(
    const LinearAlgebra::distributed::Vector<double> &locally_relevant_solution)
  {
    if (init_step)
      {
        this->estimated_error_per_cell.grow_or_shrink(
          this->triangulation.n_active_cells());

        KellyErrorEstimator<dim>::estimate(
          this->dof_handler,
          this->face_quadrature_collection,
          std::map<types::boundary_id, const Function<dim> *>(),
          locally_relevant_solution,
          this->estimated_error_per_cell,
          /*component_mask=*/ComponentMask(),
          /*coefficients=*/nullptr,
          /*n_threads=*/numbers::invalid_unsigned_int,
          /*subdomain_id=*/numbers::invalid_subdomain_id,
          /*material_id=*/numbers::invalid_material_id,
          /*strategy=*/
          KellyErrorEstimator<
            dim>::Strategy::face_diameter_over_twice_max_degree);

        for (const auto &cell : this->triangulation.active_cell_iterators())
          if (cell->is_locally_owned())
            cell->set_refine_flag();

        execute_refinement();

        init_step = false;
      }
    else
      {
        this->flag_adaptation(locally_relevant_solution);
        decide_hp(locally_relevant_solution);
        this->limit_levels();
        execute_refinement();
      }
  }



  template <int dim>
  void hpHistoryStrategy<dim>::decide_hp(
    const LinearAlgebra::distributed::Vector<double>
      & /*locally_relevant_solution*/)
  {
    hp_decision_indicators.grow_or_shrink(this->triangulation.n_active_cells());

    for (unsigned int i = 0; i < this->triangulation.n_active_cells(); ++i)
      hp_decision_indicators(i) =
        predicted_error_per_cell(i) - this->estimated_error_per_cell(i);

    const float global_minimum =
      Utilities::MPI::min(*std::min_element(hp_decision_indicators.begin(),
                                            hp_decision_indicators.end()),
                          get_mpi_comm(this->triangulation));
    if (global_minimum < 0)
      for (auto &indicator : hp_decision_indicators)
        indicator -= global_minimum;

    hp::Refinement::p_adaptivity_fixed_number(this->dof_handler,
                                              hp_decision_indicators,
                                              0.9,
                                              0.9);
    hp::Refinement::choose_p_over_h(this->dof_handler);
  }



  template <int dim>
  void hpHistoryStrategy<dim>::execute_refinement()
  {
    error_predictor.prepare_for_coarsening_and_refinement(
      this->estimated_error_per_cell,
      /*gamma_p=*/std::sqrt(0.4),
      /*gamma_h=*/2.,
      /*gamma_n=*/1.);

    this->triangulation.execute_coarsening_and_refinement();

    predicted_error_per_cell.grow_or_shrink(
      this->triangulation.n_active_cells());
    error_predictor.unpack(predicted_error_per_cell);
  }



  // @sect3{The <code>LaplaceProblem</code> class template}

  // Solving the Laplace equation on subsequently refined function spaces.
  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem(const ProblemParameters &prm);

    void run();

  private:
    void create_coarse_grid();
    void setup_system();

    template <typename Operator>
    void
    solve(const Operator &                            system_matrix,
          LinearAlgebra::distributed::Vector<double> &locally_relevant_solution,
          const LinearAlgebra::distributed::Vector<double> &system_rhs);

    void compute_errors();
    void output_results(const unsigned int cycle) const;

    MPI_Comm mpi_communicator;

    const ProblemParameters &prm;

    parallel::distributed::Triangulation<dim> triangulation;
    const unsigned int                        min_level, max_level;

    DoFHandler<dim>            dof_handler;
    hp::MappingCollection<dim> mapping_collection;
    hp::FECollection<dim>      fe_collection;
    hp::QCollection<dim>       quadrature_collection;

    std::unique_ptr<hp::FEValues<dim>>       fe_values_collection;
    std::unique_ptr<AdaptationStrategy<dim>> adaptation_strategy;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> constraints;

    LA::MPI::SparseMatrix                      system_matrix;
    LinearAlgebra::distributed::Vector<double> locally_relevant_solution;
    LinearAlgebra::distributed::Vector<double> system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };



  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem(const ProblemParameters &prm)
    : mpi_communicator(MPI_COMM_WORLD)
    , prm(prm)
    , triangulation(mpi_communicator,
                    typename Triangulation<dim>::MeshSmoothing(
                      Triangulation<dim>::smoothing_on_refinement |
                      Triangulation<dim>::smoothing_on_coarsening))
    , min_level(5)
    , max_level(dim <= 2 ? 10 : 8)
    , dof_handler(triangulation)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {
    TimerOutput::Scope t(computing_timer, "init");

    Assert(prm.prm_adaptation.min_level <= prm.prm_adaptation.max_level,
           ExcMessage(
             "Triangulation level limits have been incorrectly set up."));
    Assert(prm.prm_adaptation.min_degree <= prm.prm_adaptation.max_degree,
           ExcMessage("FECollection degrees have been incorrectly set up."));

    mapping_collection.push_back(MappingQ1<dim>());

    for (unsigned int degree = prm.prm_adaptation.min_degree;
         degree <= prm.prm_adaptation.max_degree;
         ++degree)
      {
        fe_collection.push_back(FE_Q<dim>(degree));
        quadrature_collection.push_back(QGauss<dim>(degree + 1));
      }

    fe_values_collection =
      std::make_unique<hp::FEValues<dim>>(fe_collection,
                                          quadrature_collection,
                                          update_gradients |
                                            update_quadrature_points |
                                            update_JxW_values);
    fe_values_collection->precalculate_fe_values();

    if (prm.prm_adaptation.type == "h")
      adaptation_strategy =
        std::make_unique<AdaptationStrategy<dim>>(prm.prm_adaptation,
                                                  dof_handler,
                                                  triangulation);
    else if (prm.prm_adaptation.type == "hp_Legendre")
      adaptation_strategy = std::make_unique<hpLegendreStrategy<dim>>(
        prm.prm_adaptation, dof_handler, triangulation, fe_collection);
    else if (prm.prm_adaptation.type == "hp_Fourier")
      adaptation_strategy = std::make_unique<hpFourierStrategy<dim>>(
        prm.prm_adaptation, dof_handler, triangulation, fe_collection);
    else if (prm.prm_adaptation.type == "hp_History")
      adaptation_strategy =
        std::make_unique<hpHistoryStrategy<dim>>(prm.prm_adaptation,
                                                 dof_handler,
                                                 triangulation);
    else
      AssertThrow(false, ExcNotImplemented());
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
    VectorTools::interpolate_boundary_values(
      mapping_collection, dof_handler, 0, Solution<dim>(), constraints);

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
  template <typename Operator>
  void LaplaceProblem<dim>::solve(
    const Operator &                                  system_matrix,
    LinearAlgebra::distributed::Vector<double> &      locally_relevant_solution,
    const LinearAlgebra::distributed::Vector<double> &system_rhs)
  {
    TimerOutput::Scope t(computing_timer, "solve");

    LinearAlgebra::distributed::Vector<double> locally_relevant_solution_;
    LinearAlgebra::distributed::Vector<double> system_rhs_;

    system_matrix.initialize_dof_vector(locally_relevant_solution_);
    system_matrix.initialize_dof_vector(system_rhs_);

    system_rhs_.copy_locally_owned_data_from(system_rhs);

    SolverControl solver_control(system_rhs_.size(),
                                 1e-12 * system_rhs_.l2_norm());

    if (prm.prm_solver.type == "AMG")
      solve_with_amg(solver_control,
                     system_matrix,
                     locally_relevant_solution_,
                     system_rhs_);
    else if (prm.prm_solver.type == "GMG")
      solve_with_gmg(solver_control,
                     system_matrix,
                     locally_relevant_solution_,
                     system_rhs_,
                     mapping_collection,
                     dof_handler,
                     quadrature_collection);
    else
      Assert(false, ExcNotImplemented());

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(locally_relevant_solution_);

    locally_relevant_solution.copy_locally_owned_data_from(
      locally_relevant_solution_);
    locally_relevant_solution.update_ghost_values();
  }



  template <int dim>
  void LaplaceProblem<dim>::compute_errors()
  {
    TimerOutput::Scope t(computing_timer, "compute errors");

    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(mapping_collection,
                                      dof_handler,
                                      locally_relevant_solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      quadrature_collection,
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(mapping_collection,
                                      dof_handler,
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
    data_out.build_patches(mapping_collection);

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2, 8);
  }



  template <int dim>
  void LaplaceProblem<dim>::run()
  {
    pcout << "Running with "
#ifdef USE_PETSC_LA
          << "PETSc"
#else
          << "Trilinos"
#endif
          << " on " << Utilities::MPI::n_mpi_processes(mpi_communicator)
          << " MPI rank(s)..." << std::endl;

    LaplaceOperatorMatrixFree<dim, double> laplace_operator;

    for (unsigned int cycle = 0;
         cycle < (prm.prm_adaptation.type != "hpHistory" ? prm.n_cycles :
                                                           prm.n_cycles + 1);
         ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            create_coarse_grid();
            triangulation.refine_global(prm.prm_adaptation.type != "hpHistory" ?
                                          prm.prm_adaptation.min_level :
                                          prm.prm_adaptation.min_level - 1);
          }
        else
          {
            adaptation_strategy->adapt_resolution(locally_relevant_solution);
          }

        setup_system();

        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl
              << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        laplace_operator.reinit(mapping_collection,
                                dof_handler,
                                quadrature_collection,
                                constraints,
                                system_rhs);

        solve(laplace_operator, locally_relevant_solution, system_rhs);
        compute_errors();

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
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



// @sect4{main()}

// The final function.
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step75;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      ProblemParameters prm_problem;

      const std::string filename        = (argc > 1) ? argv[1] : "",
                        output_filename = (argc > 1) ? "" : "default.json";
      ParameterAcceptor::initialize(filename, output_filename);

      const int dim =
        ParameterAcceptor::prm.get_integer({"problem"}, "dimension");
      if (dim == 2)
        {
          LaplaceProblem<2> laplace_problem(prm_problem);
          laplace_problem.run();
        }
      else if (dim == 3)
        {
          LaplaceProblem<3> laplace_problem(prm_problem);
          laplace_problem.run();
        }
      else
        Assert(false, ExcNotImplemented());
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
