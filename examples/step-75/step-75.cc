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

  template <int dim, typename number>
  class LaplaceOperator;



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

    template <int dim, typename number>
    friend class LaplaceOperator;
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



  // @sect3{The Laplace operator}

  // @sect4{Operator base}

  // The following class provides a minimal interface needed by the solvers
  // and preconditioners used in this tutorial. In particular the interface
  // consists of:
  //  - a function to initialize the operator
  //  - an operator evaluation function (vmult()), and
  //  - utility functions to extract some quantities from the matrix.
  //
  // Since some of the functions are identical in the matrix-based and
  // matrix-free case this class also contains the implementation of these
  // functions.
  template <int dim_, typename number>
  class LaplaceOperator : public Subscriptor
  {
  public:
    static const int dim = dim_;
    using value_type     = number;
    using VectorType     = LinearAlgebra::distributed::Vector<number>;

    // Initialize operator and compute the right-hand side.
    virtual void reinit(const hp::MappingCollection<dim> &mapping_collection,
                        const DoFHandler<dim> &           dof_handler,
                        const hp::QCollection<dim> &      quadrature_collection,
                        const AffineConstraints<number> & constraints,
                        VectorType &                      system_rhs) = 0;

    // Return number of rows of the matrix. Since we are dealing with a
    // symmetrical matrix, the returned value is the same as the number of
    // columns.
    virtual types::global_dof_index m() const = 0;

    // Access a particular element in the matrix. This function is neither
    // needed nor implemented, however, is required to compile the program.
    number el(unsigned int, unsigned int) const;

    // Allocate memory for a distributed vector.
    virtual void initialize_dof_vector(VectorType &vec) const = 0;

    // Perform an operator application on the vector @p src.
    virtual void vmult(VectorType &dst, const VectorType &src) const = 0;

    // Perform the transposed operator evaluation. Since we are considering
    // symmetric matrices, this function is identical to the above function.
    void Tvmult(VectorType &dst, const VectorType &src) const;

    // Compute the inverse of the diagonal of the vector and store it into the
    // provided vector. The inverse diagonal is used below in a Chebyshev
    // smoother.
    virtual void compute_inverse_diagonal(VectorType &diagonal) const = 0;

    // Return the actual system matrix, which can be used in any matrix-based
    // solvers (like AMG).
    virtual const TrilinosWrappers::SparseMatrix &get_system_matrix() const = 0;
  };



  template <int dim_, typename number>
  void LaplaceOperator<dim_, number>::Tvmult(VectorType &      dst,
                                             const VectorType &src) const
  {
    this->vmult(dst, src);
  }



  template <int dim_, typename number>
  number LaplaceOperator<dim_, number>::el(unsigned int, unsigned int) const
  {
    Assert(false, ExcNotImplemented());
    return 0;
  }



  // @sect4{Matrix-based operator}
  // The following class is a simple wrapper around a sparse matrix.
  template <int dim, typename number>
  class LaplaceOperatorMatrixBased : public LaplaceOperator<dim, number>
  {
  public:
    using typename LaplaceOperator<dim, number>::VectorType;

    // Set up a partitioner, as well as, compute system matrix and
    // right-hand-side vector.
    void reinit(const hp::MappingCollection<dim> &mapping_collection,
                const DoFHandler<dim> &           dof_handler,
                const hp::QCollection<dim> &      quadrature_collection,
                const AffineConstraints<number> & constraints,
                VectorType &                      system_rhs) override;

    // Query the matrix for its number of rows.
    types::global_dof_index m() const override;

    // Initialize vector via the precomputed partitioner.
    void initialize_dof_vector(VectorType &vec) const override;

    // The operator evaluation is a simple matrix-vector multiplication in
    // this case.
    void vmult(VectorType &dst, const VectorType &src) const override;

    // Computing the inverse diagonal is quite simply done by looping over
    // all entries on the diagonal of the matrix and inverting these values.
    void compute_inverse_diagonal(VectorType &diagonal) const override;

    // Since the matrix is the basis of all the relevant functions of this
    // class and as a consequence the matrix is explicitly stored, this function
    // simply returns a reference to the local matrix.
    const TrilinosWrappers::SparseMatrix &get_system_matrix() const override;

  private:
    // The actual system matrix.
    TrilinosWrappers::SparseMatrix system_matrix;

    // A partitioner used for initializing of ghosted vectors.
    std::shared_ptr<const Utilities::MPI::Partitioner> partitioner;
  };



  template <int dim, typename number>
  void LaplaceOperatorMatrixBased<dim, number>::reinit(
    const hp::MappingCollection<dim> &mapping_collection,
    const DoFHandler<dim> &           dof_handler,
    const hp::QCollection<dim> &      quadrature_collection,
    const AffineConstraints<number> & constraints,
    VectorType &                      system_rhs)
  {
#ifndef DEAL_II_WITH_TRILINOS
    Assert(false, StandardExceptions::ExcNotImplemented());
    (void)mapping_collection;
    (void)dof_handler;
    (void)quadrature_collection;
    (void)constraints;
    (void)system_rhs;
#else

    // Create partitioner.
    const auto create_partitioner = [](const DoFHandler<dim> &dof_handler) {
      IndexSet locally_relevant_dofs;

      DoFTools::extract_locally_relevant_dofs(dof_handler,
                                              locally_relevant_dofs);

      return std::make_shared<const Utilities::MPI::Partitioner>(
        dof_handler.locally_owned_dofs(),
        locally_relevant_dofs,
        get_mpi_comm(dof_handler));
    };

    this->partitioner = create_partitioner(dof_handler);

    // Allocate memory for system matrix and right-hand-side vector.
    TrilinosWrappers::SparsityPattern dsp(dof_handler.locally_owned_dofs(),
                                          get_mpi_comm(dof_handler));
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dsp.compress();

    system_matrix.reinit(dsp);

    initialize_dof_vector(system_rhs);

    // Assemble system matrix and right-hand-side vector.
    hp::FEValues<dim> hp_fe_values(mapping_collection,
                                   dof_handler.get_fe_collection(),
                                   quadrature_collection,
                                   update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values);

    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        if (cell->is_locally_owned() == false)
          continue;

        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
        cell_matrix = 0;
        cell_rhs.reinit(dofs_per_cell);
        cell_rhs = 0;
        hp_fe_values.reinit(cell);
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        for (unsigned int q_point = 0; q_point < fe_values.n_quadrature_points;
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

        constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }

    system_rhs.compress(VectorOperation::values::add);
    system_matrix.compress(VectorOperation::values::add);
#endif
  }



  template <int dim, typename number>
  types::global_dof_index LaplaceOperatorMatrixBased<dim, number>::m() const
  {
#ifdef DEAL_II_WITH_TRILINOS
    return system_matrix.m();
#else
    Assert(false, ExcNotImplemented());
    return 0;
#endif
  }



  template <int dim, typename number>
  void LaplaceOperatorMatrixBased<dim, number>::initialize_dof_vector(
    VectorType &vec) const
  {
    vec.reinit(partitioner);
  }



  template <int dim, typename number>
  void
  LaplaceOperatorMatrixBased<dim, number>::vmult(VectorType &      dst,
                                                 const VectorType &src) const
  {
    system_matrix.vmult(dst, src);
  }



  template <int dim, typename number>
  void LaplaceOperatorMatrixBased<dim, number>::compute_inverse_diagonal(
    VectorType &diagonal) const
  {
    this->initialize_dof_vector(diagonal);

#ifdef DEAL_II_WITH_TRILINOS
    for (auto entry : system_matrix)
      if (entry.row() == entry.column())
        diagonal[entry.row()] = 1.0 / entry.value();
#else
    Assert(false, ExcNotImplemented());
#endif
  }



  template <int dim, typename number>
  const TrilinosWrappers::SparseMatrix &
  LaplaceOperatorMatrixBased<dim, number>::get_system_matrix() const
  {
    return this->system_matrix;
  }



  // @sect4{Matrix-free operator}
  // A matrix-free implementation of the Laplace operator.
  template <int dim, typename number>
  class LaplaceOperatorMatrixFree : public LaplaceOperator<dim, number>
  {
  public:
    using typename LaplaceOperator<dim, number>::VectorType;

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
                VectorType &                      system_rhs) override;

    // Since we do not have a matrix, query the DoFHandler for the number of
    // degrees of freedom.
    types::global_dof_index m() const override;

    // Delegate the task to MatrixFree.
    void initialize_dof_vector(VectorType &vec) const override;

    // Perform an operator evaluation by looping with the help of MatrixFree
    // over all cells and evaluating the effect of the cell integrals (see also:
    // do_cell_integral_local() and do_cell_integral_global()).
    void vmult(VectorType &dst, const VectorType &src) const override;

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
      const dealii::MatrixFree<dim, number> &      matrix_free,
      VectorType &                                 dst,
      const VectorType &                           src,
      const std::pair<unsigned int, unsigned int> &range) const;

    // MatrixFree object.
    dealii::MatrixFree<dim, number> matrix_free;

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

      dealii::MatrixFree<dim, number> matrix_free;
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
    const dealii::MatrixFree<dim, number> &      matrix_free,
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

  class SolverAMG
  {
  public:
    template <typename VectorType, typename Operator>
    static void solve(SolverControl &   solver_control,
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
  };



  // @sect4{Conjugate-gradient solver preconditioned by hybrid polynomial-global-coarsening multigrid approach}

  class SolverGMG
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

    struct TestMultigridParameters
    {
      std::string            solver_type;
      unsigned int           maxiter  = 100;
      double                 abstol   = 1e-10;
      double                 reltol   = 1e-6;
      unsigned int           v_cycles = 1;
      SmootherParameters     smoother;
      CoarseSolverParameters coarse_solver;
    };

  public:
    template <typename VectorType, typename Operator, int dim>
    static void solve(SolverControl &                  solver_control,
                      const Operator &                 system_matrix,
                      VectorType &                     dst,
                      const VectorType &               src,
                      const hp::MappingCollection<dim> mapping_collection,
                      const DoFHandler<dim> &          dof_handler,
                      const hp::QCollection<dim> &     quadrature_collection)
    {
      if (const auto op = dynamic_cast<
            const LaplaceOperatorMatrixFree<dim,
                                            typename VectorType::value_type> *>(
            &system_matrix))
        solve_internal(solver_control,
                       *op,
                       dst,
                       src,
                       mapping_collection,
                       dof_handler,
                       quadrature_collection);
      else if (const auto op = dynamic_cast<const LaplaceOperatorMatrixBased<
                 dim,
                 typename VectorType::value_type> *>(&system_matrix))
        solve_internal(solver_control,
                       *op,
                       dst,
                       src,
                       mapping_collection,
                       dof_handler,
                       quadrature_collection);
      else
        Assert(false, ExcNotImplemented());
    }

  private:
    template <typename VectorType, typename Operator, int dim>
    static void
    solve_internal(SolverControl &                  solver_control,
                   const Operator &                 system_matrix,
                   VectorType &                     dst,
                   const VectorType &               src,
                   const hp::MappingCollection<dim> mapping_collection,
                   const DoFHandler<dim> &          dof_handler,
                   const hp::QCollection<dim> &     quadrature_collection)
    {
      // parameters
      const std::string p_sequence = "decreasebyone"; // TODO

      // TODO
      MGLevelObject<DoFHandler<dim>> dof_handlers;

      // Vector of transfer operators for each pair of levels
      MGLevelObject<MGTwoLevelTransfer<dim, VectorType>> transfers;

      // Vector of transfer operators for each pair of levels
      MGLevelObject<Operator> operators;

      const auto get_max_active_fe_index = [&](const auto &dof_handler) {
        unsigned int min = 0;

        for (auto &cell : dof_handler.active_cell_iterators())
          {
            if (cell->is_locally_owned())
              min = std::max(min, cell->active_fe_index());
          }

        return Utilities::MPI::max(min, MPI_COMM_WORLD);
      };

      const unsigned int n_levels =
        create_p_sequence(get_max_active_fe_index(dof_handler) + 1, p_sequence)
          .size();

      unsigned int minlevel = 0;
      unsigned int maxlevel = n_levels - 1;

      // 2) allocate memory for all levels
      dof_handlers.resize(minlevel, maxlevel, dof_handler.get_triangulation());

      // loop over levels
      for (unsigned int i = 0, l = maxlevel; i < n_levels; ++i, --l)
        {
          if (l == maxlevel) // set FEs on fine level
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
          else // set FEs on coarse level
            {
              auto &dof_handler_fine   = dof_handlers[l + 1];
              auto &dof_handler_coarse = dof_handlers[l + 0];

              auto cell_other = dof_handler_fine.begin_active();
              for (auto &cell : dof_handler_coarse.active_cell_iterators())
                {
                  if (cell->is_locally_owned())
                    cell->set_active_fe_index(
                      generate_level_degree(cell_other->active_fe_index() + 1,
                                            p_sequence) -
                      1);
                  cell_other++;
                }
            }

          // create dof_handler
          dof_handlers[l].distribute_dofs(dof_handler.get_fe_collection());
        }

      // 2) allocate memory for all levels
      transfers.resize(minlevel, maxlevel);
      operators.resize(minlevel, maxlevel);

      // 3) create objects on each multigrid level
      MGLevelObject<AffineConstraints<typename VectorType::value_type>>
        constraints(minlevel, maxlevel);

      for (unsigned int level = minlevel; level <= maxlevel; level++)
        {
          const auto &dof_handler = dof_handlers[level];
          auto &      constraint  = constraints[level];

          // b) setup constraint
          {
            IndexSet locally_relevant_dofs;
            DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                    locally_relevant_dofs);
            constraint.reinit(locally_relevant_dofs);


            DoFTools::make_hanging_node_constraints(dof_handler, constraint);
            VectorTools::interpolate_boundary_values(
              dof_handler, 0, Functions::ZeroFunction<dim>(), constraint);
            constraint.close();
          }

          // c) setup operator
          {
            VectorType dummy;
            operators[level].reinit(mapping_collection,
                                    dof_handler,
                                    quadrature_collection,
                                    constraint,
                                    dummy);
          }
        }

      // 4) set up intergrid operators
      for (unsigned int level = minlevel; level < maxlevel; level++)
        transfers[level + 1].reinit_polynomial_transfer(dof_handlers[level + 1],
                                                        dof_handlers[level],
                                                        constraints[level + 1],
                                                        constraints[level]);

      MGTransferGlobalCoarsening<Operator, VectorType> transfer(operators,
                                                                transfers);

      TestMultigridParameters mg_data; // TODO
      mg_solve(solver_control,
               dst,
               src,
               mg_data,
               dof_handler,
               system_matrix,
               operators,
               transfer);
    }

    static unsigned int
    generate_level_degree(const unsigned int previous_fe_degree,
                          const std::string &p_sequence)
    {
      if (p_sequence == "bisect")
        return std::max(previous_fe_degree / 2, 1u);
      else if (p_sequence == "decreasebyone")
        return std::max(previous_fe_degree - 1, 1u);
      else if (p_sequence == "gotoone")
        return 1;

      Assert(false, StandardExceptions::ExcNotImplemented());

      return 1;
    }

    template <typename VectorType,
              int dim,
              typename SystemMatrixType,
              typename LevelMatrixType,
              typename MGTransferType>
    static void mg_solve(SolverControl &                       solver_control,
                         VectorType &                          dst,
                         const VectorType &                    src,
                         const TestMultigridParameters &       mg_data,
                         const DoFHandler<dim> &               dof,
                         const SystemMatrixType &              fine_matrix,
                         const MGLevelObject<LevelMatrixType> &mg_matrices,
                         const MGTransferType &                mg_transfer)
    {
      AssertThrow(mg_data.smoother.type == "chebyshev", ExcNotImplemented());

      const unsigned int min_level = mg_matrices.min_level();
      const unsigned int max_level = mg_matrices.max_level();

      using Number                     = typename VectorType::value_type;
      using SmootherPreconditionerType = dealii::DiagonalMatrix<VectorType>;
      using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                                 VectorType,
                                                 SmootherPreconditionerType>;

      using PreconditionerType =
        dealii::PreconditionMG<dim, VectorType, MGTransferType>;

      // 1) initialize level mg_matrices
      dealii::mg::Matrix<VectorType> mg_matrix(mg_matrices);

      // 2) initialize smoothers
      dealii::MGLevelObject<typename SmootherType::AdditionalData>
        smoother_data(min_level, max_level);

      // ... initialize levels
      for (unsigned int level = min_level; level <= max_level; level++)
        {
          // ... initialize smoother
          smoother_data[level].preconditioner =
            std::make_shared<SmootherPreconditionerType>();
          mg_matrices[level].compute_inverse_diagonal(
            smoother_data[level].preconditioner->get_vector());
          smoother_data[level].smoothing_range =
            mg_data.smoother.smoothing_range;
          smoother_data[level].degree = mg_data.smoother.degree;
          smoother_data[level].eig_cg_n_iterations =
            mg_data.smoother.eig_cg_n_iterations;
        }

      // ... collect in one object
      dealii::MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>
        mg_smoother;
      mg_smoother.initialize(mg_matrices, smoother_data);

      // 3) initialize coarse-grid solver
      dealii::ReductionControl coarse_grid_solver_control(
        mg_data.coarse_solver.maxiter,
        mg_data.coarse_solver.abstol,
        mg_data.coarse_solver.reltol,
        false,
        false);
      dealii::SolverCG<VectorType> coarse_grid_solver(
        coarse_grid_solver_control);

      PreconditionIdentity precondition_identity;
      PreconditionChebyshev<LevelMatrixType,
                            VectorType,
                            dealii::DiagonalMatrix<VectorType>>
        precondition_chebyshev;

#ifdef DEAL_II_WITH_TRILINOS
      TrilinosWrappers::PreconditionAMG precondition_amg;
#endif

      std::shared_ptr<MGCoarseGridBase<VectorType>> mg_coarse;

      if (mg_data.coarse_solver.type == "cg")
        {
          // CG with identity matrix as preconditioner
          auto temp = new dealii::MGCoarseGridIterativeSolver<
            VectorType,
            dealii::SolverCG<VectorType>,
            LevelMatrixType,
            PreconditionIdentity>();

          temp->initialize(coarse_grid_solver,
                           mg_matrices[min_level],
                           precondition_identity);

          mg_coarse.reset(temp);
        }
      else if (mg_data.coarse_solver.type == "cg_with_chebyshev")
        {
          // CG with Chebyshev as preconditioner

          typename SmootherType::AdditionalData smoother_data;

          smoother_data.preconditioner =
            std::make_shared<dealii::DiagonalMatrix<VectorType>>();
          mg_matrices[min_level].compute_inverse_diagonal(
            smoother_data.preconditioner->get_vector());
          smoother_data.smoothing_range = mg_data.smoother.smoothing_range;
          smoother_data.degree          = mg_data.smoother.degree;
          smoother_data.eig_cg_n_iterations =
            mg_data.smoother.eig_cg_n_iterations;

          precondition_chebyshev.initialize(mg_matrices[min_level],
                                            smoother_data);

          auto temp = new dealii::MGCoarseGridIterativeSolver<
            VectorType,
            dealii::SolverCG<VectorType>,
            LevelMatrixType,
            decltype(precondition_chebyshev)>();

          temp->initialize(coarse_grid_solver,
                           mg_matrices[min_level],
                           precondition_chebyshev);

          mg_coarse.reset(temp);
        }
      else if (mg_data.coarse_solver.type == "cg_with_amg")
        {
#ifdef DEAL_II_WITH_TRILINOS
          TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
          amg_data.smoother_sweeps = mg_data.coarse_solver.smoother_sweeps;
          amg_data.n_cycles        = mg_data.coarse_solver.n_cycles;
          amg_data.smoother_type = mg_data.coarse_solver.smoother_type.c_str();

          // CG with AMG as preconditioner
          precondition_amg.initialize(
            mg_matrices[min_level].get_system_matrix(), amg_data);

          auto temp = new dealii::MGCoarseGridIterativeSolver<
            VectorType,
            dealii::SolverCG<VectorType>,
            LevelMatrixType,
            decltype(precondition_amg)>();

          temp->initialize(coarse_grid_solver,
                           mg_matrices[min_level],
                           precondition_amg);

          mg_coarse.reset(temp);
#else
          AssertThrow(false, ExcNotImplemented());
#endif
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }

      // 4) create multigrid object
      Multigrid<VectorType> mg(
        mg_matrix, *mg_coarse, mg_transfer, mg_smoother, mg_smoother);

      // 5) convert it to a preconditioner
      PreconditionerType preconditioner(dof, mg, mg_transfer);

      // 6) solve with CG preconditioned with multigrid
      dealii::SolverCG<VectorType> solver(solver_control);

      solver.solve(fine_matrix, dst, src, preconditioner);
    }

    static std::vector<unsigned int>
    create_p_sequence(const unsigned int degree, const std::string p_sequence)
    {
      std::vector<unsigned int> degrees;
      degrees.push_back(degree);

      unsigned int previous_fe_degree = degree;
      while (previous_fe_degree > 1)
        {
          unsigned int level_degree = [](const unsigned int previous_fe_degree,
                                         const std::string  p_sequence) {
            if (p_sequence == "bisect")
              return std::max(previous_fe_degree / 2, 1u);
            else if (p_sequence == "decreasebyone")
              return std::max(previous_fe_degree - 1, 1u);
            else if (p_sequence == "gotoone")
              return 1u;

            AssertThrow(false, ExcNotImplemented());
            return 0u;
          }(previous_fe_degree, p_sequence);

          degrees.push_back(level_degree);
          previous_fe_degree = level_degree;
        }

      std::reverse(degrees.begin(), degrees.end());

      return degrees;
    }
  };



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
      SolverAMG::solve(solver_control,
                       system_matrix,
                       locally_relevant_solution_,
                       system_rhs_);
    else if (prm.prm_solver.type == "GMG")
      SolverGMG::solve(solver_control,
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

    std::shared_ptr<LaplaceOperator<dim, double>> laplace_operator;

    if (prm.prm_operator.type == "MatrixBased")
      laplace_operator =
        std::make_shared<LaplaceOperatorMatrixBased<dim, double>>();
    else if (prm.prm_operator.type == "MatrixFree")
      laplace_operator =
        std::make_shared<LaplaceOperatorMatrixFree<dim, double>>();
    else
      Assert(false, ExcNotImplemented());

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

        laplace_operator->reinit(mapping_collection,
                                 dof_handler,
                                 quadrature_collection,
                                 constraints,
                                 system_rhs);

        solve(*laplace_operator, locally_relevant_solution, system_rhs);
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
