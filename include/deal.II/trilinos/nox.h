// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

#ifndef dealii_trilinos_nox
#define dealii_trilinos_nox

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_TRILINOS

#  include <deal.II/base/exceptions.h>

#  include <deal.II/lac/solver_control.h>

#  include <NOX_Abstract_Group.H>
#  include <NOX_Abstract_Vector.H>
#  include <NOX_Solver_Factory.H>
#  include <NOX_Solver_Generic.H>
#  include <NOX_StatusTest_Combo.H>
#  include <NOX_StatusTest_MaxIters.H>
#  include <NOX_StatusTest_NormF.H>
#  include <NOX_StatusTest_RelativeNormF.H>
#  include <Teuchos_ParameterList.hpp>

DEAL_II_NAMESPACE_OPEN

namespace TrilinosWrappers
{
// Forward declaration
#  ifndef DOXYGEN
  template <typename VectorType>
  class NOXSolver;
#  endif


  // Indicate that NOXSolver has not converged.
  DeclException0(ExcNOXNoConvergence);



  template <typename VectorType>
  class NOXSolver
  {
  public:
    struct AdditionalData
    {
    public:
      AdditionalData(const unsigned int max_iter = 10,
                     const double       abs_tol  = 1.e-20,
                     const double       rel_tol  = 1.e-5);

      const unsigned int max_iter;
      const double       abs_tol;
      const double       rel_tol;
    };

    NOXSolver(AdditionalData &                            solver_control,
              const Teuchos::RCP<Teuchos::ParameterList> &parameters);

    unsigned int
    solve(VectorType &solution);

    std::function<int(const VectorType &, VectorType &)> residual       = {};
    std::function<int(const VectorType &)>               setup_jacobian = {};
    std::function<int(const VectorType &, VectorType &)> solve_with_jacobian =
      {};
    std::function<SolverControl::State(const unsigned int,
                                       const double,
                                       const VectorType &,
                                       const VectorType &)>
      check_iteration_status = {};

  private:
    AdditionalData &                           solver_control;
    const Teuchos::RCP<Teuchos::ParameterList> parameters;
  };
} // namespace TrilinosWrappers



#  ifndef DOXYGEN

namespace TrilinosWrappers
{
  namespace internal
  {
    template <typename VectorType>
    class Group;

    /**
     * Implementation of the abstract interface
     * NOX::Abstract::Vector for deal.II vectors. For details,
     * see
     * https://docs.trilinos.org/dev/packages/nox/doc/html/classNOX_1_1Abstract_1_1Vector.html.
     */
    template <typename VectorType>
    class Vector : public NOX::Abstract::Vector
    {
    public:
      /**
       * Create empty vector.
       */
      Vector() = default;

      /**
       * Wrap an existing vector.
       */
      Vector(VectorType &vector)
      {
        this->vector.reset(&vector, [](auto *) { /*nothing to do*/ });
      }

      /**
       * Initialize every element of this vector with gamma.
       */
      NOX::Abstract::Vector &
      init(double gamma) override
      {
        *vector = gamma;
        return *this;
      }

      /**
       * Initialize each element of this vector with a random value.
       */
      NOX::Abstract::Vector &
      random(bool useSeed = false, int seed = 1) override
      {
        AssertThrow(false, ExcNotImplemented());

        (void)useSeed;
        (void)seed;

        return *this;
      }

      /**
       * Put element-wise absolute values of source vector y into this vector.
       */
      NOX::Abstract::Vector &
      abs(const NOX::Abstract::Vector &y) override
      {
        AssertThrow(false, ExcNotImplemented());

        (void)y;

        return *this;
      }

      /**
       * Copy source vector y into this vector.
       */
      NOX::Abstract::Vector &
      operator=(const NOX::Abstract::Vector &y) override
      {
        if (vector == nullptr)
          vector = std::shared_ptr<VectorType>();

        const auto y_ = dynamic_cast<const Vector<VectorType> *>(&y);

        Assert(y_, ExcInternalError());

        vector->reinit(*y_->vector);

        *vector = *y_->vector;

        return *this;
      }

      /**
       * Put element-wise reciprocal of source vector y into this vector.
       */
      NOX::Abstract::Vector &
      reciprocal(const NOX::Abstract::Vector &y) override
      {
        AssertThrow(false, ExcNotImplemented());

        (void)y;

        return *this;
      }

      /**
       * Scale each element of this vector by gamma.
       */
      NOX::Abstract::Vector &
      scale(double gamma) override
      {
        *vector *= gamma;

        return *this;
      }

      /**
       * Scale this vector element-by-element by the vector a.
       */
      NOX::Abstract::Vector &
      scale(const NOX::Abstract::Vector &a) override
      {
        const auto a_ = dynamic_cast<const Vector<VectorType> *>(&a);

        Assert(a_, ExcInternalError());

        vector->scale(*a_->vector);

        return *this;
      }

      /**
       * Compute x = (alpha * a) + (gamma * x) where x is this vector.
       */
      NOX::Abstract::Vector &
      update(double                       alpha,
             const NOX::Abstract::Vector &a,
             double                       gamma = 0.0) override
      {
        const auto a_ = dynamic_cast<const Vector<VectorType> *>(&a);

        Assert(a_, ExcInternalError());

        vector->sadd(gamma, alpha, *a_->vector);

        return *this;
      }

      /**
       * Compute x = (alpha * a) + (beta * b) + (gamma * x) where x is this
       * vector.
       */
      NOX::Abstract::Vector &
      update(double                       alpha,
             const NOX::Abstract::Vector &a,
             double                       beta,
             const NOX::Abstract::Vector &b,
             double                       gamma = 0.0) override
      {
        const auto a_ = dynamic_cast<const Vector<VectorType> *>(&a);
        const auto b_ = dynamic_cast<const Vector<VectorType> *>(&b);

        Assert(a_, ExcInternalError());
        Assert(b_, ExcInternalError());

        vector->operator*=(gamma);
        vector->add(alpha, *a_->vector, beta, *b_->vector);

        return *this;
      }

      /**
       * Create a new Vector of the same underlying type by cloning "this",
       * and return a pointer to the new vector.
       */
      Teuchos::RCP<NOX::Abstract::Vector>
      clone(NOX::CopyType copy_type) const override
      {
        auto new_vector    = Teuchos::rcp(new Vector<VectorType>());
        new_vector->vector = std::make_shared<VectorType>();
        new_vector->vector->reinit(*this->vector);

        if (copy_type == NOX::CopyType::DeepCopy)
          *new_vector->vector = *this->vector;
        else
          Assert(copy_type == NOX::CopyType::ShapeCopy, ExcInternalError());

        return new_vector;
      }

      /**
       * Norm.
       */
      double
      norm(NOX::Abstract::Vector::NormType type =
             NOX::Abstract::Vector::TwoNorm) const override
      {
        if (type == NOX::Abstract::Vector::NormType::TwoNorm)
          return vector->l2_norm();
        if (type == NOX::Abstract::Vector::NormType::OneNorm)
          return vector->l1_norm();
        if (type == NOX::Abstract::Vector::NormType::MaxNorm)
          return vector->linfty_norm();

        Assert(false, ExcInternalError());

        return 0.0;
      }

      /**
       * Weighted 2-Norm.
       */
      double
      norm(const NOX::Abstract::Vector &weights) const override
      {
        AssertThrow(false, ExcNotImplemented());

        (void)weights;

        return 0.0;
      }

      /**
       * Inner product with y.
       */
      double
      innerProduct(const NOX::Abstract::Vector &y) const override
      {
        const auto y_ = dynamic_cast<const Vector<VectorType> *>(&y);

        Assert(y_, ExcInternalError());

        return (*vector) * (*y_->vector);
      }

      /**
       * Return the length of vector.
       */
      NOX::size_type
      length() const override
      {
        return vector->size();
      }

      /**
       * Return underlying vector.
       */
      const VectorType &
      genericVector() const
      {
        AssertThrow(vector, ExcInternalError());

        return *vector;
      }

    private:
      /**
       * Underlying deal.II vector.
       */
      std::shared_ptr<VectorType> vector;

      friend Group<VectorType>;
    };

    /**
     * Implementation of the abstract interface
     * NOX::Abstract::Group for deal.II vectors and deal.II solvers. For
     * details, see
     * https://docs.trilinos.org/dev/packages/nox/doc/html/classNOX_1_1Abstract_1_1Group.html.
     */
    template <typename VectorType>
    class Group : public NOX::Abstract::Group
    {
    public:
      /**
       * Constructor. The class is intialized by the solution vector and
       * functions to compute the residual, to setup the jacobian, and
       * to solve the Jacobian.
       */
      Group(
        VectorType &                                                solution,
        const std::function<int(const VectorType &, VectorType &)> &residual,
        const std::function<int(const VectorType &)> &setup_jacobian,
        const std::function<int(const VectorType &, VectorType &)>
          &solve_with_jacobian)
        : x(solution)
        , residual(residual)
        , setup_jacobian(setup_jacobian)
        , solve_with_jacobian(solve_with_jacobian)
        , is_valid_f(false)
        , is_valid_j(false)
      {}

      /**
       * Copies the source group into this group.
       */
      NOX::Abstract::Group &
      operator=(const NOX::Abstract::Group &source) override
      {
        if (this != &source)
          {
            const auto other = dynamic_cast<const Group<VectorType> *>(&source);

            Assert(other, ExcInternalError());

            if (other->x.vector)
              {
                if (this->x.vector == nullptr)
                  this->x.vector = std::make_shared<VectorType>();

                *this->x.vector = *other->x.vector;
              }
            else
              {
                this->x.vector = {};
              }

            if (other->f.vector)
              {
                if (this->f.vector == nullptr)
                  this->f.vector = std::make_shared<VectorType>();

                *this->f.vector = *other->x.vector;
              }
            else
              {
                this->f.vector = {};
              }

            if (other->gradient.vector)
              {
                if (this->gradient.vector == nullptr)
                  this->gradient.vector = std::make_shared<VectorType>();

                *this->gradient.vector = *other->gradient.vector;
              }
            else
              {
                this->gradient.vector = {};
              }

            if (other->newton.vector)
              {
                if (this->newton.vector == nullptr)
                  this->newton.vector = std::make_shared<VectorType>();

                *this->newton.vector = *other->newton.vector;
              }
            else
              {
                this->newton.vector = {};
              }

            this->residual            = other->residual;
            this->setup_jacobian      = other->setup_jacobian;
            this->solve_with_jacobian = other->solve_with_jacobian;

            this->is_valid_f = other->is_valid_f;
            this->is_valid_j = other->is_valid_j;
          }

        return *this;
      }

      /**
       * Set the solution vector x to y.
       */
      void
      setX(const NOX::Abstract::Vector &y) override
      {
        reset();

        x = y;
      }

      /**
       * Compute x = grp.x + step * d.
       */
      void
      computeX(const NOX::Abstract::Group & grp,
               const NOX::Abstract::Vector &d,
               double                       step) override
      {
        reset();

        const auto grp_ = dynamic_cast<const Group *>(&grp);

        Assert(grp_, ExcInternalError());

        x.update(1.0, grp_->x, step, d);
      }

      /**
       * Compute and store F(x).
       */
      NOX::Abstract::Group::ReturnType
      computeF() override
      {
        if (isF() == false)
          {
            f.vector = std::make_shared<VectorType>();
            f.vector->reinit(*x.vector);

            if (residual(*x.vector, *f.vector) != 0)
              return NOX::Abstract::Group::Failed;

            is_valid_f = true;
          }

        return NOX::Abstract::Group::Ok;
      }

      /**
       * Return true if F is valid.
       */
      bool
      isF() const override
      {
        return is_valid_f;
      }

      /**
       * Compute and store Jacobian.
       */
      NOX::Abstract::Group::ReturnType
      computeJacobian() override
      {
        if (isJacobian() == false)
          {
            if (setup_jacobian(*x.vector) != 0)
              return NOX::Abstract::Group::Failed;

            is_valid_j = true;
          }

        return NOX::Abstract::Group::Ok;
      }

      /**
       * Return true if the Jacobian is valid.
       */
      bool
      isJacobian() const override
      {
        return is_valid_j;
      }

      /**
       * Return solution vector.
       */
      const NOX::Abstract::Vector &
      getX() const override
      {
        return x;
      }

      /**
       * Return F(x).
       */
      const NOX::Abstract::Vector &
      getF() const override
      {
        return f;
      }

      /**
       * Return 2-norm of F(x)
       */
      double
      getNormF() const override
      {
        return f.norm();
      }

      /**
       * Return gradient.
       */
      const NOX::Abstract::Vector &
      getGradient() const override
      {
        return gradient;
      }

      /**
       * Return Newton direction.
       */
      const NOX::Abstract::Vector &
      getNewton() const override
      {
        return newton;
      }

      /**
       * Return RCP to solution vector.
       */
      Teuchos::RCP<const NOX::Abstract::Vector>
      getXPtr() const override
      {
        AssertThrow(false, ExcNotImplemented());
        return {};
      }

      /**
       * Return RCP to F(x).
       */
      Teuchos::RCP<const NOX::Abstract::Vector>
      getFPtr() const override
      {
        AssertThrow(false, ExcNotImplemented());
        return {};
      }

      /**
       * Return RCP to gradient.
       */
      Teuchos::RCP<const NOX::Abstract::Vector>
      getGradientPtr() const override
      {
        AssertThrow(false, ExcNotImplemented());
        return {};
      }

      /**
       * Return RCP to Newton direction.
       */
      Teuchos::RCP<const NOX::Abstract::Vector>
      getNewtonPtr() const override
      {
        AssertThrow(false, ExcNotImplemented());
        return {};
      }

      /**
       * Create a new Group of the same derived type as this one by
       * cloning this one, and return a ref count pointer to the new group.
       */
      Teuchos::RCP<NOX::Abstract::Group>
      clone(NOX::CopyType copy_type) const override
      {
        auto new_group = Teuchos::rcp(new Group<VectorType>(
          *x.vector, residual, setup_jacobian, solve_with_jacobian));

        if (x.vector)
          {
            new_group->x.vector = std::make_shared<VectorType>();
            new_group->x.vector->reinit(*x.vector);
          }

        if (f.vector)
          {
            new_group->f.vector = std::make_shared<VectorType>();
            new_group->f.vector->reinit(*f.vector);
          }

        if (gradient.vector)
          {
            new_group->gradient.vector = std::make_shared<VectorType>();
            new_group->gradient.vector->reinit(*gradient.vector);
          }

        if (newton.vector)
          {
            new_group->newton.vector = std::make_shared<VectorType>();
            new_group->newton.vector->reinit(*newton.vector);
          }

        if (copy_type == NOX::CopyType::DeepCopy)
          {
            if (x.vector)
              *new_group->x.vector = *x.vector;

            if (f.vector)
              *new_group->f.vector = *f.vector;

            if (gradient.vector)
              *new_group->gradient.vector = *gradient.vector;

            if (newton.vector)
              *new_group->newton.vector = *newton.vector;

            new_group->is_valid_f = is_valid_f;
            new_group->is_valid_j = is_valid_j;
          }
        else
          Assert(copy_type == NOX::CopyType::ShapeCopy, ExcInternalError());

        return new_group;
      }

      /**
       * Compute the Newton direction, using parameters for the linear solve.
       */
      NOX::Abstract::Group::ReturnType
      computeNewton(Teuchos::ParameterList &p) override
      {
        (void)p; // TODO

        if (isNewton())
          return NOX::Abstract::Group::Ok;

        if (isF() == false || isJacobian() == false)
          return NOX::Abstract::Group::BadDependency;

        if (newton.vector == nullptr)
          newton.vector = std::make_shared<VectorType>();

        newton.vector->reinit(*f.vector, false);

        if (solve_with_jacobian(*f.vector, *newton.vector) != 0)
          return NOX::Abstract::Group::NotConverged;

        newton.scale(-1.0);

        return NOX::Abstract::Group::Ok;
      }

      /**
       * Applies Jacobian-Transpose to the given input vector and puts
       * the answer in the result.
       */
      NOX::Abstract::Group::ReturnType
      applyJacobian(const NOX::Abstract::Vector &input,
                    NOX::Abstract::Vector &      result) const override
      {
        if (!isJacobian())
          return NOX::Abstract::Group::BadDependency;

        const auto *input_  = dynamic_cast<const Vector<VectorType> *>(&input);
        const auto *result_ = dynamic_cast<const Vector<VectorType> *>(&result);

        if (solve_with_jacobian(*input_->vector, *result_->vector) != 0)
          return NOX::Abstract::Group::NotConverged;

        return NOX::Abstract::Group::Ok;
      }

    private:
      /**
       * Reset state.
       */
      void
      reset()
      {
        is_valid_f = false;
        is_valid_j = false;
      }

      // internal vectors
      Vector<VectorType> x, f, gradient, newton;

      // helper functions to compute residual, to setup jacobian, and
      // solve jacobian
      std::function<int(const VectorType &, VectorType &)> residual;
      std::function<int(const VectorType &)>               setup_jacobian;
      std::function<int(const VectorType &, VectorType &)> solve_with_jacobian;

      // internal state (are residuum and jacobian computed?)
      bool is_valid_f, is_valid_j;
    };


    template <typename VectorType>
    class NOXCheck : public NOX::StatusTest::Generic
    {
    public:
      NOXCheck(std::function<SolverControl::State(const unsigned int,
                                                  const double,
                                                  const VectorType &,
                                                  const VectorType &)>
                    check_iteration_status,
               bool as_dummy = false)
        : check_iteration_status(check_iteration_status)
        , as_dummy(as_dummy)
        , status(NOX::StatusTest::Unevaluated)
      {}

      NOX::StatusTest::StatusType
      checkStatus(const NOX::Solver::Generic &problem,
                  NOX::StatusTest::CheckType  checkType) override
      {
        if (checkType == NOX::StatusTest::None)
          {
            status = NOX::StatusTest::Unevaluated;
          }
        else
          {
            if (check_iteration_status == nullptr)
              {
                status = NOX::StatusTest::Converged;
              }
            else
              {
                const auto &x = problem.getSolutionGroup().getX();
                const auto *x_ =
                  dynamic_cast<const internal::Vector<VectorType> *>(&x);

                const auto &f = problem.getSolutionGroup().getF();
                const auto *f_ =
                  dynamic_cast<const internal::Vector<VectorType> *>(&f);

                const unsigned int step = problem.getNumIterations();

                const double norm_f = f_->genericVector().l2_norm();

                state = this->check_iteration_status(step,
                                                     norm_f,
                                                     x_->genericVector(),
                                                     f_->genericVector());

                switch (state)
                  {
                    case SolverControl::iterate:
                      status = NOX::StatusTest::Unconverged;
                      break;
                    case SolverControl::failure:
                      status = NOX::StatusTest::Failed;
                      break;
                    case SolverControl::success:
                      status = NOX::StatusTest::Converged;
                      break;
                    default:
                      AssertThrow(false, ExcNotImplemented());
                  }
              }
          }

        if (as_dummy)
          status = NOX::StatusTest::Unconverged;

        return status;
      }

      NOX::StatusTest::StatusType
      getStatus() const override
      {
        return status;
      }

      virtual std::ostream &
      print(std::ostream &stream, int indent = 0) const override
      {
        (void)indent;

        std::string state_str;
        switch (state)
          {
            case SolverControl::iterate:
              state_str = "iterate";
              break;
            case SolverControl::failure:
              state_str = "failure";
              break;
            case SolverControl::success:
              state_str = "success";
              break;
            default:
              AssertThrow(false, ExcNotImplemented());
          }

        for (int j = 0; j < indent; j++)
          stream << ' ';
        stream << status;
        stream << "check_iteration_status() = " << state_str
               << " (dummy = " << (as_dummy ? "yes" : "no") << ")";
        stream << std::endl;

        return stream;
      }

    private:
      std::function<SolverControl::State(const unsigned int,
                                         const double,
                                         const VectorType &,
                                         const VectorType &)>
        check_iteration_status = {};

      const bool as_dummy = false;

      NOX::StatusTest::StatusType status;
      SolverControl::State        state;
    };
  } // namespace internal



  template <typename VectorType>
  NOXSolver<VectorType>::AdditionalData::AdditionalData(
    const unsigned int max_iter,
    const double       abs_tol,
    const double       rel_tol)
    : max_iter(max_iter)
    , abs_tol(abs_tol)
    , rel_tol(rel_tol)
  {}



  template <typename VectorType>
  NOXSolver<VectorType>::NOXSolver(
    AdditionalData &                            solver_control,
    const Teuchos::RCP<Teuchos::ParameterList> &parameters)
    : solver_control(solver_control)
    , parameters(parameters)
  {}



  template <typename VectorType>
  unsigned int
  NOXSolver<VectorType>::solve(VectorType &solution)
  {
    // create group
    const auto group = Teuchos::rcp(new internal::Group<VectorType>(
      solution,
      [&](const VectorType &src, VectorType &dst) {
        return this->residual(src, dst);
      },
      [&](const VectorType &src) { return this->setup_jacobian(src); },
      [&](const VectorType &src, VectorType &dst) {
        return this->solve_with_jacobian(src, dst);
      }));

    // setup solver control
    auto check =
      Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));

    if (solver_control.abs_tol > 0.0)
      {
        const auto solver_control_norm_f_abs =
          Teuchos::rcp(new NOX::StatusTest::NormF(solver_control.abs_tol));
        check->addStatusTest(solver_control_norm_f_abs);
      }

    if (solver_control.rel_tol > 0.0)
      {
        const auto solver_control_norm_f_rel = Teuchos::rcp(
          new NOX::StatusTest::RelativeNormF(solver_control.rel_tol));
        check->addStatusTest(solver_control_norm_f_rel);
      }

    if (solver_control.max_iter > 0)
      {
        const auto solver_control_max_iterations =
          Teuchos::rcp(new NOX::StatusTest::MaxIters(solver_control.max_iter));
        check->addStatusTest(solver_control_max_iterations);
      }

    if (this->check_iteration_status)
      {
        const auto info = Teuchos::rcp(
          new internal::NOXCheck(this->check_iteration_status, true));
        check->addStatusTest(info);
      }

    // create non-linear solver
    const auto solver = NOX::Solver::buildSolver(group, check, parameters);

    // solve
    const auto status = solver->solve();

    AssertThrow(status == NOX::StatusTest::Converged, ExcNOXNoConvergence());

    return solver->getNumIterations();
  }

} // namespace TrilinosWrappers

#  endif

DEAL_II_NAMESPACE_CLOSE

#endif

#endif
