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

#ifndef dealii_nox_solver
#define dealii_nox_solver

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_TRILINOS

#  include <deal.II/base/exceptions.h>

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

namespace NOXWrappers
{
// Forward declaration
#  ifndef DOXYGEN
  template <typename VectorType>
  class NOXSolver;
#  endif

  struct SolverControl
  {
  public:
    DeclExceptionMsg(NoConvergence, "NOX solver did not converge.");

    enum State
    {
      iterate,
      success,
      failure
    };

    SolverControl(const unsigned int max_iter = 10,
                  const double       abs_tol  = 1.e-20,
                  const double       rel_tol  = 1.e-5);

    unsigned int
    n_newton_iterations() const;

    unsigned int
    n_linear_iterations() const;

    unsigned int
    n_residual_evaluations() const;

    unsigned int
    get_max_iter() const;

    double
    get_abs_tol() const;

    double
    get_rel_tol() const;

  private:
    const unsigned int max_iter;
    const double       abs_tol;
    const double       rel_tol;

    unsigned int newton_iterations    = 0;
    unsigned int linear_iterations    = 0;
    unsigned int residual_evaluations = 0;

    template <typename>
    friend class NOXSolver;
  };



  template <typename VectorType>
  class NOXSolver
  {
  public:
    NOXSolver(SolverControl &                             solver_control,
              const Teuchos::RCP<Teuchos::ParameterList> &parameters);

    void
    solve(VectorType &solution);

    std::function<void(VectorType &)>                     reinit_vector  = {};
    std::function<void(const VectorType &, VectorType &)> residual       = {};
    std::function<void(const VectorType &, const bool)>   setup_jacobian = {};
    std::function<unsigned int(const VectorType &, VectorType &)>
      solve_with_jacobian = {};
    std::function<SolverControl::State(const unsigned int,
                                       const double,
                                       const VectorType &,
                                       const VectorType &)>
      check_iteration_status = {};

  private:
    SolverControl &                            solver_control;
    const Teuchos::RCP<Teuchos::ParameterList> parameters;
  };
} // namespace NOXWrappers



#  ifndef DOXYGEN

namespace NOXWrappers
{
  namespace internal
  {
    template <typename VectorType>
    class Group;

    template <typename VectorType>
    class Vector : public NOX::Abstract::Vector
    {
    public:
      Vector() = default;

      Vector(VectorType &vector)
      {
        this->vector.reset(&vector, [](auto *) { /*nothing to do*/ });
      }

      NOX::Abstract::Vector &
      init(double gamma) override
      {
        *vector = gamma;
        return *this;
      }

      NOX::Abstract::Vector &
      random(bool useSeed = false, int seed = 1) override
      {
        AssertThrow(false, ExcNotImplemented());

        (void)useSeed;
        (void)seed;

        return *this;
      }

      NOX::Abstract::Vector &
      abs(const NOX::Abstract::Vector &y) override
      {
        AssertThrow(false, ExcNotImplemented());

        (void)y;

        return *this;
      }

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

      NOX::Abstract::Vector &
      reciprocal(const NOX::Abstract::Vector &y) override
      {
        AssertThrow(false, ExcNotImplemented());

        (void)y;

        return *this;
      }

      NOX::Abstract::Vector &
      scale(double gamma) override
      {
        *vector *= gamma;

        return *this;
      }

      NOX::Abstract::Vector &
      scale(const NOX::Abstract::Vector &a) override
      {
        const auto a_ = dynamic_cast<const Vector<VectorType> *>(&a);

        Assert(a_, ExcInternalError());

        vector->scale(*a_->vector);

        return *this;
      }

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

      NOX::Abstract::Vector &
      update(double                       alpha,
             const NOX::Abstract::Vector &a,
             double                       beta,
             const NOX::Abstract::Vector &b,
             double                       gamma = 0.0)
      {
        const auto a_ = dynamic_cast<const Vector<VectorType> *>(&a);
        const auto b_ = dynamic_cast<const Vector<VectorType> *>(&b);

        Assert(a_, ExcInternalError());
        Assert(b_, ExcInternalError());

        vector->operator*=(gamma);
        vector->add(alpha, *a_->vector, beta, *b_->vector);

        return *this;
      }

      Teuchos::RCP<NOX::Abstract::Vector>
      clone(NOX::CopyType copy_type) const override
      {
        auto new_vector    = Teuchos::rcp(new Vector<VectorType>());
        new_vector->vector = std::make_shared<VectorType>();
        new_vector->vector->reinit(*this->vector);

        if (copy_type == NOX::CopyType::DeepCopy)
          *new_vector->vector = *this->vector;

        return new_vector;
      }

      double
      norm(NOX::Abstract::Vector::NormType type =
             NOX::Abstract::Vector::TwoNorm) const
      {
        if (type == NOX::Abstract::Vector::NormType::TwoNorm)
          return vector->l2_norm();
        if (type == NOX::Abstract::Vector::NormType::OneNorm)
          return vector->l1_norm();
        if (type == NOX::Abstract::Vector::NormType::MaxNorm)
          return vector->linfty_norm();

        return 0.0;
      }

      double
      norm(const NOX::Abstract::Vector &weights) const override
      {
        AssertThrow(false, ExcNotImplemented());

        (void)weights;

        return 0.0;
      }

      double
      innerProduct(const NOX::Abstract::Vector &y) const override
      {
        const auto y_ = dynamic_cast<const Vector<VectorType> *>(&y);

        Assert(y_, ExcInternalError());

        return (*vector) * (*y_->vector);
      }

      NOX::size_type
      length() const override
      {
        return vector->size();
      }

      std::shared_ptr<VectorType>
      genericVector() const
      {
        return vector;
      }

    private:
      std::shared_ptr<VectorType> vector;

      friend Group<VectorType>;
    };

    template <typename VectorType>
    class Group : public NOX::Abstract::Group
    {
    public:
      Group(
        VectorType &                                                 solution,
        const std::function<void(const VectorType &, VectorType &)> &residual,
        const std::function<void(const VectorType &, const bool)>
          &setup_jacobian,
        const std::function<unsigned int(const VectorType &, VectorType &)>
          &solve_with_jacobian)
        : x(solution)
        , residual(residual)
        , setup_jacobian(setup_jacobian)
        , solve_with_jacobian(solve_with_jacobian)
        , is_valid_f(false)
        , is_valid_j(false)
      {}

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

      void
      setX(const NOX::Abstract::Vector &y) override
      {
        reset();

        x = y;
      }

      void
      computeX(const NOX::Abstract::Group & grp,
               const NOX::Abstract::Vector &d,
               double                       step)
      {
        reset();

        const auto grp_ = dynamic_cast<const Group *>(&grp);

        Assert(grp_, ExcInternalError());

        x.update(1.0, grp_->x, step, d);
      }

      NOX::Abstract::Group::ReturnType
      computeF() override
      {
        if (is_valid_f == false)
          {
            f.vector = std::make_shared<VectorType>();
            f.vector->reinit(*x.vector);

            residual(*x.vector, *f.vector);
            is_valid_f = true;
          }

        return NOX::Abstract::Group::Ok;
      }

      bool
      isF() const override
      {
        return is_valid_f;
      }

      NOX::Abstract::Group::ReturnType
      computeJacobian() override
      {
        if (is_valid_j == false)
          {
            setup_jacobian(*x.vector, true);

            is_valid_j = true;
          }

        return NOX::Abstract::Group::Ok;
      }

      bool
      isJacobian() const override
      {
        return is_valid_j;
      }


      const NOX::Abstract::Vector &
      getX() const override
      {
        return x;
      }

      const NOX::Abstract::Vector &
      getF() const override
      {
        return f;
      }

      double
      getNormF() const override
      {
        return f.norm();
      }

      const NOX::Abstract::Vector &
      getGradient() const override
      {
        return gradient;
      }

      const NOX::Abstract::Vector &
      getNewton() const override
      {
        return newton;
      }

      Teuchos::RCP<const NOX::Abstract::Vector>
      getXPtr() const override
      {
        AssertThrow(false, ExcNotImplemented());
        return {};
      }

      Teuchos::RCP<const NOX::Abstract::Vector>
      getFPtr() const override
      {
        AssertThrow(false, ExcNotImplemented());
        return {};
      }

      Teuchos::RCP<const NOX::Abstract::Vector>
      getGradientPtr() const override
      {
        AssertThrow(false, ExcNotImplemented());
        return {};
      }

      Teuchos::RCP<const NOX::Abstract::Vector>
      getNewtonPtr() const override
      {
        AssertThrow(false, ExcNotImplemented());
        return {};
      }

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

        return new_group;
      }

      NOX::Abstract::Group::ReturnType
      computeNewton(Teuchos::ParameterList &p)
      {
        (void)p; // TODO

        if (isNewton())
          return NOX::Abstract::Group::Ok;

        Assert(isF(), ExcMessage("Residual has not been computed yet!"));
        Assert(isJacobian(), ExcMessage("Jacobian has not been setup yet!"));

        if (newton.vector == nullptr)
          newton.vector = std::make_shared<VectorType>();

        newton.vector->reinit(*f.vector, false);

        solve_with_jacobian(*f.vector, *newton.vector);

        // TODO: use status of linear solver

        newton.scale(-1.0);

        return NOX::Abstract::Group::Ok;
      }

      NOX::Abstract::Group::ReturnType
      applyJacobian(const NOX::Abstract::Vector &input,
                    NOX::Abstract::Vector &      result) const override
      {
        if (!isJacobian())
          return NOX::Abstract::Group::BadDependency;

        const auto *input_  = dynamic_cast<const Vector<VectorType> *>(&input);
        const auto *result_ = dynamic_cast<const Vector<VectorType> *>(&result);

        solve_with_jacobian(*input_->vector, *result_->vector);

        return NOX::Abstract::Group::Ok;
      }

    private:
      void
      reset()
      {
        is_valid_f = false;
        is_valid_j = false;
      }

      Vector<VectorType> x, f, gradient, newton;

      std::function<void(const VectorType &, VectorType &)> residual;
      std::function<void(const VectorType &, const bool)>   setup_jacobian;
      std::function<unsigned int(const VectorType &, VectorType &)>
        solve_with_jacobian;

      bool is_valid_f;
      bool is_valid_j;
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

                const double norm_f = f_->genericVector()->l2_norm();

                state = this->check_iteration_status(step,
                                                     norm_f,
                                                     *x_->genericVector(),
                                                     *f_->genericVector());

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



  SolverControl::SolverControl(const unsigned int max_iter,
                               const double       abs_tol,
                               const double       rel_tol)
    : max_iter(max_iter)
    , abs_tol(abs_tol)
    , rel_tol(rel_tol)
  {}



  unsigned int
  SolverControl::n_newton_iterations() const
  {
    return newton_iterations;
  }



  unsigned int
  SolverControl::n_linear_iterations() const
  {
    return linear_iterations;
  }



  unsigned int
  SolverControl::n_residual_evaluations() const
  {
    return residual_evaluations;
  }



  unsigned int
  SolverControl::get_max_iter() const
  {
    return max_iter;
  }



  double
  SolverControl::get_abs_tol() const
  {
    return abs_tol;
  }



  double
  SolverControl::get_rel_tol() const
  {
    return rel_tol;
  }



  template <typename VectorType>
  NOXSolver<VectorType>::NOXSolver(
    SolverControl &                             solver_control,
    const Teuchos::RCP<Teuchos::ParameterList> &parameters)
    : solver_control(solver_control)
    , parameters(parameters)
  {}



  template <typename VectorType>
  void
  NOXSolver<VectorType>::solve(VectorType &solution)
  {
    unsigned int total_linear_iterations    = 0;
    unsigned int total_residual_evaluations = 0;

    // create group
    const auto group = Teuchos::rcp(new internal::Group<VectorType>(
      solution,
      [&](const VectorType &src, VectorType &dst) {
        total_residual_evaluations++;
        this->residual(src, dst);
      },
      [&](const VectorType &src, const bool flag) {
        this->setup_jacobian(src, flag);
      },
      [&](const VectorType &src, VectorType &dst) -> unsigned int {
        const auto linear_iterations = this->solve_with_jacobian(src, dst);
        total_linear_iterations += linear_iterations;
        return linear_iterations;
      }));

    // setup solver control
    const auto solver_control_norm_f_abs =
      Teuchos::rcp(new NOX::StatusTest::NormF(solver_control.get_abs_tol()));

    const auto solver_control_norm_f_rel = Teuchos::rcp(
      new NOX::StatusTest::RelativeNormF(solver_control.get_rel_tol()));

    const auto solver_control_max_iterations = Teuchos::rcp(
      new NOX::StatusTest::MaxIters(solver_control.get_max_iter()));

    auto check =
      Teuchos::rcp(new NOX::StatusTest::Combo(NOX::StatusTest::Combo::OR));

    if (this->check_iteration_status)
      {
        const auto info = Teuchos::rcp(
          new internal::NOXCheck(this->check_iteration_status, true));
        check->addStatusTest(info);
      }

    check->addStatusTest(solver_control_norm_f_abs);
    check->addStatusTest(solver_control_norm_f_rel);
    check->addStatusTest(solver_control_max_iterations);

    // create non-linear solver
    const auto solver = NOX::Solver::buildSolver(group, check, parameters);

    // solve
    const auto status = solver->solve();

    AssertThrow(status == NOX::StatusTest::Converged,
                SolverControl::NoConvergence());

    solver_control.newton_iterations    = solver->getNumIterations();
    solver_control.linear_iterations    = total_linear_iterations;
    solver_control.residual_evaluations = total_residual_evaluations;
  }

} // namespace NOXWrappers

#  endif

DEAL_II_NAMESPACE_CLOSE

#endif

#endif
