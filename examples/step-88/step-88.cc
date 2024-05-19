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
 * Authors: Marco Feder, SISSA, 2023
 *          Peter Munch, University of Augsburg/Uppsala University, 2023
 */

// @sect3{Include files}

// The first include files have all been treated in previous examples.

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

// The file most relevant for this tutorial is the one that
// contain the class MGTwoLevelTransferNonNested.
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

// We pack everything that is specific for this program into a namespace
// of its own.

namespace Step88
{
  using namespace dealii;

  // @sect3{Parameters}
  // This class contains relevant parameters.
  struct Parameters
  {
    // This parameter specifies the mesh type. Options are:
    // "hyper_cube" for creating a mesh with
    // GridGenerator::hyper_cube(), "hyper_cube_with_simplices",
    // for creating a mesh with
    // GridGenerator::subdivided_hyper_cube_with_simplices() with a single
    // subdivision, "mesh_file" for reading a sequence of external meshes.
    std::string mesh_type;

    // In the case of "mesh_file", this parameter specifies
    // the name of the mesh file on each level.
    std::string mesh_file_format;

    // Number of dimensions.
    unsigned int dim;

    // Number of global refinements. In the case of
    // "hyper_cube" and "hyper_cube_with_simplices", the mesh
    // is refined the specified amount. In the case of "mesh_file",
    // only the first external files are read.
    unsigned int n_global_refinements;

    // Polynomial degree.
    unsigned int fe_degree;

    // Maximal number of iterations of the solver.
    unsigned int solver_max_iterations;

    // Absolute tolerance of the solver.
    double solver_abs_tolerance;

    // Relative tolerance of the solver.
    double solver_rel_tolerance;

    // Smoothing gange of the smoothers of the multigrid algorithm.
    unsigned int mg_smoothing_range;

    // Smoothing degree of the smoothers of the multigrid algorithm.
    unsigned int mg_smoother_degree;

    // Number of iterations to determine the eigenvalues on the levels,
    // needed for setting up the smoothers of the multigrid algorithm.
    unsigned int mg_smoother_eig_cg_n_iterations;

    // Specify whether the nested or non-nested global-coarsening algorithm
    // should be used. Note: in the case of "mesh_file", only
    // a non-nested algorithm can be selected.
    bool mg_non_nested;

    // Constructor, which sets the default values of the parameters.
    Parameters();

    void parse(const std::string file_name);

    void print();

    std::string get_mesh_file_name(const unsigned int level) const;

  private:
    void add_parameters(ParameterHandler &prm);
  };



  Parameters::Parameters()
    : mesh_type("hyper_cube")
    , mesh_file_format("")
    , dim(2)
    , n_global_refinements(3)
    , fe_degree(2)
    , solver_max_iterations(100)
    , solver_abs_tolerance(1e-20)
    , solver_rel_tolerance(1e-4)
    , mg_smoothing_range(20)
    , mg_smoother_degree(5)
    , mg_smoother_eig_cg_n_iterations(20)
    , mg_non_nested(true)
  {}



  // Parse a file.
  void Parameters::parse(const std::string file_name)
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);

    std::ifstream file;
    file.open(file_name);
    prm.parse_input_from_json(file, true);
  }



  // Print parameters to the screen.
  void Parameters::print()
  {
    dealii::ParameterHandler prm;
    add_parameters(prm);
    prm.print_parameters(std::cout,
                         dealii::ParameterHandler::OutputStyle::ShortJSON);
  }



  // Get name of the mesh on the given level.
  std::string Parameters::get_mesh_file_name(const unsigned int level) const
  {
    char buffer[100];

    std::snprintf(buffer, 100, mesh_file_format.c_str(), level);

    return {buffer};
  }



  // Add parameters used for parse() and print().
  void Parameters::add_parameters(ParameterHandler &prm)
  {
    prm.add_parameter("MeshType",
                      mesh_type,
                      "",
                      Patterns::Selection(
                        "hyper_cube|hyper_cube_with_simplices|mesh_file"));
    prm.add_parameter("MeshFileFormat", mesh_file_format);
    prm.add_parameter("Dimension", dim);
    prm.add_parameter("NGlobalRefinements", n_global_refinements);
    prm.add_parameter("Degree", fe_degree);

    prm.add_parameter("SolverMaxIterations", solver_max_iterations);
    prm.add_parameter("SolverAbsTolerance", solver_abs_tolerance);
    prm.add_parameter("SolverRelTolerance", solver_rel_tolerance);

    prm.add_parameter("MGSmoothingScheme", mg_smoothing_range);
    prm.add_parameter("MGSmootherDegree", mg_smoother_degree);
    prm.add_parameter("MGSmootherEigNIterations",
                      mg_smoother_eig_cg_n_iterations);
    prm.add_parameter("MGNonNested", mg_non_nested);
  }



  // @sect3{Laplace operator}
  // A basic matrix-free implementation of the Laplace operator. For
  // more details, see step-75.
  template <int dim, typename number>
  class LaplaceOperator : public Subscriptor
  {
  public:
    using VectorType       = LinearAlgebra::distributed::Vector<number>;
    using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, number>;

    void reinit(const Mapping<dim>              &mapping,
                const DoFHandler<dim>           &dof_handler,
                const AffineConstraints<number> &constraints,
                const Quadrature<dim>           &quad);

    types::global_dof_index m() const;

    number el(unsigned int, unsigned int) const;

    void initialize_dof_vector(VectorType &vec) const;

    void vmult(VectorType &dst, const VectorType &src) const;

    void Tvmult(VectorType &dst, const VectorType &src) const;

    void compute_inverse_diagonal(VectorType &diagonal) const;

    const TrilinosWrappers::SparseMatrix &get_system_matrix() const;

    std::shared_ptr<const Utilities::MPI::Partitioner> get_partitioner() const;

  private:
    void do_cell_integral_local(FECellIntegrator &integrator) const;

    void do_cell_integral_global(FECellIntegrator &integrator,
                                 VectorType       &dst,
                                 const VectorType &src) const;


    void do_cell_integral_range(
      const MatrixFree<dim, number>               &matrix_free,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &range) const;

    MatrixFree<dim, number>                matrix_free;
    mutable TrilinosWrappers::SparseMatrix system_matrix;
  };



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::reinit(
    const Mapping<dim>              &mapping,
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    const Quadrature<dim>           &quad)
  {
    this->system_matrix.clear();

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_gradients;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);
  }


  template <int dim, typename number>
  types::global_dof_index LaplaceOperator<dim, number>::m() const
  {
    return matrix_free.get_dof_handler().n_dofs();
  }



  template <int dim, typename number>
  number LaplaceOperator<dim, number>::el(unsigned int, unsigned int) const
  {
    Assert(false, ExcNotImplemented());
    return 0;
  }



  template <int dim, typename number>
  void
  LaplaceOperator<dim, number>::initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::vmult(VectorType       &dst,
                                           const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &LaplaceOperator::do_cell_integral_range, this, dst, src, true);
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::Tvmult(VectorType       &dst,
                                            const VectorType &src) const
  {
    this->vmult(dst, src);
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::compute_inverse_diagonal(
    VectorType &diagonal) const
  {
    this->matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(matrix_free,
                                      diagonal,
                                      &LaplaceOperator::do_cell_integral_local,
                                      this);

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }



  template <int dim, typename number>
  const TrilinosWrappers::SparseMatrix &
  LaplaceOperator<dim, number>::get_system_matrix() const
  {
    if (system_matrix.m() == 0 && system_matrix.n() == 0)
      {
        const auto &dof_handler = this->matrix_free.get_dof_handler();

        TrilinosWrappers::SparsityPattern dsp(
          dof_handler.locally_owned_dofs(),
          dof_handler.get_triangulation().get_communicator());

        DoFTools::make_sparsity_pattern(dof_handler,
                                        dsp,
                                        matrix_free.get_affine_constraints());

        dsp.compress();
        system_matrix.reinit(dsp);

        MatrixFreeTools::compute_matrix(
          matrix_free,
          matrix_free.get_affine_constraints(),
          system_matrix,
          &LaplaceOperator::do_cell_integral_local,
          this);
      }

    return this->system_matrix;
  }



  template <int dim, typename number>
  std::shared_ptr<const Utilities::MPI::Partitioner>
  LaplaceOperator<dim, number>::get_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::do_cell_integral_local(
    FECellIntegrator &integrator) const
  {
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate(EvaluationFlags::gradients);
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::do_cell_integral_range(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);

    for (unsigned cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);

        do_cell_integral_global(integrator, dst, src);
      }
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::do_cell_integral_global(
    FECellIntegrator &integrator,
    VectorType       &dst,
    const VectorType &src) const
  {
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
  }

  // @sect3{Laplace problem}
  // We then define the main class that solves the Laplace problem.
  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem(const Parameters &params);

    void run();

  private:
    bool create_grids();

    void setup_system();

    void solve();

    void output_results();

    using Number     = double;
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    const MPI_Comm     comm;
    const Parameters   params;
    const unsigned int min_level;
    const unsigned int max_level;
    ConditionalOStream pcout;

    std::vector<std::shared_ptr<Triangulation<dim>>> triangulations;

    std::unique_ptr<FiniteElement<dim>> fe;
    std::unique_ptr<Mapping<dim>>       mapping;
    Quadrature<dim>                     quadrature;

    MGLevelObject<DoFHandler<dim>>              dof_handlers;
    MGLevelObject<AffineConstraints<Number>>    constraints;
    MGLevelObject<LaplaceOperator<dim, Number>> operators;

    VectorType rhs;
    VectorType solution;
  };



  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem(const Parameters &params)
    : comm(MPI_COMM_WORLD)
    , params(params)
    , min_level(0)
    , max_level(params.n_global_refinements)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0))
  {}



  // This function that creates the grid on each level, sets up
  // the system, solves the resulting problem, and finally outputs
  // the result.
  template <int dim>
  void LaplaceProblem<dim>::run()
  {
    const bool nested_mesh = create_grids();
    AssertThrow(nested_mesh || params.mg_non_nested, ExcNotImplemented());

    setup_system();

    solve();

    output_results();
  }



  // This function creates the mesh sequences.
  template <int dim>
  bool LaplaceProblem<dim>::create_grids()
  {
    pcout << "Create mesh: " << std::endl;

    // create hyper-cube mesh sequence
    if (params.mesh_type == "hyper_cube")
      {
        pcout << " - hyper_cube" << std::endl;

        for (unsigned int l = min_level; l <= max_level; ++l)
          {
            auto triangulation =
              std::make_shared<parallel::shared::Triangulation<dim>>(comm);
            GridGenerator::hyper_cube(*triangulation);
            triangulation->refine_global(l);
            triangulations.push_back(triangulation);
          }

        return true; // nested mesh
      }
    else
      // create hyper-cube mesh sequence with simplices
      if (params.mesh_type == "hyper_cube_with_simplices")
        {
          pcout << " - hyper_cube_with_simplices" << std::endl;

          Triangulation<dim> dummy;
          GridGenerator::hyper_cube(dummy);

          for (unsigned int l = min_level; l <= max_level; ++l)
            {
              auto triangulation =
                std::make_shared<parallel::shared::Triangulation<dim>>(comm);
              GridGenerator::convert_hypercube_to_simplex_mesh(dummy,
                                                               *triangulation);
              triangulation->refine_global(l);
              triangulations.push_back(triangulation);
            }

          return true; // nested mesh
        }
      else
        // read mesh for each level
        if (params.mesh_type == "mesh_file")
          {
            for (unsigned int l = min_level; l <= max_level; ++l)
              {
                auto triangulation =
                  std::make_shared<parallel::distributed::Triangulation<dim>>(
                    comm);

                GridIn<dim> grid_in(*triangulation);

                const auto mesh_file_name = params.get_mesh_file_name(l);
                pcout << " - read " << mesh_file_name << std::endl;
                grid_in.read(mesh_file_name, GridIn<dim>::abaqus);

                triangulations.push_back(triangulation);
              }

            pcout << std::endl;

            return false; // non-nested mesh
          }

    AssertThrow(false, ExcNotImplemented());

    return true;
  }



  // This function sets up the system including the ones on the
  // multigrid levels.
  template <int dim>
  void LaplaceProblem<dim>::setup_system()
  {
    pcout << "Set up system:" << std::endl;

    // Create finite element, mapping, and quadrature depending on
    // the mesh type. In the case of hyper-cube meshes, FE_Q,
    // MappingQ, and QGauss are used; in the case of simplex meshes,
    // FE_SimplexP, MappingFE, and QGaussSimplex are used.
    if (triangulations.back()->all_reference_cells_are_hyper_cube())
      fe = std::make_unique<FE_Q<dim>>(params.fe_degree);
    else if (triangulations.back()->all_reference_cells_are_simplex())
      fe = std::make_unique<FE_SimplexP<dim>>(params.fe_degree);
    else
      AssertThrow(false, ExcNotImplemented());

    const auto reference_cell = triangulations.back()->get_reference_cells()[0];

    mapping = reference_cell.template get_default_mapping<dim>(1);

    quadrature = reference_cell.template get_gauss_type_quadrature<dim>(
      params.fe_degree + 1);

    // Initialize system of levels. This includes the setup
    // of the corresponding DoFHandler, AffineConstraints, and
    // LaplaceOperator objects.
    dof_handlers.resize(min_level, max_level);
    constraints.resize(min_level, max_level);
    operators.resize(min_level, max_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        dof_handlers[l].reinit(*triangulations[l]);
        dof_handlers[l].distribute_dofs(*fe);

        pcout << " - number of DoFs: " << dof_handlers[l].n_dofs() << std::endl;

        constraints[l].reinit(
          DoFTools::extract_locally_relevant_dofs(dof_handlers[l]));
        DoFTools::make_zero_boundary_constraints(dof_handlers[l],
                                                 constraints[l]);
        constraints[l].close();

        operators[l].reinit(*mapping,
                            dof_handlers[l],
                            constraints[l],
                            quadrature);
      }

    // Intialize vectors and set up right-hand-side vector.
    operators.back().initialize_dof_vector(solution);
    operators.back().initialize_dof_vector(rhs);

    VectorTools::create_right_hand_side(*mapping,
                                        dof_handlers.back(),
                                        quadrature,
                                        Functions::ConstantFunction<dim>(1.0),
                                        rhs,
                                        constraints.back());

    pcout << std::endl;
  }



  // Solve the laplace problem with multigrid.
  template <int dim>
  void LaplaceProblem<dim>::solve()
  {
    using LevelMatrixType            = LaplaceOperator<dim, Number>;
    using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
    using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                               VectorType,
                                               SmootherPreconditionerType>;
    using MGTransferType             = MGTransferMF<dim, Number>;
    using PreconditionerType = PreconditionMG<dim, VectorType, MGTransferType>;

    // Initialize multigrid transfer operator. For this purpose, we initialize
    // two-level transfer operators between each neighboring level. In
    // the non-nested case, we use MGTwoLevelTransferNonNested and, in the
    // case of the nested case, we use MGTwoLevelTransfer. Both classes
    // inherit from the class MGTwoLevelTransferBase.
    MGLevelObject<std::shared_ptr<const MGTwoLevelTransferBase<VectorType>>>
      transfers(min_level, max_level);

    for (unsigned int l = min_level; l < max_level; ++l)
      if (params.mg_non_nested) // non-nested case
        {
          auto transfer =
            std::make_shared<MGTwoLevelTransferNonNested<dim, VectorType>>();

          transfer->reinit(dof_handlers[l + 1],
                           dof_handlers[l],
                           *mapping,
                           *mapping,
                           constraints[l + 1],
                           constraints[l]);

          transfers[l + 1] = transfer;
        }
      else // nested case
        {
          auto transfer =
            std::make_shared<MGTwoLevelTransfer<dim, VectorType>>();

          transfer->reinit_geometric_transfer(dof_handlers[l + 1],
                                              dof_handlers[l],
                                              constraints[l + 1],
                                              constraints[l]);

          transfers[l + 1] = transfer;
        }

    std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
      partitioners;

    for (unsigned int l = min_level; l <= max_level; ++l)
      partitioners.push_back(operators[l].get_partitioner());

    MGTransferMF<dim, Number> mg_transfer;
    mg_transfer.initialize_two_level_transfers(transfers);
    mg_transfer.build(partitioners);

    // Intialize other ingredients of the multigrid operator
    // (smoother, coarse-grid solver) and put them together
    // to a multigrid-operator.
    mg::Matrix<VectorType> mg_matrix(operators);

    MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
      min_level, max_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        smoother_data[l].preconditioner =
          std::make_shared<SmootherPreconditionerType>();
        operators[l].compute_inverse_diagonal(
          smoother_data[l].preconditioner->get_vector());
        smoother_data[l].smoothing_range = params.mg_smoothing_range;
        smoother_data[l].degree          = params.mg_smoother_degree;
        smoother_data[l].eig_cg_n_iterations =
          params.mg_smoother_eig_cg_n_iterations;
      }

    MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>
      mg_smoother;
    mg_smoother.initialize(operators, smoother_data);

    TrilinosWrappers::PreconditionAMG precondition_amg;
    precondition_amg.initialize(operators[min_level].get_system_matrix());

    MGCoarseGridApplyPreconditioner<VectorType,
                                    TrilinosWrappers::PreconditionAMG>
      mg_coarse(precondition_amg);

    Multigrid<VectorType> mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);

    PreconditionerType preconditioner(dof_handlers.back(), mg, mg_transfer);

    // Finally, solve the system.
    ReductionControl solver_control(params.solver_max_iterations,
                                    params.solver_abs_tolerance,
                                    params.solver_rel_tolerance);

    SolverCG<VectorType>(solver_control)
      .solve(operators.back(), solution, rhs, preconditioner);

    pcout << "Solved in " << solver_control.last_step() << " steps."
          << std::endl;
  }



  // Output results. Here, we output the solution on the
  // finest mesh and we output the mesh on each level.
  template <int dim>
  void LaplaceProblem<dim>::output_results()
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handlers.back());
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(*mapping);

    data_out.write_vtu_in_parallel("solution.vtu", comm);


    for (unsigned int l = 0; l < triangulations.size(); ++l)
      {
        DataOut<dim> data_out;
        data_out.attach_triangulation(*triangulations[l]);
        data_out.build_patches();

        data_out.write_vtu_in_parallel("grid_" + std::to_string(l) + ".vtu",
                                       comm);
      }
  }

} // namespace Step88

// @sect3{Driver}
//
// Finally, the driver of the program reads the parameters
// and solves the Laplace problem.
int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step88;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      Parameters prm;
      if (argc > 1)
        prm.parse(std::string(argv[1]));
      else
        {
          prm.print();
          return 0;
        }

      if (prm.dim == 2)
        {
          LaplaceProblem<2> laplace_problem(prm);
          laplace_problem.run();
        }
      else if (prm.dim == 3)
        {
          LaplaceProblem<3> laplace_problem(prm);
          laplace_problem.run();
        }
      else
        {
          AssertThrow(false, ExcNotImplemented());
        }
    }
  catch (const std::exception &exc)
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
