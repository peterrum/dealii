
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_base.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/point_value_history.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include "../tests.h"


const unsigned int fe_degree = 1;
const unsigned int levels    = 2;
const unsigned int dim       = 2;
const unsigned int nr_faces  = GeometryInfo<dim>::faces_per_cell;

template <int dim, int fe_degree, typename Number>
void
calculate_penalty_parameter(
  AlignedVector<VectorizedArray<Number>> &array_penalty_parameter,
  MatrixFree<dim, Number> const &         data)
{
  unsigned int n_cells = data.n_macro_cells() + data.n_macro_ghost_cells();
  array_penalty_parameter.resize(n_cells);

  for (unsigned int i = 0; i < n_cells; ++i)
    {
      for (unsigned int v = 0; v < data.n_components_filled(i); ++v)
        {
          auto s  = data.get_cell_iterator(i, v)->id().to_string();
          auto ss = s.substr(s.length() - 2);
          array_penalty_parameter[i][v] = (ss.at(0) - 48) * 4 + (ss.at(1) - 48);
          // std::cout << data.get_cell_iterator(i, v)->id() <<  " -> " <<
          // (ss.at(0)-48)*4+(ss.at(1)-48) << std::endl;
        }
    }
}

template <int dim,
          int fe_degree,
          int n_q_points_1d = fe_degree + 1,
          typename number   = double>
class LaplaceOperator : public Subscriptor
{
public:
  typedef number                                value_type;
  typedef MatrixFree<dim, number>               MF;
  typedef std::pair<unsigned int, unsigned int> Range;
  typedef LaplaceOperator                       This;

  LaplaceOperator(MatrixFree<dim, number> &data)
    : data(data)
  {
    calculate_penalty_parameter<dim, fe_degree, number>(ip, this->data);
  };

  void
  apply_loop() const
  {
    int dummy;
    data.loop(&This::local_diagonal_cell,
              &This::local_diagonal_face,
              &This::local_diagonal_boundary,
              this,
              dummy,
              dummy);
  }

  void
  apply() const
  {
    int dummy;
    deallog << std::endl;
    data.cell_loop(&This::local_diagonal_by_cell, this, dummy, dummy);
    deallog << std::endl;
  }

private:
  void
  local_diagonal_by_cell(const MF &data,
                         int &,
                         const int &,
                         const Range &cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number>     phi(data);
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phif_1(data,
                                                                      true);
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phif_2(data,
                                                                      false);

    for (auto cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        // do stuff
        auto temp = phi.read_cell_data(ip);
        deallog << "c   ";
        for (int i = 0; i < 4; i++)
          deallog << std::setw(3) << (int)temp[i] << " ";
        deallog << std::endl;

        for (unsigned int face = 0; face < nr_faces; ++face)
          {
            phif_1.reinit(cell, face);
            phif_2.reinit(cell, face);
            {
              auto temp = phif_1.read_cell_data(ip);
              deallog << "m-" << face << " ";
              for (int i = 0; i < 4; i++)
                deallog << std::setw(3) << (int)temp[i] << " ";
              deallog << std::endl;
            }
            {
              auto temp = phif_2.read_cell_data(ip);
              deallog << "p-" << face << " ";
              for (int i = 0; i < 4; i++)
                deallog << std::setw(3) << (int)temp[i] << " ";
              deallog << std::endl;
            }
          }
        deallog << std::endl;
      }
  }

  void
  local_diagonal_cell(const MF &data,
                      int &,
                      const int &,
                      const Range &cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi(data);

    for (auto cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        deallog << "c   ";
        auto temp = phi.read_cell_data(ip);
        for (int i = 0; i < 4; i++)
          deallog << std::setw(3) << (int)temp[i] << " ";
        deallog << std::endl;
      }
  }

  void
  local_diagonal_face(const MF &data,
                      int &,
                      const int &,
                      const Range &cell_range) const
  {
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi(data);

    for (auto cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        auto temp = phi.read_cell_data(ip);
        deallog << "f   ";
        for (int i = 0; i < 4; i++)
          deallog << std::setw(3) << (int)temp[i] << " ";
        deallog << std::endl;
      }
  }

  void
  local_diagonal_boundary(const MF &, int &, const int &, const Range &) const
  {}

  MatrixFree<dim, number> &              data;
  AlignedVector<VectorizedArray<number>> ip;
};

void
test()
{
  // create triangulation
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria, -1, +1);
  tria.refine_global(levels);

  // create fe
  FE_DGQ<dim> fe(fe_degree);

  // create dof_handler
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);

  // create constraint matrix
  ConstraintMatrix constraints;
  constraints.close();

  // create matrix_free
  MatrixFree<dim> mf_data;

  // ... setup additional data
  typename MatrixFree<dim>::AdditionalData data;
  data.tasks_parallel_scheme         = MatrixFree<dim>::AdditionalData::none;
  data.build_face_info               = true;
  data.hold_all_faces_to_owned_cells = true;


  data.mapping_update_flags_faces_by_cells =
    (update_JxW_values | update_normal_vectors | update_quadrature_points |
     update_values);

  // ... create list for the category of each cell
  data.cell_vectorization_category.resize(tria.n_active_cells());

  // ... setup scaling factor
  std::vector<unsigned int> factors(dim * 2);


  std::map<unsigned int, unsigned int> bid_map;
  for (unsigned int i = 0; i < tria.get_boundary_ids().size(); i++)
    bid_map[tria.get_boundary_ids()[i]] = i + 1;

  {
    unsigned int bids   = tria.get_boundary_ids().size() + 1;
    int          offset = 1;
    for (unsigned int i = 0; i < dim * 2; i++, offset = offset * bids)
      factors[i] = offset;
  }

  for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
      // accumulator for category of this cell: start with 0
      unsigned int c_num = 0;
      if (cell->is_locally_owned())
        // loop over all faces
        for (unsigned int i = 0; i < dim * 2; i++)
          {
            auto &face = *cell->face(i);
            if (face.at_boundary())
              // and update accumulator if on boundary
              c_num += factors[i] * bid_map[face.boundary_id()];
          }
      // save the category of this cell
      data.cell_vectorization_category[cell->active_cell_index()] = c_num;
    }

  // ... finalize setup of matrix_free
  data.cell_vectorization_categories_strict = true;
  mf_data.reinit(dof, constraints, QGauss<1>(2), data);

  // print results

  LaplaceOperator<dim, fe_degree> lo(mf_data);
  lo.apply();
  lo.apply_loop();
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);
  MPILogInitAll                    log;
  test();
}
