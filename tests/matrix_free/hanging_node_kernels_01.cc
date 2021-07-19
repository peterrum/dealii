#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"


namespace dealii::internal
{
  template <int dim, typename Number, bool is_face>
  struct FEEvaluationImplHangingNodesReference
  {
    template <int fe_degree, int n_q_points_1d>
    static bool
    run(const FEEvaluationBaseData<dim,
                                   typename Number::value_type,
                                   is_face,
                                   Number> &            fe_eval,
        const bool                                      transpose,
        const std::array<unsigned int, Number::size()> &c_mask,
        Number *                                        values)
    {
      Assert(is_face == false, ExcInternalError());

      if (dim == 2)
        {
          if (transpose)
            {
              run_2D<fe_degree, 0, true>(fe_eval, c_mask, values);
              run_2D<fe_degree, 1, true>(fe_eval, c_mask, values);
            }
          else
            {
              run_2D<fe_degree, 0, false>(fe_eval, c_mask, values);
              run_2D<fe_degree, 1, false>(fe_eval, c_mask, values);
            }
        }
      else if (dim == 3)
        {
          if (transpose)
            {
              run_3D<fe_degree, 0, true>(fe_eval, c_mask, values);
              run_3D<fe_degree, 1, true>(fe_eval, c_mask, values);
              run_3D<fe_degree, 2, true>(fe_eval, c_mask, values);
            }
          else
            {
              run_3D<fe_degree, 0, false>(fe_eval, c_mask, values);
              run_3D<fe_degree, 1, false>(fe_eval, c_mask, values);
              run_3D<fe_degree, 2, false>(fe_eval, c_mask, values);
            }
        }

      return false; // TODO
    }

  private:
    static unsigned int
    index2(unsigned int size, unsigned int i, unsigned int j)
    {
      return i + size * j;
    }

    static inline unsigned int
    index3(unsigned int size, unsigned int i, unsigned int j, unsigned int k)
    {
      return i + size * j + size * size * k;
    }

    template <int fe_degree_, unsigned int direction, bool transpose>
    static void
    run_2D(const FEEvaluationBaseData<dim,
                                      typename Number::value_type,
                                      is_face,
                                      Number> &            fe_eval,
           const std::array<unsigned int, Number::size()> &constraint_mask,
           Number *                                        values)
    {
      const auto &constraint_weights =
        fe_eval.get_shape_info().data.front().subface_interpolation_matrix;

      const unsigned int fe_degree =
        fe_degree_ != -1 ? fe_degree_ :
                           fe_eval.get_shape_info().data.front().fe_degree;

      const unsigned int n_dofs =
        Utilities::pow<unsigned int>(fe_degree + 1, 2);

      AlignedVector<Number> values_temp(n_dofs);

      for (unsigned int i = 0; i < n_dofs; ++i)
        values_temp[i] = values[i];

      for (unsigned int v = 0; v < Number::size(); ++v)
        {
          if (constraint_mask[v] == 0)
            continue;

          const unsigned int this_type =
            (direction == 0) ?
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x :
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y;

          const bool constrained_face =
            (constraint_mask[v] &
             (((direction == 0) ? dealii::internal::MatrixFreeFunctions::
                                    ConstraintTypes::face_y :
                                  0) |
              ((direction == 1) ? dealii::internal::MatrixFreeFunctions::
                                    ConstraintTypes::face_x :
                                  0)));

          const bool type = constraint_mask[v] & this_type;

          for (unsigned int x_idx = 0; x_idx < fe_degree + 1; ++x_idx)
            for (unsigned int y_idx = 0; y_idx < fe_degree + 1; ++y_idx)
              {
                const unsigned int interp_idx =
                  (direction == 0) ? x_idx : y_idx;

                typename Number::value_type t = 0.0;
                // Flag is true if dof is constrained for the given direction
                // and the given face.

                // Flag is true if for the given direction, the dof is
                // constrained with the right type and is on the correct side
                // (left (= 0) or right (= fe_degree))
                const bool constrained_dof =
                  ((direction == 0) && ((constraint_mask[v] &
                                         dealii::internal::MatrixFreeFunctions::
                                           ConstraintTypes::type_y) ?
                                          (y_idx == 0) :
                                          (y_idx == fe_degree))) ||
                  ((direction == 1) && ((constraint_mask[v] &
                                         dealii::internal::MatrixFreeFunctions::
                                           ConstraintTypes::type_x) ?
                                          (x_idx == 0) :
                                          (x_idx == fe_degree)));

                if (constrained_face && constrained_dof)
                  {
                    const unsigned int real_idx =
                      index2(fe_degree + 1, x_idx, y_idx);

                    // deallog << "dir=" << direction << " real=" << real_idx <<
                    // std::endl;


                    if (type)
                      {
                        for (unsigned int i = 0; i <= fe_degree; ++i)
                          {
                            const unsigned int real_idx =
                              (direction == 0) ?
                                index2(fe_degree + 1, i, y_idx) :
                                index2(fe_degree + 1, x_idx, i);

                            const auto w =
                              transpose ?
                                constraint_weights[i * (fe_degree + 1) +
                                                   interp_idx][v] :
                                constraint_weights[interp_idx *
                                                     (fe_degree + 1) +
                                                   i][v];
                            t += w * values_temp[real_idx][v];
                          }
                      }
                    else
                      {
                        for (unsigned int i = 0; i <= fe_degree; ++i)
                          {
                            const unsigned int real_idx =
                              (direction == 0) ?
                                index2(fe_degree + 1, i, y_idx) :
                                index2(fe_degree + 1, x_idx, i);

                            const auto w =
                              transpose ?
                                constraint_weights[(fe_degree - i) *
                                                     (fe_degree + 1) +
                                                   fe_degree - interp_idx][v] :
                                constraint_weights[(fe_degree - interp_idx) *
                                                     (fe_degree + 1) +
                                                   fe_degree - i][v];
                            t += w * values_temp[real_idx][v];
                          }
                      }

                    values[index2(fe_degree + 1, x_idx, y_idx)][v] = t;
                  }
              }
        }
    }

    template <int fe_degree_, unsigned int direction, bool transpose>
    static void
    run_3D(const FEEvaluationBaseData<dim,
                                      typename Number::value_type,
                                      is_face,
                                      Number> &            fe_eval,
           const std::array<unsigned int, Number::size()> &constraint_mask,
           Number *                                        values)
    {
      const auto &constraint_weights =
        fe_eval.get_shape_info().data.front().subface_interpolation_matrix;

      const unsigned int fe_degree =
        fe_degree_ != -1 ? fe_degree_ :
                           fe_eval.get_shape_info().data.front().fe_degree;

      const unsigned int n_dofs =
        Utilities::pow<unsigned int>(fe_degree + 1, 3);

      AlignedVector<Number> values_temp(n_dofs);

      for (unsigned int i = 0; i < n_dofs; ++i)
        values_temp[i] = values[i];

      const unsigned int this_type =
        (direction == 0) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x :
          (direction == 1) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y :
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z;
      const unsigned int face1_type =
        (direction == 0) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y :
          (direction == 1) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z :
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x;
      const unsigned int face2_type =
        (direction == 0) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z :
          (direction == 1) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x :
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y;

      // If computing in x-direction, need to match against
      // dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_y or
      // dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_z
      const unsigned int face1 =
        (direction == 0) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_y :
          (direction == 1) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_z :
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_x;
      const unsigned int face2 =
        (direction == 0) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_z :
          (direction == 1) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_x :
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_y;
      const unsigned int edge =
        (direction == 0) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_yz :
          (direction == 1) ?
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_zx :
          dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_xy;

      for (unsigned int v = 0; v < Number::size(); ++v)
        {
          if (constraint_mask[v] == 0)
            continue;

          const unsigned int constrained_face =
            constraint_mask[v] & (face1 | face2 | edge);

          const bool type = constraint_mask[v] & this_type;

          for (unsigned int x_idx = 0; x_idx < fe_degree + 1; ++x_idx)
            for (unsigned int y_idx = 0; y_idx < fe_degree + 1; ++y_idx)
              for (unsigned int z_idx = 0; z_idx < fe_degree + 1; ++z_idx)
                {
                  const unsigned int interp_idx =
                    (direction == 0) ? x_idx : (direction == 1) ? y_idx : z_idx;
                  const unsigned int face1_idx =
                    (direction == 0) ? y_idx : (direction == 1) ? z_idx : x_idx;
                  const unsigned int face2_idx =
                    (direction == 0) ? z_idx : (direction == 1) ? x_idx : y_idx;

                  typename Number::value_type t = 0;
                  const bool on_face1 = (constraint_mask[v] & face1_type) ?
                                          (face1_idx == 0) :
                                          (face1_idx == fe_degree);
                  const bool on_face2 = (constraint_mask[v] & face2_type) ?
                                          (face2_idx == 0) :
                                          (face2_idx == fe_degree);
                  const bool constrained_dof =
                    (((constraint_mask[v] & face1) && on_face1) ||
                     ((constraint_mask[v] & face2) && on_face2) ||
                     ((constraint_mask[v] & edge) && on_face1 && on_face2));

                  if (constrained_face && constrained_dof)
                    {
                      if (type)
                        {
                          for (unsigned int i = 0; i <= fe_degree; ++i)
                            {
                              const unsigned int real_idx =
                                (direction == 0) ?
                                  index3(fe_degree + 1, i, y_idx, z_idx) :
                                  (direction == 1) ?
                                  index3(fe_degree + 1, x_idx, i, z_idx) :
                                  index3(fe_degree + 1, x_idx, y_idx, i);

                              const auto w =
                                transpose ?
                                  constraint_weights[i * (fe_degree + 1) +
                                                     interp_idx][v] :
                                  constraint_weights[interp_idx *
                                                       (fe_degree + 1) +
                                                     i][v];
                              t += w * values_temp[real_idx][v];
                            }
                        }
                      else
                        {
                          for (unsigned int i = 0; i <= fe_degree; ++i)
                            {
                              const unsigned int real_idx =
                                (direction == 0) ?
                                  index3(fe_degree + 1, i, y_idx, z_idx) :
                                  (direction == 1) ?
                                  index3(fe_degree + 1, x_idx, i, z_idx) :
                                  index3(fe_degree + 1, x_idx, y_idx, i);

                              const auto w =
                                transpose ?
                                  constraint_weights[(fe_degree - i) *
                                                       (fe_degree + 1) +
                                                     fe_degree - interp_idx]
                                                    [v] :
                                  constraint_weights[(fe_degree - interp_idx) *
                                                       (fe_degree + 1) +
                                                     fe_degree - i][v];
                              t += w * values_temp[real_idx][v];
                            }
                        }

                      values[index3(fe_degree + 1, x_idx, y_idx, z_idx)][v] = t;
                    }
                }
        }
    }
  };
} // namespace dealii::internal

template <int dim>
void
test(const unsigned int degree, const unsigned int mask_value)
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, 2);
  tria.begin()->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  QGauss<dim>    quad(degree + 1);
  FE_Q<dim>      fe(degree);
  MappingQ1<dim> mapping;

  DoFHandler<dim> dof_handler;
  dof_handler.reinit(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;

  typename MatrixFree<dim, double, VectorizedArray<double>>::AdditionalData
    additional_data;
  additional_data.mapping_update_flags = update_values | update_gradients |
                                         update_JxW_values |
                                         dealii::update_quadrature_points;

  MatrixFree<dim, double, VectorizedArray<double>> matrix_free;
  matrix_free.reinit(mapping, dof_handler, constraints, quad, additional_data);

  FEEvaluation<dim, -1, 0, 1, double> eval(matrix_free);
  eval.reinit(0);

  std::array<unsigned int, VectorizedArray<double>::size()> cmask;
  std::fill(cmask.begin(), cmask.end(), 0);
  cmask[0] = mask_value;

  for (unsigned int b = 0; b < 2; ++b)
    {
      AlignedVector<VectorizedArray<double>> values1(fe.n_dofs_per_cell());
      AlignedVector<VectorizedArray<double>> values2(fe.n_dofs_per_cell());

      for (unsigned int i = 0; i < values1.size(); ++i)
        {
          values1[i][0] = i;
          values2[i][0] = i;
        }

      for (const auto i : values1)
        deallog << i[0] << " ";
      deallog << std::endl;

      internal::FEEvaluationImplHangingNodesReference<
        dim,
        VectorizedArray<double>,
        false>::template run<-1, -1>(eval, b == 1, cmask, values1.data());
      internal::FEEvaluationImplHangingNodes<
        dim,
        VectorizedArray<double>,
        false>::template run<-1, -1>(1, eval, b == 1, cmask, values2.data());

      for (const auto i : values1)
        deallog << i[0] << " ";
      deallog << std::endl;

      for (const auto i : values2)
        deallog << i[0] << " ";
      deallog << std::endl;
      deallog << std::endl;

      for (unsigned int i = 0; i < values1.size(); ++i)
        Assert(std::abs(values1[i][0] - values2[i][0]) < 1e-5,
               ExcInternalError());
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  using namespace dealii::internal;

  for (unsigned int degree = 1; degree <= 3; ++degree)
    {
      test<2>(degree, 0);
      deallog << std::endl;

      test<2>(degree,
              0 |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::
                  type_y); // face 0/0
      test<2>(degree,
              0 |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::
                  type_x); // face 0/1
      test<2>(degree,
              0 |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::
                  type_y); // face 1/0
      test<2>(degree,
              0 | dealii::internal::MatrixFreeFunctions::ConstraintTypes::
                    face_x); // face 1/1
      deallog << std::endl;

      test<2>(degree,
              0 |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_y |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::
                  type_x); // face 2/0
      test<2>(degree,
              0 |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_y |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::
                  type_y); // face 2/1
      test<2>(degree,
              0 |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_y |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::
                  type_x); // face 3/0
      test<2>(degree,
              0 | dealii::internal::MatrixFreeFunctions::ConstraintTypes::
                    face_y); // face 3/1
      deallog << std::endl;
    }

  for (unsigned int degree = 1; degree <= 3; ++degree)
    {
      // edge 2
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_yz |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_yz |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x);

      // edge 3
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_yz |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_yz |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x);

      // edge 6
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_yz |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_yz |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x);

      // edge 7
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_yz);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_yz |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x);


      // edge 0
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_zx |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_zx |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y);

      // edge 1
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_zx |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_zx |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y);

      // edge 4
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_zx |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_zx |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y);

      // edge 5
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_zx);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_zx |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y);


      // edge 8
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_xy |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_xy |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z);

      // edge 9
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_xy |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_xy |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z);

      // edge 10
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_xy |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_xy |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z);

      // edge 11
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_xy);
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::edge_xy |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z);


      // face 0
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_x |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_x);

      // face 1
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_x);

      // face 2
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_y |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_y);

      // face 3
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_y);

      // face 4
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_z |
                dealii::internal::MatrixFreeFunctions::ConstraintTypes::type_z);

      // face 5
      test<3>(degree,
              dealii::internal::MatrixFreeFunctions::ConstraintTypes::face_z);
    }
}
