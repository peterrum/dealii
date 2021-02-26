// --------------------------------------------------------------------------
//
// Copyright (C) 2021 by the adaflo authors
//
// This file is part of the adaflo library.
//
// The adaflo library is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.  The full text of the
// license can be found in the file LICENSE at the top level of the adaflo
// distribution.
//
// --------------------------------------------------------------------------

#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_point_evaluation.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_q_iso_q1.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/la_parallel_block_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;


using VectorType = LinearAlgebra::distributed::Vector<double>;

namespace dealii
{
  namespace VectorTools
  {
    template <int dim, int spacedim, typename VectorType>
    void
    get_position_vector(const DoFHandler<dim, spacedim> &dof_handler_dim,
                        VectorType &                  euler_coordinates_vector,
                        const Mapping<dim, spacedim> &mapping)
    {
      FEValues<dim, spacedim> fe_eval(
        mapping,
        dof_handler_dim.get_fe(),
        Quadrature<dim>(dof_handler_dim.get_fe().get_unit_support_points()),
        update_quadrature_points);

      Vector<double> temp;

      for (const auto &cell : dof_handler_dim.active_cell_iterators())
        {
          fe_eval.reinit(cell);

          temp.reinit(fe_eval.dofs_per_cell);

          for (const auto q : fe_eval.quadrature_point_indices())
            {
              const auto point = fe_eval.quadrature_point(q);

              const unsigned int comp =
                dof_handler_dim.get_fe().system_to_component_index(q).first;

              temp[q] = point[comp];
            }

          cell->set_dof_values(temp, euler_coordinates_vector);
        }
    }
  } // namespace VectorTools
} // namespace dealii

template <int dim, int spacedim, typename VectorType>
void
compute_normal(const Mapping<dim, spacedim> &   mapping,
               const DoFHandler<dim, spacedim> &dof_handler_dim,
               VectorType &                     normal_vector)
{
  FEValues<dim, spacedim> fe_eval_dim(
    mapping,
    dof_handler_dim.get_fe(),
    dof_handler_dim.get_fe().get_unit_support_points(),
    update_normal_vectors | update_gradients);

  Vector<double> normal_temp;

  for (const auto &cell : dof_handler_dim.active_cell_iterators())
    {
      fe_eval_dim.reinit(cell);

      normal_temp.reinit(fe_eval_dim.dofs_per_cell);
      normal_temp = 0.0;

      for (const auto q : fe_eval_dim.quadrature_point_indices())
        {
          const auto normal = fe_eval_dim.normal_vector(q);

          const unsigned int comp =
            dof_handler_dim.get_fe().system_to_component_index(q).first;

          normal_temp[q] = normal[comp];
        }

      cell->set_dof_values(normal_temp, normal_vector);
    }
}



template <int dim, int spacedim, typename VectorType>
void
compute_curvature(const Mapping<dim, spacedim> &   mapping,
                  const DoFHandler<dim, spacedim> &dof_handler_dim,
                  const DoFHandler<dim, spacedim> &dof_handler,
                  const Quadrature<dim>            quadrature,
                  const VectorType &               normal_vector,
                  VectorType &                     curvature_vector)
{
  FEValues<dim, spacedim> fe_eval(mapping,
                                  dof_handler.get_fe(),
                                  quadrature,
                                  update_gradients);
  FEValues<dim, spacedim> fe_eval_dim(mapping,
                                      dof_handler_dim.get_fe(),
                                      quadrature,
                                      update_gradients);

  Vector<double> curvature_temp;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
        &dof_handler_dim.get_triangulation(),
        cell->level(),
        cell->index(),
        &dof_handler_dim);

      fe_eval.reinit(cell);
      fe_eval_dim.reinit(dof_cell_dim);

      curvature_temp.reinit(quadrature.size());

      std::vector<std::vector<Tensor<1, spacedim, double>>> normal_gradients(
        quadrature.size(), std::vector<Tensor<1, spacedim, double>>(spacedim));

      fe_eval_dim.get_function_gradients(normal_vector, normal_gradients);

      for (const auto q : fe_eval_dim.quadrature_point_indices())
        {
          double curvature = 0.0;

          for (unsigned c = 0; c < spacedim; ++c)
            curvature += normal_gradients[q][c][c];

          curvature_temp[q] = curvature;
        }

      cell->set_dof_values(curvature_temp, curvature_vector);
    }
}



template <int dim, int spacedim, typename VectorType>
void
compute_force_vector_sharp_interface(
  const Mapping<dim, spacedim> &   surface_mapping,
  const DoFHandler<dim, spacedim> &surface_dofhandler,
  const DoFHandler<dim, spacedim> &surface_dofhandler_dim,
  const Quadrature<dim> &          surface_quadrature,
  const Mapping<spacedim> &        mapping,
  const DoFHandler<spacedim> &     dof_handler,
  const double                     surface_tension,
  const VectorType &               normal_vector,
  const VectorType &               curvature_vector,
  VectorType &                     force_vector)
{
  using T = Tensor<1, spacedim, double>;

  const auto integration_points = [&]() {
    std::vector<Point<spacedim>> integration_points;

    FEValues<dim, spacedim> fe_eval(surface_mapping,
                                    surface_dofhandler.get_fe(),
                                    surface_quadrature,
                                    update_values | update_quadrature_points |
                                      update_JxW_values);

    for (const auto &cell :
         surface_dofhandler.get_triangulation().active_cell_iterators())
      {
        fe_eval.reinit(cell);

        for (const auto q : fe_eval.quadrature_point_indices())
          integration_points.push_back(fe_eval.quadrature_point(q));
      }

    return integration_points;
  }();

  Utilities::MPI::RemotePointEvaluation<spacedim, spacedim> eval;
  eval.reinit(integration_points, dof_handler.get_triangulation(), mapping);

  const auto integration_values = [&]() {
    std::vector<T> integration_values;

    FEValues<dim, spacedim> fe_eval(surface_mapping,
                                    surface_dofhandler.get_fe(),
                                    surface_quadrature,
                                    update_values | update_quadrature_points |
                                      update_JxW_values);
    FEValues<dim, spacedim> fe_eval_dim(surface_mapping,
                                        surface_dofhandler_dim.get_fe(),
                                        surface_quadrature,
                                        update_values);

    const auto &tria_surface = surface_dofhandler.get_triangulation();

    for (const auto &cell : tria_surface.active_cell_iterators())
      {
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell(
          &tria_surface, cell->level(), cell->index(), &surface_dofhandler);
        TriaIterator<DoFCellAccessor<dim, spacedim, false>> dof_cell_dim(
          &tria_surface, cell->level(), cell->index(), &surface_dofhandler_dim);

        fe_eval.reinit(dof_cell);
        fe_eval_dim.reinit(dof_cell_dim);

        std::vector<double>         curvature_values(fe_eval.dofs_per_cell);
        std::vector<Vector<double>> normal_values(fe_eval.dofs_per_cell,
                                                  Vector<double>(spacedim));

        fe_eval.get_function_values(curvature_vector, curvature_values);
        fe_eval_dim.get_function_values(normal_vector, normal_values);

        for (const auto q : fe_eval_dim.quadrature_point_indices())
          {
            T result;
            for (unsigned int i = 0; i < spacedim; ++i)
              result[i] = -curvature_values[q] * normal_values[q][i] *
                          fe_eval.JxW(q) * surface_tension;

            integration_values.push_back(result);
          }
      }

    return integration_values;
  }();

  const auto fu = [&](const auto &values, const auto &quadrature_points) {
    AffineConstraints<double> constraints; // TODO: use the right ones

    FEPointEvaluation<spacedim, spacedim> phi_normal_force(
      mapping, dof_handler.get_fe());

    std::vector<double>                  buffer;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<T>                       force_JxW;

    unsigned int i = 0;

    for (const auto &cells_and_n : std::get<0>(quadrature_points))
      {
        typename DoFHandler<spacedim>::active_cell_iterator cell = {
          &eval.get_triangulation(),
          cells_and_n.first.first,
          cells_and_n.first.second,
          &dof_handler};

        const unsigned int n_dofs_per_cell = cell->get_fe().n_dofs_per_cell();

        local_dof_indices.resize(n_dofs_per_cell);
        buffer.resize(n_dofs_per_cell);

        cell->get_dof_indices(local_dof_indices);

        const ArrayView<const Point<spacedim>> unit_points(
          std::get<1>(quadrature_points).data() + i, cells_and_n.second);

        force_JxW.resize(unit_points.size());
        for (unsigned int q = 0; q < unit_points.size(); ++q, ++i)
          force_JxW[q] = values[std::get<2>(quadrature_points)[i]];

        for (unsigned int q = 0; q < unit_points.size(); ++q)
          phi_normal_force.submit_value(force_JxW[q], q);

        phi_normal_force.integrate(cell,
                                   unit_points,
                                   buffer,
                                   EvaluationFlags::values);

        constraints.distribute_local_to_global(buffer,
                                               local_dof_indices,
                                               force_vector);
      }
  };

  std::vector<T> buffer;

  eval.template process_and_evaluate<T>(integration_values, buffer, fu);
}



template <int dim>
void
test()
{
  const unsigned int spacedim = dim + 1;

  const unsigned int fe_degree      = 3;
  const unsigned int mapping_degree = fe_degree;
  const unsigned int n_refinements  = 5;

  Triangulation<dim, spacedim> tria;
#if false
  GridGenerator::hyper_sphere(tria, Point<spacedim>(), 0.5);
#else
  GridGenerator::hyper_sphere(tria, Point<spacedim>(0.02, 0.03), 0.5);
#endif
  tria.refine_global(n_refinements);

  // quadrature rule and FE for curvature
  FE_Q<dim, spacedim>       fe(fe_degree);
  QGaussLobatto<dim>        quadrature(fe_degree + 1);
  DoFHandler<dim, spacedim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // FE for normal
  FESystem<dim, spacedim>   fe_dim(fe, spacedim);
  DoFHandler<dim, spacedim> dof_handler_dim(tria);
  dof_handler_dim.distribute_dofs(fe_dim);

  // Set up MappingFEField
  Vector<double> euler_vector(dof_handler_dim.n_dofs());
  VectorTools::get_position_vector(dof_handler_dim,
                                   euler_vector,
                                   MappingQGeneric<dim, spacedim>(
                                     mapping_degree));
  MappingFEField<dim, spacedim> mapping(dof_handler_dim, euler_vector);


  // compute normal vector
  VectorType normal_vector(dof_handler_dim.n_dofs());
  compute_normal(mapping, dof_handler_dim, normal_vector);

  // compute curvature
  VectorType curvature_vector(dof_handler.n_dofs());
  compute_curvature(mapping,
                    dof_handler_dim,
                    dof_handler,
                    quadrature,
                    normal_vector,
                    curvature_vector);

#if false
  const unsigned int background_n_global_refinements = 6;
#else
  const unsigned int background_n_global_refinements = 80;
#endif
  const unsigned int background_fe_degree = 2;

  Triangulation<spacedim> background_tria;
#if false
  GridGenerator::hyper_cube(background_tria, -1.0, +1.0);
#else
  GridGenerator::subdivided_hyper_cube(background_tria,
                                       background_n_global_refinements,
                                       -2.5,
                                       2.5);
#endif
  if (background_n_global_refinements < 20)
    background_tria.refine_global(background_n_global_refinements);

  FESystem<spacedim>   background_fe(FE_Q<spacedim>{background_fe_degree},
                                   spacedim);
  DoFHandler<spacedim> background_dof_handler(background_tria);
  background_dof_handler.distribute_dofs(background_fe);

  MappingQ1<spacedim> background_mapping;

  VectorType force_vector_sharp_interface(background_dof_handler.n_dofs());

  compute_force_vector_sharp_interface(mapping,
                                       dof_handler,
                                       dof_handler_dim,
                                       QGauss<dim>(fe_degree + 1),
                                       background_mapping,
                                       background_dof_handler,
                                       1.0,
                                       normal_vector,
                                       curvature_vector,
                                       force_vector_sharp_interface);

  // write computed vectors to Paraview
  if (false)
    {
      DataOutBase::VtkFlags flags;
      // flags.write_higher_order_cells = true;

      DataOut<dim, DoFHandler<dim, spacedim>> data_out;
      data_out.set_flags(flags);
      data_out.add_data_vector(dof_handler, curvature_vector, "curvature");
      data_out.add_data_vector(dof_handler_dim, normal_vector, "normal");

      data_out.build_patches(mapping,
                             fe_degree + 1,
                             DataOut<dim, DoFHandler<dim, spacedim>>::
                               CurvedCellRegion::curved_inner_cells);
      data_out.write_vtu_with_pvtu_record(
        "./", "output-sharp_interfaces_02/data_surface", 0, MPI_COMM_WORLD);
    }

  if (false)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;

      DataOut<spacedim> data_out;
      data_out.set_flags(flags);
      data_out.attach_dof_handler(background_dof_handler);
      data_out.add_data_vector(background_dof_handler,
                               force_vector_sharp_interface,
                               "force");

      data_out.build_patches(background_mapping, background_fe_degree + 1);
      data_out.write_vtu_with_pvtu_record(
        "./", "output-sharp_interfaces_02/data_background", 0, MPI_COMM_WORLD);
    }

  {
    curvature_vector.print(std::cout);
  }
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  test<1>();
}
