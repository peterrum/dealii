// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
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


#ifndef dealii_vector_tools_evaluation_h
#define dealii_vector_tools_evaluation_h

#include <deal.II/base/config.h>

#include <deal.II/base/mpi_remote_point_evaluation.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_point_evaluation.h>

#include <map>

DEAL_II_NAMESPACE_OPEN

namespace VectorTools
{
  namespace EvaluationFlags
  {
    enum EvaluationFlags
    {
      avg    = 0,
      max    = 1,
      min    = 2,
      insert = 3
    };
  }

  /**
   * Given a (distributed) solution vector @p vector, evaluate the values at
   * the (arbitrary and even remote) points specified by @p evaluation_points.
   */
  template <int n_components, int dim, int spacedim, typename VectorType>
  std::vector<typename FEPointEvaluation<n_components, dim>::value_type>
  evaluate_at_points(
    const Mapping<dim> &                                  mapping,
    const DoFHandler<dim, spacedim> &                     dof_handler,
    const VectorType &                                    vector,
    const std::vector<Point<spacedim>> &                  evaluation_points,
    Utilities::MPI::RemotePointEvaluation<dim, spacedim> &eval);

  // inlined functions

  template <int n_components, int dim, int spacedim, typename VectorType>
  inline std::vector<typename FEPointEvaluation<n_components, dim>::value_type>
  evaluate_at_points(const Mapping<dim> &                mapping,
                     const DoFHandler<dim, spacedim> &   dof_handler,
                     const VectorType &                  vector,
                     const std::vector<Point<spacedim>> &evaluation_points,
                     Utilities::MPI::RemotePointEvaluation<dim, spacedim> &eval)
  {
    const auto modus = EvaluationFlags::avg;

    using value_type =
      typename FEPointEvaluation<n_components, dim>::value_type;

    const auto evaluation_point_results = [&]() {
      eval.reinit(evaluation_points, dof_handler.get_triangulation(), mapping);

      std::vector<std::unique_ptr<FEPointEvaluation<n_components, dim>>>
        evaluators(dof_handler.get_fe_collection().size());

      std::vector<value_type> solution_values;

      const auto fu = [&](auto &values, const auto &quadrature_points) {
        unsigned int i = 0;

        for (const auto &cells_and_n : std::get<0>(quadrature_points))
          {
            typename DoFHandler<dim>::active_cell_iterator cell = {
              &dof_handler.get_triangulation(),
              cells_and_n.first.first,
              cells_and_n.first.second,
              &dof_handler};

            const ArrayView<const Point<dim>> unit_points(
              std::get<1>(quadrature_points).data() + i, cells_and_n.second);
            solution_values.resize(
              dof_handler.get_fe(cell->active_fe_index()).n_dofs_per_cell());

            cell->get_dof_values(vector,
                                 solution_values.begin(),
                                 solution_values.end());

            const unsigned int active_fe_index = cell->active_fe_index();

            if (evaluators[active_fe_index] == nullptr)
              evaluators[active_fe_index] =
                std::make_unique<FEPointEvaluation<1, dim>>(
                  mapping, dof_handler.get_fe(active_fe_index));

            auto &evaluator = *evaluators[active_fe_index];

            evaluator.evaluate(cell,
                               unit_points,
                               solution_values,
                               dealii::EvaluationFlags::values);

            for (unsigned int q = 0; q < unit_points.size(); ++q, ++i)
              values[std::get<2>(quadrature_points)[i]] =
                evaluator.get_value(q);
          }
      };

      std::vector<value_type> evaluation_point_results;
      std::vector<value_type> buffer;

      eval.template process<value_type>(evaluation_point_results, buffer, fu);

      return evaluation_point_results;
    }();

    if (eval.is_unique_mapping())
      {
        return evaluation_point_results;
      }
    else
      {
        std::vector<value_type> unique_evaluation_point_results(
          evaluation_points.size());

        const auto reduce = [modus](const auto &values) {
          switch (modus)
            {
              case EvaluationFlags::avg:
                {
                  value_type result = {};
                  for (const auto &v : values)
                    result += v;
                  return result / values.size();
                }
              case EvaluationFlags::max:
                return *std::max_element(values.begin(), values.end());
              case EvaluationFlags::min:
                return *std::min_element(values.begin(), values.end());
              case EvaluationFlags::insert:
                return values[0];
              default:
                Assert(false, ExcNotImplemented());
                return values[0];
            }
        };

        const auto &ptr = eval.get_quadrature_points_ptr();

        for (unsigned int i = 0; i < evaluation_points.size(); ++i)
          {
            const auto n_entries = ptr[i + 1] - ptr[i];
            if (n_entries == 0)
              continue;

            unique_evaluation_point_results[i] =
              reduce(ArrayView<const value_type>(
                evaluation_point_results.data() + ptr[i], n_entries));
          }

        return unique_evaluation_point_results;
      }
  }
} // namespace VectorTools

DEAL_II_NAMESPACE_CLOSE

#endif // dealii_vector_tools_boundary_h
