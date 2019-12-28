// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
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


#ifndef dealii_mg_transfer_inteface_util_templates_h
#define dealii_mg_transfer_inteface_util_templates_h

#include <deal.II/base/config.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/hp/dof_handler.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/multigrid/mg_transfer_interface.h>

DEAL_II_NAMESPACE_OPEN

namespace MGTransferUtil
{
  template <int dim, typename Number>
  void
  setup_global_coarsening_transfer(
    const DoFHandler<dim> &          dof_handler_fine,
    const DoFHandler<dim> &          dof_handler_coarse,
    const AffineConstraints<Number> &constraint_coarse,
    Transfer<dim, Number> &          transfer)
  {
    transfer.constraint_coarse.copy_from(constraint_coarse);

    {
      const parallel::TriangulationBase<dim> *dist_tria =
        dynamic_cast<const parallel::TriangulationBase<dim> *>(
          &(dof_handler_coarse.get_triangulation()));

      MPI_Comm comm =
        dist_tria != nullptr ? dist_tria->get_communicator() : MPI_COMM_SELF;

      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler_fine,
                                                locally_relevant_dofs);

        transfer.partitioner_fine.reset(new Utilities::MPI::Partitioner(
          dof_handler_fine.locally_owned_dofs(), locally_relevant_dofs, comm));
        transfer.vec_fine.reinit(transfer.partitioner_fine);
      }
      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler_coarse,
                                                locally_relevant_dofs);

        transfer.partitioner_coarse.reset(new Utilities::MPI::Partitioner(
          dof_handler_coarse.locally_owned_dofs(),
          locally_relevant_dofs,
          comm));
        transfer.vec_coarse.reinit(transfer.partitioner_coarse);
      }
    }

    auto process_cells = [&](const auto &fu_1, const auto &fu_2) {
      const auto &triangulation_fine = dof_handler_fine.get_triangulation();

      for (const auto cell_coarse : dof_handler_coarse.active_cell_iterators())
        {
          if (cell_coarse->id().to_cell(triangulation_fine)->has_children())
            {
              for (unsigned int c = 0;
                   c <
                   cell_coarse->id().to_cell(triangulation_fine)->n_children();
                   c++)
                {
                  typename DoFHandler<dim>::cell_iterator cell_fine(
                    *cell_coarse->id().to_cell(triangulation_fine)->child(c),
                    &dof_handler_fine);
                  fu_2(cell_coarse, cell_fine, c);
                }
            }
          else
            {
              typename DoFHandler<dim>::cell_iterator cell_fine(
                *cell_coarse->id().to_cell(triangulation_fine),
                &dof_handler_fine);
              fu_1(cell_coarse, cell_fine);
            }
        }
    };

    auto compute_shift_within_children = [](const unsigned int child,
                                            const unsigned int fe_shift_1d,
                                            const unsigned int fe_degree) {
      // we put the degrees of freedom of all child cells in lexicographic
      // ordering
      unsigned int c_tensor_index[dim];
      unsigned int tmp = child;
      for (unsigned int d = 0; d < dim; ++d)
        {
          c_tensor_index[d] = tmp % 2;
          tmp /= 2;
        }
      const unsigned int n_child_dofs_1d = fe_degree + 1 + fe_shift_1d;
      unsigned int       factor          = 1;
      unsigned int       shift           = fe_shift_1d * c_tensor_index[0];
      for (unsigned int d = 1; d < dim; ++d)
        {
          factor *= n_child_dofs_1d;
          shift = shift + factor * fe_shift_1d * c_tensor_index[d];
        }
      return shift;
    };

    auto get_child_offset = [compute_shift_within_children](
                              const unsigned int         child,
                              const unsigned int         fe_shift_1d,
                              const unsigned int         fe_degree,
                              std::vector<unsigned int> &local_dof_indices) {
      const unsigned int n_child_dofs_1d = fe_degree + 1 + fe_shift_1d;
      const unsigned int shift =
        compute_shift_within_children(child, fe_shift_1d, fe_degree);
      const unsigned int n_components =
        local_dof_indices.size() / Utilities::fixed_power<dim>(fe_degree + 1);
      const unsigned int n_scalar_cell_dofs =
        Utilities::fixed_power<dim>(n_child_dofs_1d);
      for (unsigned int c = 0, m = 0; c < n_components; ++c)
        for (unsigned int k = 0; k < (dim > 2 ? (fe_degree + 1) : 1); ++k)
          for (unsigned int j = 0; j < (dim > 1 ? (fe_degree + 1) : 1); ++j)
            for (unsigned int i = 0; i < (fe_degree + 1); ++i, ++m)
              local_dof_indices[m] = c * n_scalar_cell_dofs +
                                     k * n_child_dofs_1d * n_child_dofs_1d +
                                     j * n_child_dofs_1d + i + shift;
    };

    transfer.schemes.resize(2);
    auto &scheme = transfer.schemes.front();

    AssertDimension(dof_handler_coarse.get_fe().dofs_per_cell,
                    dof_handler_fine.get_fe().dofs_per_cell);

    transfer.schemes[0].n_cell_dofs_coarse =
      transfer.schemes[0].n_cell_dofs_fine =
        transfer.schemes[1].n_cell_dofs_coarse =
          dof_handler_coarse.get_fe().dofs_per_cell;
    transfer.schemes[1].n_cell_dofs_fine =
      dof_handler_coarse.get_fe().dofs_per_cell *
      GeometryInfo<dim>::max_children_per_cell;

    transfer.schemes[0].degree_coarse   = transfer.schemes[0].degree_fine =
      transfer.schemes[1].degree_coarse = dof_handler_coarse.get_fe().degree;
    transfer.schemes[1].degree_fine =
      dof_handler_coarse.get_fe().degree * 2 + 1;

    transfer.schemes[0].fine_element_is_continuous =
      transfer.schemes[1].fine_element_is_continuous =
        dof_handler_fine.get_fe().dofs_per_vertex > 0;

    // extract number of coarse cells
    {
      transfer.schemes[0].n_cells_coarse = 0;
      transfer.schemes[1].n_cells_coarse = 0;

      process_cells([&](const auto &,
                        const auto &) { transfer.schemes[0].n_cells_coarse++; },
                    [&](const auto &, const auto &, const auto c) {
                      if (c == 0)
                        transfer.schemes[1].n_cells_coarse++;
                    });
    }


    std::vector<std::vector<unsigned int>> offsets(
      GeometryInfo<dim>::max_children_per_cell,
      std::vector<unsigned int>(transfer.schemes[0].n_cell_dofs_coarse));
    {
      for (unsigned int c = 0; c < GeometryInfo<dim>::max_children_per_cell;
           c++)
        get_child_offset(c,
                         dof_handler_fine.get_fe().degree + 1,
                         dof_handler_fine.get_fe().degree,
                         offsets[c]);
    }


    {
      transfer.schemes[0].level_dof_indices_fine.resize(
        transfer.schemes[0].n_cell_dofs_fine *
        transfer.schemes[0].n_cells_coarse);
      transfer.schemes[0].level_dof_indices_coarse.resize(
        transfer.schemes[0].n_cell_dofs_coarse *
        transfer.schemes[0].n_cells_coarse);

      transfer.schemes[1].level_dof_indices_fine.resize(
        transfer.schemes[1].n_cell_dofs_fine *
        transfer.schemes[1].n_cells_coarse);
      transfer.schemes[1].level_dof_indices_coarse.resize(
        transfer.schemes[1].n_cell_dofs_coarse *
        transfer.schemes[1].n_cells_coarse);

      std::vector<types::global_dof_index> local_dof_indices(
        transfer.schemes[0].n_cell_dofs_coarse);

      // ----------------------- lexicographic_numbering -----------------------
      std::vector<unsigned int> lexicographic_numbering;
      {
        const Quadrature<1> dummy_quadrature(
          std::vector<Point<1>>(1, Point<1>()));
        internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
        shape_info.reinit(dummy_quadrature, dof_handler_fine.get_fe(), 0);
        lexicographic_numbering = shape_info.lexicographic_numbering;
      }

      // ------------------------------- indices -------------------------------
      unsigned int *level_dof_indices_coarse_0 =
        &transfer.schemes[0].level_dof_indices_coarse[0];
      unsigned int *level_dof_indices_fine_0 =
        &transfer.schemes[0].level_dof_indices_fine[0];

      unsigned int *level_dof_indices_coarse_1 =
        &transfer.schemes[1].level_dof_indices_coarse[0];
      unsigned int *level_dof_indices_fine_1 =
        &transfer.schemes[1].level_dof_indices_fine[0];

      process_cells(
        [&](const auto &cell_coarse, const auto &cell_fine) {
          // parent
          {
            cell_coarse->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < transfer.schemes[0].n_cell_dofs_coarse;
                 i++)
              level_dof_indices_coarse_0[i] =
                transfer.partitioner_coarse->global_to_local(
                  local_dof_indices[lexicographic_numbering[i]]);
          }

          // child
          {
            cell_fine->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < transfer.schemes[0].n_cell_dofs_coarse;
                 i++)
              level_dof_indices_fine_0[i] =
                transfer.partitioner_fine->global_to_local(
                  local_dof_indices[lexicographic_numbering[i]]);
          }

          // move pointers
          {
            level_dof_indices_coarse_0 +=
              transfer.schemes[0].n_cell_dofs_coarse;
            level_dof_indices_fine_0 += transfer.schemes[0].n_cell_dofs_fine;
          }
        },
        [&](const auto &cell_coarse, const auto &cell_fine, const auto c) {
          // parent (only once at the beginning)
          if (c == 0)
            {
              cell_coarse->get_dof_indices(local_dof_indices);
              for (unsigned int i = 0;
                   i < transfer.schemes[0].n_cell_dofs_coarse;
                   i++)
                level_dof_indices_coarse_1[offsets[c][i]] =
                  transfer.partitioner_coarse->global_to_local(
                    local_dof_indices[lexicographic_numbering[i]]);
            }

          // child
          {
            cell_fine->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < transfer.schemes[1].n_cell_dofs_coarse;
                 i++)
              level_dof_indices_fine_1[offsets[c][i]] =
                transfer.partitioner_fine->global_to_local(
                  local_dof_indices[lexicographic_numbering[i]]);
          }

          // move pointers (only once at the end)
          if (c + 1 == GeometryInfo<dim>::max_children_per_cell)
            {
              level_dof_indices_coarse_1 +=
                transfer.schemes[1].n_cell_dofs_coarse;
              level_dof_indices_fine_1 += transfer.schemes[1].n_cell_dofs_fine;
            }
        });
    }

    // -------------- prolongation matrix (0) -> identity matrix ---------------
    {
      AssertDimension(dof_handler_fine.get_fe().n_base_elements(), 1);
      std::string fe_name =
        dof_handler_fine.get_fe().base_element(0).get_name();
      {
        const std::size_t template_starts = fe_name.find_first_of('<');
        Assert(fe_name[template_starts + 1] ==
                 (dim == 1 ? '1' : (dim == 2 ? '2' : '3')),
               ExcInternalError());
        fe_name[template_starts + 1] = '1';
      }
      const std::unique_ptr<FiniteElement<1>> fe(
        FETools::get_fe_by_name<1, 1>(fe_name));

      scheme.prolongation_matrix_1d.resize(fe->dofs_per_cell *
                                           fe->dofs_per_cell);

      for (unsigned int i = 0; i < fe->dofs_per_cell; i++)
        scheme.prolongation_matrix_1d[i + i * fe->dofs_per_cell] = Number(1.0);
    }

    // ------------------------ prolongation matrix (1) ------------------------
    {
      AssertDimension(dof_handler_fine.get_fe().n_base_elements(), 1);
      std::string fe_name =
        dof_handler_fine.get_fe().base_element(0).get_name();
      {
        const std::size_t template_starts = fe_name.find_first_of('<');
        Assert(fe_name[template_starts + 1] ==
                 (dim == 1 ? '1' : (dim == 2 ? '2' : '3')),
               ExcInternalError());
        fe_name[template_starts + 1] = '1';
      }
      const std::unique_ptr<FiniteElement<1>> fe(
        FETools::get_fe_by_name<1, 1>(fe_name));

      std::vector<unsigned int> renumbering(fe->dofs_per_cell);
      {
        AssertIndexRange(fe->dofs_per_vertex, 2);
        renumbering[0] = 0;
        for (unsigned int i = 0; i < fe->dofs_per_line; ++i)
          renumbering[i + fe->dofs_per_vertex] =
            GeometryInfo<1>::vertices_per_cell * fe->dofs_per_vertex + i;
        if (fe->dofs_per_vertex > 0)
          renumbering[fe->dofs_per_cell - fe->dofs_per_vertex] =
            fe->dofs_per_vertex;
      }

      // TODO: data structures are saved in form of DG data structures here
      const unsigned int shift           = fe->dofs_per_cell;
      const unsigned int n_child_dofs_1d = fe->dofs_per_cell * 2;

      transfer.schemes[1].prolongation_matrix_1d.resize(fe->dofs_per_cell *
                                                        n_child_dofs_1d);

      for (unsigned int c = 0; c < GeometryInfo<1>::max_children_per_cell; ++c)
        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
          for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
            transfer.schemes[1]
              .prolongation_matrix_1d[i * n_child_dofs_1d + j + c * shift] =
              fe->get_prolongation_matrix(c)(renumbering[j], renumbering[i]);
    }


    // -------------------------------- weights --------------------------------
    if (transfer.schemes[0].fine_element_is_continuous)
      {
        transfer.schemes[0].weights.resize(
          transfer.schemes[0].n_cells_coarse *
          transfer.schemes[0].n_cell_dofs_fine);
        transfer.schemes[1].weights.resize(
          transfer.schemes[1].n_cells_coarse *
          transfer.schemes[1].n_cell_dofs_fine);

        LinearAlgebra::distributed::Vector<Number> touch_count;
        touch_count.reinit(transfer.partitioner_fine);

        const Quadrature<1> dummy_quadrature(
          std::vector<Point<1>>(1, Point<1>()));
        internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
        shape_info.reinit(dummy_quadrature, dof_handler_fine.get_fe(), 0);

        std::vector<Number> local_weights(transfer.schemes[0].n_cell_dofs_fine);

        unsigned int *level_dof_indices_fine_0 =
          &transfer.schemes[0].level_dof_indices_fine[0];
        unsigned int *level_dof_indices_fine_1 =
          &transfer.schemes[1].level_dof_indices_fine[0];

        process_cells(
          [&](const auto &, const auto &cell_fine) {
            std::fill(local_weights.begin(), local_weights.end(), Number(1.));

            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
              if (!cell_fine->at_boundary(f))
                if (cell_fine->level() > cell_fine->neighbor_level(f))
                  {
                    auto &sh = shape_info.face_to_cell_index_nodal;
                    for (unsigned int i = 0; i < sh.size()[1]; i++)
                      local_weights[sh[f][i]] = Number(0.);
                  }

            for (unsigned int i = 0; i < transfer.schemes[0].n_cell_dofs_coarse;
                 i++)
              touch_count.local_element(level_dof_indices_fine_0[i]) +=
                local_weights[i];

            level_dof_indices_fine_0 += transfer.schemes[0].n_cell_dofs_fine;
          },
          [&](const auto &, const auto &cell_fine, const auto c) {
            std::fill(local_weights.begin(), local_weights.end(), Number(1.));

            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
              if (!cell_fine->at_boundary(f))
                if (cell_fine->level() > cell_fine->neighbor_level(f))
                  {
                    auto &sh = shape_info.face_to_cell_index_nodal;
                    for (unsigned int i = 0; i < sh.size()[1]; i++)
                      local_weights[sh[f][i]] = Number(0.);
                  }

            for (unsigned int i = 0; i < transfer.schemes[1].n_cell_dofs_coarse;
                 i++)
              touch_count.local_element(
                level_dof_indices_fine_1[offsets[c][i]]) += local_weights[i];

            // move pointers (only once at the end)
            if (c + 1 == GeometryInfo<dim>::max_children_per_cell)
              level_dof_indices_fine_1 += transfer.schemes[1].n_cell_dofs_fine;
          });

        touch_count.compress(VectorOperation::add);
        touch_count.update_ghost_values();

        Number *weights_0 = &transfer.schemes[0].weights[0];
        Number *weights_1 = &transfer.schemes[1].weights[0];
        level_dof_indices_fine_0 =
          &transfer.schemes[0].level_dof_indices_fine[0];
        level_dof_indices_fine_1 =
          &transfer.schemes[1].level_dof_indices_fine[0];

        process_cells(
          [&](const auto &, const auto &cell_fine) {
            std::fill(local_weights.begin(), local_weights.end(), Number(1.));

            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
              if (!cell_fine->at_boundary(f))
                if (cell_fine->level() > cell_fine->neighbor_level(f))
                  {
                    auto &sh = shape_info.face_to_cell_index_nodal;
                    for (unsigned int i = 0; i < sh.size()[1]; i++)
                      local_weights[sh[f][i]] = Number(0.);
                  }

            for (unsigned int i = 0; i < transfer.schemes[0].n_cell_dofs_fine;
                 i++)
              if (local_weights[i] == 0.0)
                weights_0[i] = Number(0.);
              else
                weights_0[i] = Number(1.) / touch_count.local_element(
                                              level_dof_indices_fine_0[i]);

            level_dof_indices_fine_0 += transfer.schemes[0].n_cell_dofs_fine;
            weights_0 += transfer.schemes[0].n_cell_dofs_fine;
          },
          [&](const auto &, const auto &cell_fine, const auto c) {
            std::fill(local_weights.begin(), local_weights.end(), Number(1.));

            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
              if (!cell_fine->at_boundary(f))
                if (cell_fine->level() > cell_fine->neighbor_level(f))
                  {
                    auto &sh = shape_info.face_to_cell_index_nodal;
                    for (unsigned int i = 0; i < sh.size()[1]; i++)
                      local_weights[sh[f][i]] = Number(0.);
                  }

            for (unsigned int i = 0; i < transfer.schemes[1].n_cell_dofs_coarse;
                 i++)
              if (local_weights[i] == 0.0)
                weights_1[offsets[c][i]] = Number(0.);
              else
                weights_1[offsets[c][i]] =
                  Number(1.) / touch_count.local_element(
                                 level_dof_indices_fine_1[offsets[c][i]]);

            // move pointers (only once at the end)
            if (c + 1 == GeometryInfo<dim>::max_children_per_cell)
              {
                level_dof_indices_fine_1 +=
                  transfer.schemes[1].n_cell_dofs_fine;
                weights_1 += transfer.schemes[1].n_cell_dofs_fine;
              }
          });
      }
  }


  template <int dim, typename Number, typename MeshType>
  void
  setup_polynomial_transfer(const MeshType &                 dof_handler_fine,
                            const MeshType &                 dof_handler_coarse,
                            const AffineConstraints<Number> &constraint_fine,
                            const AffineConstraints<Number> &constraint_coarse,
                            Transfer<dim, Number> &          transfer)
  {
    AssertDimension(dof_handler_fine.get_triangulation().n_active_cells(),
                    dof_handler_coarse.get_triangulation().n_active_cells());

    auto process_cells = [&](const auto &fu) {
      typename MeshType::active_cell_iterator            //
        cell_coarse = dof_handler_coarse.begin_active(), //
        end_coarse  = dof_handler_coarse.end(),          //
        cell_fine   = dof_handler_fine.begin_active();

      for (; cell_coarse != end_coarse; cell_coarse++, cell_fine++)
        {
          if (!cell_coarse->is_locally_owned())
            continue;

          fu(cell_coarse, cell_fine);
        }
    };

    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      fe_index_pairs;

    process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
      fe_index_pairs.emplace(
        std::pair<unsigned int, unsigned int>(cell_coarse->active_fe_index(),
                                              cell_fine->active_fe_index()),
        0);
    });

    unsigned int counter = 0;
    for (auto &f : fe_index_pairs)
      f.second = counter++;

    transfer.schemes.resize(fe_index_pairs.size());

    // extract number of coarse cells
    {
      for (auto &scheme : transfer.schemes)
        scheme.n_cells_coarse = 0;
      process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
        transfer
          .schemes[fe_index_pairs[std::pair<unsigned int, unsigned int>(
            cell_coarse->active_fe_index(), cell_fine->active_fe_index())]]
          .n_cells_coarse++;
      });
    }

    for (const auto fe_index_pair : fe_index_pairs)
      {
        transfer.schemes[fe_index_pair.second].n_cell_dofs_coarse =
          dof_handler_coarse.get_fe(fe_index_pair.first.first).dofs_per_cell;
        transfer.schemes[fe_index_pair.second].n_cell_dofs_fine =
          dof_handler_fine.get_fe(fe_index_pair.first.second).dofs_per_cell;

        transfer.schemes[fe_index_pair.second].degree_coarse =
          dof_handler_coarse.get_fe(fe_index_pair.first.first).degree;
        transfer.schemes[fe_index_pair.second].degree_fine =
          dof_handler_fine.get_fe(fe_index_pair.first.second).degree;

        transfer.schemes[fe_index_pair.second].fine_element_is_continuous =
          dof_handler_fine.get_fe(fe_index_pair.first.second).dofs_per_vertex >
          0;
      }

    transfer.constraint_coarse.copy_from(constraint_coarse);

    {
      const parallel::TriangulationBase<dim> *dist_tria =
        dynamic_cast<const parallel::TriangulationBase<dim> *>(
          &(dof_handler_coarse.get_triangulation()));

      MPI_Comm comm =
        dist_tria != nullptr ? dist_tria->get_communicator() : MPI_COMM_SELF;

      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler_fine,
                                                locally_relevant_dofs);

        transfer.partitioner_fine.reset(new Utilities::MPI::Partitioner(
          dof_handler_fine.locally_owned_dofs(), locally_relevant_dofs, comm));
        transfer.vec_fine.reinit(transfer.partitioner_fine);
      }
      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler_coarse,
                                                locally_relevant_dofs);

        transfer.partitioner_coarse.reset(new Utilities::MPI::Partitioner(
          dof_handler_coarse.locally_owned_dofs(),
          locally_relevant_dofs,
          comm));
        transfer.vec_coarse.reinit(transfer.partitioner_coarse);
      }
    }

    {
      std::vector<std::vector<unsigned int>> lexicographic_numbering_fine(
        fe_index_pairs.size());
      std::vector<std::vector<unsigned int>> lexicographic_numbering_coarse(
        fe_index_pairs.size());
      std::vector<std::vector<types::global_dof_index>>
                                                        local_dof_indices_coarse(fe_index_pairs.size());
      std::vector<std::vector<types::global_dof_index>> local_dof_indices_fine(
        fe_index_pairs.size());

      for (const auto fe_index_pair : fe_index_pairs)
        {
          local_dof_indices_coarse[fe_index_pair.second].resize(
            transfer.schemes[fe_index_pair.second].n_cell_dofs_coarse);
          local_dof_indices_fine[fe_index_pair.second].resize(
            transfer.schemes[fe_index_pair.second].n_cell_dofs_fine);

          transfer.schemes[fe_index_pair.second].level_dof_indices_fine.resize(
            transfer.schemes[fe_index_pair.second].n_cell_dofs_fine *
            transfer.schemes[fe_index_pair.second].n_cells_coarse);
          transfer.schemes[fe_index_pair.second]
            .level_dof_indices_coarse.resize(
              transfer.schemes[fe_index_pair.second].n_cell_dofs_coarse *
              transfer.schemes[fe_index_pair.second].n_cells_coarse);


          // ----------------------- lexicographic_numbering
          // -----------------------
          {
            const Quadrature<1> dummy_quadrature(
              std::vector<Point<1>>(1, Point<1>()));
            internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
            shape_info.reinit(dummy_quadrature,
                              dof_handler_fine.get_fe(
                                fe_index_pair.first.second),
                              0);
            lexicographic_numbering_fine[fe_index_pair.second] =
              shape_info.lexicographic_numbering;

            shape_info.reinit(dummy_quadrature,
                              dof_handler_coarse.get_fe(
                                fe_index_pair.first.first),
                              0);
            lexicographic_numbering_coarse[fe_index_pair.second] =
              shape_info.lexicographic_numbering;
          }
        }

      // ------------------------------- indices -------------------------------
      std::vector<unsigned int *> level_dof_indices_coarse_(
        fe_index_pairs.size());
      std::vector<unsigned int *> level_dof_indices_fine_(
        fe_index_pairs.size());

      for (unsigned int i = 0; i < fe_index_pairs.size(); i++)
        {
          level_dof_indices_coarse_[i] =
            &transfer.schemes[i].level_dof_indices_coarse[0];
          level_dof_indices_fine_[i] =
            &transfer.schemes[i].level_dof_indices_fine[0];
        }

      process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
        const auto fe_pair_no =
          fe_index_pairs[std::pair<unsigned int, unsigned int>(
            cell_coarse->active_fe_index(), cell_fine->active_fe_index())];

        cell_coarse->get_dof_indices(local_dof_indices_coarse[fe_pair_no]);
        for (unsigned int i = 0;
             i < transfer.schemes[fe_pair_no].n_cell_dofs_coarse;
             i++)
          level_dof_indices_coarse_[fe_pair_no][i] =
            transfer.partitioner_coarse->global_to_local(
              local_dof_indices_coarse
                [fe_pair_no][lexicographic_numbering_coarse[fe_pair_no][i]]);


        cell_fine->get_dof_indices(local_dof_indices_fine[fe_pair_no]);
        for (unsigned int i = 0;
             i < transfer.schemes[fe_pair_no].n_cell_dofs_fine;
             i++)
          level_dof_indices_fine_[fe_pair_no][i] =
            transfer.partitioner_fine->global_to_local(
              local_dof_indices_fine
                [fe_pair_no][lexicographic_numbering_fine[fe_pair_no][i]]);


        level_dof_indices_coarse_[fe_pair_no] +=
          transfer.schemes[fe_pair_no].n_cell_dofs_coarse;
        level_dof_indices_fine_[fe_pair_no] +=
          transfer.schemes[fe_pair_no].n_cell_dofs_fine;
      });
    }

    // -------------------------- prolongation matrix --------------------------
    for (auto const &[fe_index_pair, fe_index_no] : fe_index_pairs)
      {
        AssertDimension(
          dof_handler_fine.get_fe(fe_index_pair.second).n_base_elements(), 1);
        std::string fe_name_fine = dof_handler_fine.get_fe(fe_index_pair.second)
                                     .base_element(0)
                                     .get_name();
        {
          const std::size_t template_starts = fe_name_fine.find_first_of('<');
          Assert(fe_name_fine[template_starts + 1] ==
                   (dim == 1 ? '1' : (dim == 2 ? '2' : '3')),
                 ExcInternalError());
          fe_name_fine[template_starts + 1] = '1';
        }
        const std::unique_ptr<FiniteElement<1>> fe_fine(
          FETools::get_fe_by_name<1, 1>(fe_name_fine));

        std::vector<unsigned int> renumbering_fine(fe_fine->dofs_per_cell);
        {
          AssertIndexRange(fe_fine->dofs_per_vertex, 2);
          renumbering_fine[0] = 0;
          for (unsigned int i = 0; i < fe_fine->dofs_per_line; ++i)
            renumbering_fine[i + fe_fine->dofs_per_vertex] =
              GeometryInfo<1>::vertices_per_cell * fe_fine->dofs_per_vertex + i;
          if (fe_fine->dofs_per_vertex > 0)
            renumbering_fine[fe_fine->dofs_per_cell -
                             fe_fine->dofs_per_vertex] =
              fe_fine->dofs_per_vertex;
        }



        AssertDimension(
          dof_handler_coarse.get_fe(fe_index_pair.first).n_base_elements(), 1);
        std::string fe_name_coarse =
          dof_handler_coarse.get_fe(fe_index_pair.first)
            .base_element(0)
            .get_name();
        {
          const std::size_t template_starts = fe_name_coarse.find_first_of('<');
          Assert(fe_name_coarse[template_starts + 1] ==
                   (dim == 1 ? '1' : (dim == 2 ? '2' : '3')),
                 ExcInternalError());
          fe_name_coarse[template_starts + 1] = '1';
        }
        const std::unique_ptr<FiniteElement<1>> fe_coarse(
          FETools::get_fe_by_name<1, 1>(fe_name_coarse));

        std::vector<unsigned int> renumbering_coarse(fe_coarse->dofs_per_cell);
        {
          AssertIndexRange(fe_coarse->dofs_per_vertex, 2);
          renumbering_coarse[0] = 0;
          for (unsigned int i = 0; i < fe_coarse->dofs_per_line; ++i)
            renumbering_coarse[i + fe_coarse->dofs_per_vertex] =
              GeometryInfo<1>::vertices_per_cell * fe_coarse->dofs_per_vertex +
              i;
          if (fe_coarse->dofs_per_vertex > 0)
            renumbering_coarse[fe_coarse->dofs_per_cell -
                               fe_coarse->dofs_per_vertex] =
              fe_coarse->dofs_per_vertex;
        }



        FullMatrix<double> matrix(fe_fine->dofs_per_cell,
                                  fe_coarse->dofs_per_cell);
        FETools::get_projection_matrix(*fe_coarse, *fe_fine, matrix);
        transfer.schemes[fe_index_no].prolongation_matrix_1d.resize(
          fe_fine->dofs_per_cell * fe_coarse->dofs_per_cell);

        for (unsigned int i = 0, k = 0; i < fe_coarse->dofs_per_cell; ++i)
          for (unsigned int j = 0; j < fe_fine->dofs_per_cell; ++j, ++k)
            transfer.schemes[fe_index_no].prolongation_matrix_1d[k] =
              matrix(renumbering_fine[j], renumbering_coarse[i]);
      }

    // -------------------------------- weights --------------------------------
    if (transfer.schemes.front().fine_element_is_continuous)
      {
        for (auto &scheme : transfer.schemes)
          scheme.weights.resize(scheme.n_cells_coarse *
                                scheme.n_cell_dofs_fine);

        LinearAlgebra::distributed::Vector<Number> touch_count;
        touch_count.reinit(transfer.partitioner_fine);

        std::vector<unsigned int *> level_dof_indices_fine_(
          fe_index_pairs.size());
        std::vector<Number *> weights_(fe_index_pairs.size());

        for (unsigned int i = 0; i < fe_index_pairs.size(); i++)
          level_dof_indices_fine_[i] =
            &transfer.schemes[i].level_dof_indices_fine[0];

        process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
          const auto fe_pair_no =
            fe_index_pairs[std::pair<unsigned int, unsigned int>(
              cell_coarse->active_fe_index(), cell_fine->active_fe_index())];

          for (unsigned int i = 0;
               i < transfer.schemes[fe_pair_no].n_cell_dofs_fine;
               i++)
            if (constraint_fine.is_constrained(
                  transfer.partitioner_fine->local_to_global(
                    level_dof_indices_fine_[fe_pair_no][i])) == false)
              touch_count.local_element(
                level_dof_indices_fine_[fe_pair_no][i]) += 1;

          level_dof_indices_fine_[fe_pair_no] +=
            transfer.schemes[fe_pair_no].n_cell_dofs_fine;
        });

        touch_count.compress(VectorOperation::add);
        touch_count.update_ghost_values();

        for (unsigned int i = 0; i < fe_index_pairs.size(); i++)
          {
            level_dof_indices_fine_[i] =
              &transfer.schemes[i].level_dof_indices_fine[0];
            weights_[i] = &transfer.schemes[i].weights[0];
          }

        process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
          const auto fe_pair_no =
            fe_index_pairs[std::pair<unsigned int, unsigned int>(
              cell_coarse->active_fe_index(), cell_fine->active_fe_index())];

          for (unsigned int i = 0;
               i < transfer.schemes[fe_pair_no].n_cell_dofs_fine;
               i++)
            if (constraint_fine.is_constrained(
                  transfer.partitioner_fine->local_to_global(
                    level_dof_indices_fine_[fe_pair_no][i])) == true)
              weights_[fe_pair_no][i] = Number(0.);
            else
              weights_[fe_pair_no][i] =
                Number(1.) / touch_count.local_element(
                               level_dof_indices_fine_[fe_pair_no][i]);

          level_dof_indices_fine_[fe_pair_no] +=
            transfer.schemes[fe_pair_no].n_cell_dofs_fine;
          weights_[fe_pair_no] += transfer.schemes[fe_pair_no].n_cell_dofs_fine;
        });
      }
  }

} // namespace MGTransferUtil

DEAL_II_NAMESPACE_CLOSE

#endif