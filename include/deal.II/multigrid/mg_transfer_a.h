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

#ifndef dealii_mg_transfer_a_h
#define dealii_mg_transfer_a_h

#include <deal.II/multigrid/mg_transfer_interface.h>

DEAL_II_NAMESPACE_OPEN

template <int dim, int degree, typename Number>
class TransferA : public Transfer<dim, Number>
{
public:
  void
  reinit(const DoFHandler<dim> &          dof_handler_fine,
         const DoFHandler<dim> &          dof_handler_coarse,
         const AffineConstraints<Number> &constraint_coarse)
  {
    (void)dof_handler_fine;
    (void)dof_handler_coarse;

    this->constraint_coarse.copy_from(constraint_coarse);

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

        this->partitioner_fine.reset(new Utilities::MPI::Partitioner(
          dof_handler_fine.locally_owned_dofs(), locally_relevant_dofs, comm));
        this->vec_fine.reinit(this->partitioner_fine);
      }
      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler_coarse,
                                                locally_relevant_dofs);

        this->partitioner_coarse.reset(new Utilities::MPI::Partitioner(
          dof_handler_coarse.locally_owned_dofs(),
          locally_relevant_dofs,
          comm));
        this->vec_coarse.reinit(this->partitioner_coarse);
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

    this->schemes.resize(2);
    auto &scheme = this->schemes.front();

    AssertDimension(dof_handler_coarse.get_fe().dofs_per_cell,
                    dof_handler_fine.get_fe().dofs_per_cell);

    this->schemes[0].n_cell_dofs_coarse = this->schemes[0].n_cell_dofs_fine =
      this->schemes[1].n_cell_dofs_coarse =
        dof_handler_coarse.get_fe().dofs_per_cell;
    this->schemes[1].n_cell_dofs_fine =
      dof_handler_coarse.get_fe().dofs_per_cell *
      GeometryInfo<dim>::max_children_per_cell;

    this->schemes[0].degree_coarse   = this->schemes[0].degree_fine =
      this->schemes[1].degree_coarse = dof_handler_coarse.get_fe().degree;
    this->schemes[1].degree_fine = dof_handler_coarse.get_fe().degree * 2 + 1;

    this->schemes[0].fine_element_is_continuous =
      this->schemes[1].fine_element_is_continuous =
        dof_handler_fine.get_fe().dofs_per_vertex > 0;

    // extract number of coarse cells
    {
      this->schemes[0].n_cells_coarse = 0;
      this->schemes[1].n_cells_coarse = 0;

      process_cells([&](const auto &,
                        const auto &) { this->schemes[0].n_cells_coarse++; },
                    [&](const auto &, const auto &, const auto c) {
                      if (c == 0)
                        this->schemes[1].n_cells_coarse++;
                    });
    }


    std::vector<std::vector<unsigned int>> offsets(
      GeometryInfo<dim>::max_children_per_cell,
      std::vector<unsigned int>(this->schemes[0].n_cell_dofs_coarse));
    {
      for (unsigned int c = 0; c < GeometryInfo<dim>::max_children_per_cell;
           c++)
        // TODO: get degree from somewhere else
        get_child_offset(c, degree + 1, degree, offsets[c]);
    }


    {
      this->schemes[0].level_dof_indices_fine.resize(
        this->schemes[0].n_cell_dofs_fine * this->schemes[0].n_cells_coarse);
      this->schemes[0].level_dof_indices_coarse.resize(
        this->schemes[0].n_cell_dofs_coarse * this->schemes[0].n_cells_coarse);

      this->schemes[1].level_dof_indices_fine.resize(
        this->schemes[1].n_cell_dofs_fine * this->schemes[1].n_cells_coarse);
      this->schemes[1].level_dof_indices_coarse.resize(
        this->schemes[1].n_cell_dofs_coarse * this->schemes[1].n_cells_coarse);

      std::vector<types::global_dof_index> local_dof_indices(
        this->schemes[0].n_cell_dofs_coarse);

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
        &this->schemes[0].level_dof_indices_coarse[0];
      unsigned int *level_dof_indices_fine_0 =
        &this->schemes[0].level_dof_indices_fine[0];

      unsigned int *level_dof_indices_coarse_1 =
        &this->schemes[1].level_dof_indices_coarse[0];
      unsigned int *level_dof_indices_fine_1 =
        &this->schemes[1].level_dof_indices_fine[0];

      process_cells(
        [&](const auto &cell_coarse, const auto &cell_fine) {
          // parent
          {
            cell_coarse->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < this->schemes[0].n_cell_dofs_coarse;
                 i++)
              level_dof_indices_coarse_0[i] =
                this->partitioner_coarse->global_to_local(
                  local_dof_indices[lexicographic_numbering[i]]);
          }

          // child
          {
            cell_fine->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < this->schemes[0].n_cell_dofs_coarse;
                 i++)
              level_dof_indices_fine_0[i] =
                this->partitioner_fine->global_to_local(
                  local_dof_indices[lexicographic_numbering[i]]);
          }

          // move pointers
          {
            level_dof_indices_coarse_0 += this->schemes[0].n_cell_dofs_coarse;
            level_dof_indices_fine_0 += this->schemes[0].n_cell_dofs_fine;
          }
        },
        [&](const auto &cell_coarse, const auto &cell_fine, const auto c) {
          // parent (only once at the beginning)
          if (c == 0)
            {
              cell_coarse->get_dof_indices(local_dof_indices);
              for (unsigned int i = 0; i < this->schemes[0].n_cell_dofs_coarse;
                   i++)
                level_dof_indices_coarse_1[offsets[c][i]] =
                  this->partitioner_coarse->global_to_local(
                    local_dof_indices[lexicographic_numbering[i]]);
            }

          // child
          {
            cell_fine->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < this->schemes[1].n_cell_dofs_coarse;
                 i++)
              level_dof_indices_fine_1[offsets[c][i]] =
                this->partitioner_fine->global_to_local(
                  local_dof_indices[lexicographic_numbering[i]]);
          }

          // move pointers (only once at the end)
          if (c + 1 == GeometryInfo<dim>::max_children_per_cell)
            {
              level_dof_indices_coarse_1 += this->schemes[1].n_cell_dofs_coarse;
              level_dof_indices_fine_1 += this->schemes[1].n_cell_dofs_fine;
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

      this->schemes[1].prolongation_matrix_1d.resize(fe->dofs_per_cell *
                                                     n_child_dofs_1d);

      for (unsigned int c = 0; c < GeometryInfo<1>::max_children_per_cell; ++c)
        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
          for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
            this->schemes[1]
              .prolongation_matrix_1d[i * n_child_dofs_1d + j + c * shift] =
              fe->get_prolongation_matrix(c)(renumbering[j], renumbering[i]);
    }


    // -------------------------------- weights --------------------------------
    if (this->schemes[0].fine_element_is_continuous)
      {
        this->schemes[0].weights.resize(this->schemes[0].n_cells_coarse *
                                        this->schemes[0].n_cell_dofs_fine);
        this->schemes[1].weights.resize(this->schemes[1].n_cells_coarse *
                                        this->schemes[1].n_cell_dofs_fine);

        LinearAlgebra::distributed::Vector<Number> touch_count;
        touch_count.reinit(this->partitioner_fine);

        const Quadrature<1> dummy_quadrature(
          std::vector<Point<1>>(1, Point<1>()));
        internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
        shape_info.reinit(dummy_quadrature, dof_handler_fine.get_fe(), 0);

        std::vector<Number> local_weights(this->schemes[0].n_cell_dofs_fine);

        unsigned int *level_dof_indices_fine_0 =
          &this->schemes[0].level_dof_indices_fine[0];
        unsigned int *level_dof_indices_fine_1 =
          &this->schemes[1].level_dof_indices_fine[0];

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

            for (unsigned int i = 0; i < this->schemes[0].n_cell_dofs_coarse;
                 i++)
              touch_count.local_element(level_dof_indices_fine_0[i]) +=
                local_weights[i];

            level_dof_indices_fine_0 += this->schemes[0].n_cell_dofs_fine;
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

            for (unsigned int i = 0; i < this->schemes[1].n_cell_dofs_coarse;
                 i++)
              touch_count.local_element(
                level_dof_indices_fine_1[offsets[c][i]]) += local_weights[i];

            // move pointers (only once at the end)
            if (c + 1 == GeometryInfo<dim>::max_children_per_cell)
              level_dof_indices_fine_1 += this->schemes[1].n_cell_dofs_fine;
          });

        touch_count.compress(VectorOperation::add);
        touch_count.update_ghost_values();

        Number *weights_0        = &this->schemes[0].weights[0];
        Number *weights_1        = &this->schemes[1].weights[0];
        level_dof_indices_fine_0 = &this->schemes[0].level_dof_indices_fine[0];
        level_dof_indices_fine_1 = &this->schemes[1].level_dof_indices_fine[0];

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

            for (unsigned int i = 0; i < this->schemes[0].n_cell_dofs_fine; i++)
              if (local_weights[i] == 0.0)
                weights_0[i] = Number(0.);
              else
                weights_0[i] = Number(1.) / touch_count.local_element(
                                              level_dof_indices_fine_0[i]);

            level_dof_indices_fine_0 += this->schemes[0].n_cell_dofs_fine;
            weights_0 += this->schemes[0].n_cell_dofs_fine;
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

            for (unsigned int i = 0; i < this->schemes[1].n_cell_dofs_coarse;
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
                level_dof_indices_fine_1 += this->schemes[1].n_cell_dofs_fine;
                weights_1 += this->schemes[1].n_cell_dofs_fine;
              }
          });
      }
  }

  void
  prolongate(const unsigned int                                to_level,
             LinearAlgebra::distributed::Vector<Number> &      dst,
             const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    this->do_prolongate_add(to_level, dst, src);
  }

  void
  prolongate_add(const unsigned int                                to_level,
                 LinearAlgebra::distributed::Vector<Number> &      dst,
                 const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    (void)to_level;
    (void)dst;
    (void)src;

    Assert(false, ExcNotImplemented());
  }

  void
  restrict_and_add(const unsigned int                                from_level,
                   LinearAlgebra::distributed::Vector<Number> &      dst,
                   const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    this->do_restrict_add(from_level, dst, src);
  }
};

DEAL_II_NAMESPACE_CLOSE

#endif
