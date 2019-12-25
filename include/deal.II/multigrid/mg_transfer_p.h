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

#ifndef dealii_mg_transfer_p_h
#define dealii_mg_transfer_p_h

#include <deal.II/multigrid/mg_transfer_interface.h>

DEAL_II_NAMESPACE_OPEN

template <int dim, int degree_fine, int degree_coarse, typename Number>
class TransferP : public Transfer<dim, Number>
{
public:
  void
  reinit(const DoFHandler<dim> &          dof_handler_fine,
         const DoFHandler<dim> &          dof_handler_coarse,
         const AffineConstraints<Number> &constraint_coarse)
  {
    AssertDimension(dof_handler_fine.get_triangulation().n_active_cells(),
                    dof_handler_coarse.get_triangulation().n_active_cells());

    auto process_cells = [&](const auto &fu) {
      typename DoFHandler<dim>::active_cell_iterator     //
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

    this->schemes.resize(1);
    auto &scheme = this->schemes.front();

    // extract number of coarse cells
    {
      scheme.n_cells_coarse = 0;
      process_cells(
        [&](const auto &, const auto &) { scheme.n_cells_coarse++; });
    }

    scheme.n_cell_dofs_coarse = dof_handler_coarse.get_fe().dofs_per_cell;
    scheme.n_cell_dofs_fine   = dof_handler_fine.get_fe().dofs_per_cell;

    scheme.fine_element_is_continuous =
      dof_handler_fine.get_fe().dofs_per_vertex > 0;

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

    {
      scheme.level_dof_indices_fine.resize(scheme.n_cell_dofs_fine *
                                           scheme.n_cells_coarse);
      scheme.level_dof_indices_coarse.resize(scheme.n_cell_dofs_coarse *
                                             scheme.n_cells_coarse);

      std::vector<types::global_dof_index> local_dof_indices_coarse(
        scheme.n_cell_dofs_coarse);
      std::vector<types::global_dof_index> local_dof_indices_fine(
        scheme.n_cell_dofs_fine);

      // ----------------------- lexicographic_numbering -----------------------
      std::vector<unsigned int> lexicographic_numbering_fine,
        lexicographic_numbering_coarse;
      {
        const Quadrature<1> dummy_quadrature(
          std::vector<Point<1>>(1, Point<1>()));
        internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
        shape_info.reinit(dummy_quadrature, dof_handler_fine.get_fe(), 0);
        lexicographic_numbering_fine = shape_info.lexicographic_numbering;

        shape_info.reinit(dummy_quadrature, dof_handler_coarse.get_fe(), 0);
        lexicographic_numbering_coarse = shape_info.lexicographic_numbering;
      }

      // ------------------------------- indices -------------------------------
      unsigned int *level_dof_indices_coarse_ =
        &scheme.level_dof_indices_coarse[0];
      unsigned int *level_dof_indices_fine_ = &scheme.level_dof_indices_fine[0];

      process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
        cell_coarse->get_dof_indices(local_dof_indices_coarse);
        for (unsigned int i = 0; i < scheme.n_cell_dofs_coarse; i++)
          level_dof_indices_coarse_[i] =
            this->partitioner_coarse->global_to_local(
              local_dof_indices_coarse[lexicographic_numbering_coarse[i]]);


        cell_fine->get_dof_indices(local_dof_indices_fine);
        for (unsigned int i = 0; i < scheme.n_cell_dofs_fine; i++)
          level_dof_indices_fine_[i] = this->partitioner_fine->global_to_local(
            local_dof_indices_fine[lexicographic_numbering_fine[i]]);


        level_dof_indices_coarse_ += scheme.n_cell_dofs_coarse;
        level_dof_indices_fine_ += scheme.n_cell_dofs_fine;
      });
    }

    // -------------------------- prolongation matrix --------------------------
    {
      AssertDimension(dof_handler_fine.get_fe().n_base_elements(), 1);
      std::string fe_name_fine =
        dof_handler_fine.get_fe().base_element(0).get_name();
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
          renumbering_fine[fe_fine->dofs_per_cell - fe_fine->dofs_per_vertex] =
            fe_fine->dofs_per_vertex;
      }



      AssertDimension(dof_handler_coarse.get_fe().n_base_elements(), 1);
      std::string fe_name_coarse =
        dof_handler_coarse.get_fe().base_element(0).get_name();
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
            GeometryInfo<1>::vertices_per_cell * fe_coarse->dofs_per_vertex + i;
        if (fe_coarse->dofs_per_vertex > 0)
          renumbering_coarse[fe_coarse->dofs_per_cell -
                             fe_coarse->dofs_per_vertex] =
            fe_coarse->dofs_per_vertex;
      }



      FullMatrix<Number> matrix(fe_fine->dofs_per_cell,
                                fe_coarse->dofs_per_cell);
      FETools::get_projection_matrix(*fe_coarse, *fe_fine, matrix);
      scheme.prolongation_matrix_1d.resize(fe_fine->dofs_per_cell *
                                           fe_coarse->dofs_per_cell);

      for (unsigned int i = 0, k = 0; i < fe_coarse->dofs_per_cell; ++i)
        for (unsigned int j = 0; j < fe_fine->dofs_per_cell; ++j, ++k)
          scheme.prolongation_matrix_1d[k] =
            matrix(renumbering_fine[j], renumbering_coarse[i]);
    }


    // -------------------------------- weights --------------------------------
    if (scheme.fine_element_is_continuous)
      {
        scheme.weights.resize(scheme.n_cells_coarse * scheme.n_cell_dofs_fine);

        LinearAlgebra::distributed::Vector<Number> touch_count;
        touch_count.reinit(this->partitioner_fine);

        const Quadrature<1> dummy_quadrature(
          std::vector<Point<1>>(1, Point<1>()));
        internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
        shape_info.reinit(dummy_quadrature, dof_handler_fine.get_fe(), 0);

        std::vector<Number> local_weights(scheme.n_cell_dofs_fine);

        unsigned int *level_dof_indices_fine_ =
          &scheme.level_dof_indices_fine[0];

        process_cells([&](const auto &, const auto &cell_fine) {
          std::fill(local_weights.begin(), local_weights.end(), Number(1.));

          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
            if (!cell_fine->at_boundary(f))
              if (cell_fine->level() > cell_fine->neighbor_level(f))
                {
                  auto &sh = shape_info.face_to_cell_index_nodal;
                  for (unsigned int i = 0; i < sh.size()[1]; i++)
                    local_weights[sh[f][i]] = Number(0.);
                }

          for (unsigned int i = 0; i < scheme.n_cell_dofs_fine; i++)
            touch_count.local_element(level_dof_indices_fine_[i]) +=
              local_weights[i];

          level_dof_indices_fine_ += scheme.n_cell_dofs_fine;
        });

        touch_count.compress(VectorOperation::add);
        touch_count.update_ghost_values();

        Number *weights_        = &scheme.weights[0];
        level_dof_indices_fine_ = &scheme.level_dof_indices_fine[0];

        process_cells([&](const auto &, const auto &cell_fine) {
          std::fill(local_weights.begin(), local_weights.end(), Number(1.));

          for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; f++)
            if (!cell_fine->at_boundary(f))
              if (cell_fine->level() > cell_fine->neighbor_level(f))
                {
                  auto &sh = shape_info.face_to_cell_index_nodal;
                  for (unsigned int i = 0; i < sh.size()[1]; i++)
                    local_weights[sh[f][i]] = Number(0.);
                }

          for (unsigned int i = 0; i < scheme.n_cell_dofs_fine; i++)
            if (local_weights[i] == 0.0)
              weights_[i] = Number(0.);
            else
              weights_[i] = Number(1.) / touch_count.local_element(
                                           level_dof_indices_fine_[i]);

          level_dof_indices_fine_ += scheme.n_cell_dofs_fine;
          weights_ += scheme.n_cell_dofs_fine;
        });
      }
  }

  void
  prolongate(const unsigned int                                to_level,
             LinearAlgebra::distributed::Vector<Number> &      dst,
             const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    this->template do_prolongate_add<degree_fine, degree_coarse>(to_level,
                                                                 dst,
                                                                 src);
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
    this->template do_restrict_add<degree_fine, degree_coarse>(from_level,
                                                               dst,
                                                               src);
  }
};

DEAL_II_NAMESPACE_CLOSE

#endif
