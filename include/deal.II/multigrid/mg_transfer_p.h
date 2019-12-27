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
  template <typename MeshType>
  void
  reinit(const MeshType &                 dof_handler_fine,
         const MeshType &                 dof_handler_coarse,
         const AffineConstraints<Number> &constraint_fine,
         const AffineConstraints<Number> &constraint_coarse)
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

    AssertDimension(fe_index_pairs.size(), 1);

    this->schemes.resize(fe_index_pairs.size());

    // extract number of coarse cells
    {
      for (auto &scheme : this->schemes)
        scheme.n_cells_coarse = 0;
      process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
        this
          ->schemes[fe_index_pairs[std::pair<unsigned int, unsigned int>(
            cell_coarse->active_fe_index(), cell_fine->active_fe_index())]]
          .n_cells_coarse++;
      });
    }

    for (const auto fe_index_pair : fe_index_pairs)
      {
        this->schemes[fe_index_pair.second].n_cell_dofs_coarse =
          dof_handler_coarse.get_fe(fe_index_pair.first.first).dofs_per_cell;
        this->schemes[fe_index_pair.second].n_cell_dofs_fine =
          dof_handler_fine.get_fe(fe_index_pair.first.second).dofs_per_cell;

        this->schemes[fe_index_pair.second].fine_element_is_continuous =
          dof_handler_fine.get_fe(fe_index_pair.first.second).dofs_per_vertex >
          0;
      }

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
            this->schemes[fe_index_pair.second].n_cell_dofs_coarse);
          local_dof_indices_fine[fe_index_pair.second].resize(
            this->schemes[fe_index_pair.second].n_cell_dofs_fine);

          this->schemes[fe_index_pair.second].level_dof_indices_fine.resize(
            this->schemes[fe_index_pair.second].n_cell_dofs_fine *
            this->schemes[fe_index_pair.second].n_cells_coarse);
          this->schemes[fe_index_pair.second].level_dof_indices_coarse.resize(
            this->schemes[fe_index_pair.second].n_cell_dofs_coarse *
            this->schemes[fe_index_pair.second].n_cells_coarse);


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
            &this->schemes[i].level_dof_indices_coarse[0];
          level_dof_indices_fine_[i] =
            &this->schemes[i].level_dof_indices_fine[0];
        }

      process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
        const auto fe_pair_no =
          fe_index_pairs[std::pair<unsigned int, unsigned int>(
            cell_coarse->active_fe_index(), cell_fine->active_fe_index())];

        cell_coarse->get_dof_indices(local_dof_indices_coarse[fe_pair_no]);
        for (unsigned int i = 0;
             i < this->schemes[fe_pair_no].n_cell_dofs_coarse;
             i++)
          level_dof_indices_coarse_[fe_pair_no][i] =
            this->partitioner_coarse->global_to_local(
              local_dof_indices_coarse
                [fe_pair_no][lexicographic_numbering_coarse[fe_pair_no][i]]);


        cell_fine->get_dof_indices(local_dof_indices_fine[fe_pair_no]);
        for (unsigned int i = 0; i < this->schemes[fe_pair_no].n_cell_dofs_fine;
             i++)
          level_dof_indices_fine_[fe_pair_no][i] =
            this->partitioner_fine->global_to_local(
              local_dof_indices_fine
                [fe_pair_no][lexicographic_numbering_fine[fe_pair_no][i]]);


        level_dof_indices_coarse_[fe_pair_no] +=
          this->schemes[fe_pair_no].n_cell_dofs_coarse;
        level_dof_indices_fine_[fe_pair_no] +=
          this->schemes[fe_pair_no].n_cell_dofs_fine;
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



        FullMatrix<Number> matrix(fe_fine->dofs_per_cell,
                                  fe_coarse->dofs_per_cell);
        FETools::get_projection_matrix(*fe_coarse, *fe_fine, matrix);
        this->schemes[fe_index_no].prolongation_matrix_1d.resize(
          fe_fine->dofs_per_cell * fe_coarse->dofs_per_cell);

        for (unsigned int i = 0, k = 0; i < fe_coarse->dofs_per_cell; ++i)
          for (unsigned int j = 0; j < fe_fine->dofs_per_cell; ++j, ++k)
            this->schemes[fe_index_no].prolongation_matrix_1d[k] =
              matrix(renumbering_fine[j], renumbering_coarse[i]);
      }

    // -------------------------------- weights --------------------------------
    if (this->schemes.front().fine_element_is_continuous)
      {
        for (auto &scheme : this->schemes)
          scheme.weights.resize(scheme.n_cells_coarse *
                                scheme.n_cell_dofs_fine);

        LinearAlgebra::distributed::Vector<Number> touch_count;
        touch_count.reinit(this->partitioner_fine);

        std::vector<unsigned int *> level_dof_indices_fine_(
          fe_index_pairs.size());
        std::vector<Number *> weights_(fe_index_pairs.size());

        for (unsigned int i = 0; i < fe_index_pairs.size(); i++)
          level_dof_indices_fine_[i] =
            &this->schemes[i].level_dof_indices_fine[0];

        process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
          const auto fe_pair_no =
            fe_index_pairs[std::pair<unsigned int, unsigned int>(
              cell_coarse->active_fe_index(), cell_fine->active_fe_index())];

          for (unsigned int i = 0;
               i < this->schemes[fe_pair_no].n_cell_dofs_fine;
               i++)
            if (constraint_fine.is_constrained(
                  this->partitioner_fine->local_to_global(
                    level_dof_indices_fine_[fe_pair_no][i])) == false)
              touch_count.local_element(
                level_dof_indices_fine_[fe_pair_no][i]) += 1;

          level_dof_indices_fine_[fe_pair_no] +=
            this->schemes[fe_pair_no].n_cell_dofs_fine;
        });

        touch_count.compress(VectorOperation::add);
        touch_count.update_ghost_values();

        for (unsigned int i = 0; i < fe_index_pairs.size(); i++)
          {
            level_dof_indices_fine_[i] =
              &this->schemes[i].level_dof_indices_fine[0];
            weights_[i] = &this->schemes[i].weights[0];
          }

        process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
          const auto fe_pair_no =
            fe_index_pairs[std::pair<unsigned int, unsigned int>(
              cell_coarse->active_fe_index(), cell_fine->active_fe_index())];

          for (unsigned int i = 0;
               i < this->schemes[fe_pair_no].n_cell_dofs_fine;
               i++)
            if (constraint_fine.is_constrained(
                  this->partitioner_fine->local_to_global(
                    level_dof_indices_fine_[fe_pair_no][i])) == true)
              weights_[fe_pair_no][i] = Number(0.);
            else
              weights_[fe_pair_no][i] =
                Number(1.) / touch_count.local_element(
                               level_dof_indices_fine_[fe_pair_no][i]);

          level_dof_indices_fine_[fe_pair_no] +=
            this->schemes[fe_pair_no].n_cell_dofs_fine;
          weights_[fe_pair_no] += this->schemes[fe_pair_no].n_cell_dofs_fine;
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
