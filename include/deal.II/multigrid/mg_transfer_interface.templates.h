// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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


#ifndef dealii_mg_transfer_inteface_templates_h
#define dealii_mg_transfer_inteface_templates_h

#include <deal.II/base/config.h>

#include <deal.II/matrix_free/evaluation_kernels.h>

#include <deal.II/multigrid/mg_transfer_interface.h>

DEAL_II_NAMESPACE_OPEN

namespace
{
  class CellTransfer
  {
  public:
    CellTransfer(const unsigned int degree_fine,
                 const unsigned int degree_coarse)
      : n(100 * degree_fine + degree_coarse)
    {}

    template <typename Fu>
    bool
    run(Fu &fu)
    {
      return do_run_degree<1>(fu) || do_run_degree<2>(fu) ||
             do_run_degree<3>(fu) || do_run_degree<4>(fu) ||
             do_run_degree<5>(fu) || do_run_degree<6>(fu) ||
             do_run_degree<7>(fu) || do_run_degree<8>(fu) ||
             do_run_degree<9>(fu);
    }

    template <int degree, typename Fu>
    bool
    do_run_degree(Fu &fu)
    {
      if (n == 100 * (2 * degree + 1) + degree) // h-MG
        fu.template run<2 * degree + 1, degree>();
      else if (n == 100 * degree + std::max(degree / 2, 1)) // p-MG: bisection
        fu.template run<degree, std::max(degree / 2, 1)>();
      else if (n == 100 * degree + degree) // identity (nothing to do)
        fu.template run<degree, degree>();
      else if (n == 100 * degree + std::max(degree - 1, 1)) // p-MG: --
        fu.template run<degree, std::max(degree - 1, 1)>();
      else if (n == 100 * degree + 1) // p-MG: jump to 1
        fu.template run<degree, 1>();
      else
        return false;

      return true;
    }

    const int n;
  };

  template <int dim, typename Number>
  class CellProlongator
  {
  public:
    CellProlongator(const AlignedVector<Number> &prolongation_matrix_1d,
                    const Number *               evaluation_data_coarse,
                    Number *                     evaluation_data_fine)
      : prolongation_matrix_1d(prolongation_matrix_1d)
      , evaluation_data_coarse(evaluation_data_coarse)
      , evaluation_data_fine(evaluation_data_fine)
    {}

    template <int degree_fine, int degree_coarse>
    void
    run()
    {
      internal::FEEvaluationImplBasisChange<
        internal::evaluate_general,
        dim,
        degree_coarse + 1,
        degree_fine + 1,
        1,
        Number,
        Number>::do_forward(prolongation_matrix_1d,
                            evaluation_data_coarse,
                            evaluation_data_fine);
    }

  private:
    const AlignedVector<Number> &prolongation_matrix_1d;
    const Number *               evaluation_data_coarse;
    Number *                     evaluation_data_fine;
  };

  template <int dim, typename Number>
  class CellRestrictor
  {
  public:
    CellRestrictor(const AlignedVector<Number> &prolongation_matrix_1d,
                   Number *                     evaluation_data_fine,
                   Number *                     evaluation_data_coarse)
      : prolongation_matrix_1d(prolongation_matrix_1d)
      , evaluation_data_fine(evaluation_data_fine)
      , evaluation_data_coarse(evaluation_data_coarse)
    {}

    template <int degree_fine, int degree_coarse>
    void
    run()
    {
      internal::FEEvaluationImplBasisChange<
        internal::evaluate_general,
        dim,
        degree_coarse + 1,
        degree_fine + 1,
        1,
        Number,
        Number>::do_backward(prolongation_matrix_1d,
                             false,
                             evaluation_data_fine,
                             evaluation_data_coarse);
    }

  private:
    const AlignedVector<Number> &prolongation_matrix_1d;
    Number *                     evaluation_data_fine;
    Number *                     evaluation_data_coarse;
  };

} // namespace



template <int dim, typename Number>
void
Transfer<dim, Number>::prolongate(
  const unsigned int                                to_level,
  LinearAlgebra::distributed::Vector<Number> &      dst,
  const LinearAlgebra::distributed::Vector<Number> &src) const
{
  (void)to_level;

  const unsigned int vec_size = VectorizedArray<Number>::n_array_elements;

  this->vec_coarse.copy_locally_owned_data_from(src);
  this->vec_coarse.update_ghost_values();
  this->constraint_coarse.distribute(this->vec_coarse);
  this->vec_coarse
    .update_ghost_values(); // note: make sure that ghost values are set

  this->vec_fine = Number(0.);

  for (const auto &scheme : schemes)
    {
      AlignedVector<VectorizedArray<Number>> evaluation_data_fine(
        scheme.n_cell_dofs_fine);
      AlignedVector<VectorizedArray<Number>> evaluation_data_coarse(
        scheme.n_cell_dofs_fine);

      CellProlongator<dim, VectorizedArray<Number>> cell_prolongator(
        scheme.prolongation_matrix_1d,
        evaluation_data_coarse.begin(),
        evaluation_data_fine.begin());
      CellTransfer cell_transfer(scheme.degree_fine, scheme.degree_coarse);

      for (unsigned int cell = 0; cell < scheme.n_cells_coarse;
           cell += vec_size)
        {
          const unsigned int n_lanes = cell + vec_size > scheme.n_cells_coarse ?
                                         scheme.n_cells_coarse - cell :
                                         vec_size;

          // read from source vector
          {
            const unsigned int *indices =
              &scheme
                 .level_dof_indices_coarse[cell * scheme.n_cell_dofs_coarse];
            for (unsigned int v = 0; v < n_lanes; ++v)
              {
                for (unsigned int i = 0; i < scheme.n_cell_dofs_coarse; ++i)
                  evaluation_data_coarse[i][v] =
                    this->vec_coarse.local_element(indices[i]);
                indices += scheme.n_cell_dofs_coarse;
              }
          }

          // ---------------------------- coarse -----------------------------
          const auto ierr = cell_transfer.run(cell_prolongator);
          (void)ierr;
          Assert(ierr,
                 ExcMessage("Prolongation " +
                            std::to_string(scheme.degree_coarse) + " -> " +
                            std::to_string(scheme.degree_fine) +
                            " not instantiated!"));
          // ------------------------------ fine -----------------------------

          if (scheme.fine_element_is_continuous)
            {
              const Number *w = &scheme.weights[cell * scheme.n_cell_dofs_fine];
              for (unsigned int v = 0; v < n_lanes; ++v)
                {
                  for (unsigned int i = 0; i < scheme.n_cell_dofs_fine; ++i)
                    evaluation_data_fine[i][v] *= w[i];
                  w += scheme.n_cell_dofs_fine;
                }
            }


          // write into dst vector
          {
            const unsigned int *indices =
              &scheme.level_dof_indices_fine[cell * scheme.n_cell_dofs_fine];
            for (unsigned int v = 0; v < n_lanes; ++v)
              {
                for (unsigned int i = 0; i < scheme.n_cell_dofs_fine; ++i)
                  this->vec_fine.local_element(indices[i]) +=
                    evaluation_data_fine[i][v];
                indices += scheme.n_cell_dofs_fine;
              }
          }
        }
    }

  this->vec_coarse.zero_out_ghosts(); // clear ghost values; else compress in
                                      // do_restrict_add does not work

  if (schemes.size() > 0 && schemes.front().fine_element_is_continuous)
    this->vec_fine.compress(VectorOperation::add);

  dst.copy_locally_owned_data_from(this->vec_fine);
}



template <int dim, typename Number>
void
Transfer<dim, Number>::restrict_and_add(
  const unsigned int                                from_level,
  LinearAlgebra::distributed::Vector<Number> &      dst,
  const LinearAlgebra::distributed::Vector<Number> &src) const
{
  (void)from_level;

  const unsigned int vec_size = VectorizedArray<Number>::n_array_elements;

  this->vec_fine.copy_locally_owned_data_from(src);
  this->vec_fine.update_ghost_values();

  this->vec_coarse.copy_locally_owned_data_from(dst);

  for (const auto &scheme : schemes)
    {
      AlignedVector<VectorizedArray<Number>> evaluation_data_fine(
        scheme.n_cell_dofs_fine);
      AlignedVector<VectorizedArray<Number>> evaluation_data_coarse(
        scheme.n_cell_dofs_fine);

      CellRestrictor<dim, VectorizedArray<Number>> cell_restrictor(
        scheme.prolongation_matrix_1d,
        evaluation_data_fine.begin(),
        evaluation_data_coarse.begin());
      CellTransfer cell_transfer(scheme.degree_fine, scheme.degree_coarse);

      for (unsigned int cell = 0; cell < scheme.n_cells_coarse;
           cell += vec_size)
        {
          const unsigned int n_lanes = cell + vec_size > scheme.n_cells_coarse ?
                                         scheme.n_cells_coarse - cell :
                                         vec_size;

          // read from source vector
          {
            const unsigned int *indices =
              &scheme.level_dof_indices_fine[cell * scheme.n_cell_dofs_fine];
            for (unsigned int v = 0; v < n_lanes; ++v)
              {
                for (unsigned int i = 0; i < scheme.n_cell_dofs_fine; ++i)
                  evaluation_data_fine[i][v] =
                    this->vec_fine.local_element(indices[i]);
                indices += scheme.n_cell_dofs_fine;
              }
          }

          if (scheme.fine_element_is_continuous)
            {
              const Number *w = &scheme.weights[cell * scheme.n_cell_dofs_fine];
              for (unsigned int v = 0; v < n_lanes; ++v)
                {
                  for (unsigned int i = 0; i < scheme.n_cell_dofs_fine; ++i)
                    evaluation_data_fine[i][v] *= w[i];
                  w += scheme.n_cell_dofs_fine;
                }
            }

          // ------------------------------ fine -----------------------------
          const auto ierr = cell_transfer.run(cell_restrictor);
          (void)ierr;
          Assert(ierr,
                 ExcMessage("Restriction " +
                            std::to_string(scheme.degree_fine) + " -> " +
                            std::to_string(scheme.degree_coarse) +
                            " not instantiated!"));
          // ----------------------------- coarse ----------------------------


          // write into dst vector
          {
            const unsigned int *indices =
              &scheme
                 .level_dof_indices_coarse[cell * scheme.n_cell_dofs_coarse];
            for (unsigned int v = 0; v < n_lanes; ++v)
              {
                for (unsigned int i = 0; i < scheme.n_cell_dofs_coarse; ++i)
                  constraint_coarse.distribute_local_to_global(
                    partitioner_coarse->local_to_global(indices[i]),
                    evaluation_data_coarse[i][v],
                    this->vec_coarse);
                indices += scheme.n_cell_dofs_coarse;
              }
          }
        }
    }

  if (schemes.size() && schemes.front().fine_element_is_continuous)
    this->vec_coarse.compress(VectorOperation::add);

  dst.copy_locally_owned_data_from(this->vec_coarse);
}



template <int dim, typename Number>
void
VectorRepartitioner<dim, Number>::update_forwards(
  LinearAlgebra::distributed::Vector<Number> &      dst,
  const LinearAlgebra::distributed::Vector<Number> &src) const
{
  // create new source vector with matching ghost values
  LinearAlgebra::distributed::Vector<Number> src_extended(extended_partitioner);

  // copy locally owned values from original source vector
  src_extended.copy_locally_owned_data_from(src);

  // update ghost values
  src_extended.update_ghost_values();

  // copy locally owned values from temporal array to destination vector
  for (unsigned int i = 0; i < indices.size(); ++i)
    dst.local_element(i) = src_extended.local_element(indices[i]);
}

template <int dim, typename Number>
void
VectorRepartitioner<dim, Number>::update_backwards(
  LinearAlgebra::distributed::Vector<Number> &      dst,
  const LinearAlgebra::distributed::Vector<Number> &src) const
{
  // create new source vector with matching ghost values
  LinearAlgebra::distributed::Vector<Number> dst_extended(extended_partitioner);

  // copy locally owned values from temporal array to destination vector
  for (unsigned int i = 0; i < indices.size(); ++i)
    dst_extended.local_element(indices[i]) = src.local_element(i);

  // update ghost values
  dst_extended.compress(VectorOperation::values::add);

  // copy locally owned values from original source vector
  dst.copy_locally_owned_data_from(dst_extended);
}


DEAL_II_NAMESPACE_CLOSE

#endif
