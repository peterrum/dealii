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

#ifndef dealii_mg_transfer_interface_h
#define dealii_mg_transfer_interface_h

#include <deal.II/multigrid/mg_base.h>

DEAL_II_NAMESPACE_OPEN

template <typename Number>
struct TransferScheme
{
  unsigned int n_cells_coarse;

  unsigned int n_cell_dofs_coarse;
  unsigned int n_cell_dofs_fine;

  bool                fine_element_is_continuous;
  std::vector<Number> weights;

  AlignedVector<VectorizedArray<Number>> prolongation_matrix_1d;

  std::vector<unsigned int> level_dof_indices_coarse;
  std::vector<unsigned int> level_dof_indices_fine;

  void
  print() const
  {
    std::cout << "weights:" << std::endl;
    for (const auto w : weights)
      std::cout << w << " ";
    std::cout << std::endl;

    std::cout << "level_dof_indices_fine:" << std::endl;
    for (const auto w : level_dof_indices_fine)
      std::cout << w << " ";
    std::cout << std::endl;

    std::cout << "level_dof_indices_coarse:" << std::endl;
    for (const auto w : level_dof_indices_coarse)
      std::cout << w << " ";
    std::cout << std::endl;

    std::cout << "prolongation_matrix_1d:" << std::endl;
    for (const auto w : prolongation_matrix_1d)
      std::cout << w[0] << " ";
    std::cout << std::endl;
  }
};

template <int dim, typename Number>
class Transfer
  : public MGTransferBase<LinearAlgebra::distributed::Vector<Number>>
{
public:
  void
  print_internal() const
  {
    for (const auto &scheme : schemes)
      scheme.print();
  }

protected:
  template <int degree_fine, int degree_coarse>
  void
  do_prolongate_add(const unsigned int                                to_level,
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

        for (unsigned int cell = 0; cell < scheme.n_cells_coarse;
             cell += vec_size)
          {
            const unsigned int n_lanes =
              cell + vec_size > scheme.n_cells_coarse ?
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

            // ------------------------------ coarse
            // -------------------------------
            if (scheme.n_cell_dofs_fine ==
                scheme.n_cell_dofs_coarse) // TODO: create jump table
              {
                internal::FEEvaluationImplBasisChange<
                  internal::evaluate_general,
                  dim,
                  degree_coarse + 1,
                  degree_coarse + 1,
                  1,
                  VectorizedArray<Number>,
                  VectorizedArray<Number>>::
                  do_forward(scheme.prolongation_matrix_1d,
                             evaluation_data_coarse.begin(),
                             evaluation_data_fine.begin());
              }
            else
              {
                internal::FEEvaluationImplBasisChange<
                  internal::evaluate_general,
                  dim,
                  degree_coarse + 1,
                  degree_fine + 1,
                  1,
                  VectorizedArray<Number>,
                  VectorizedArray<Number>>::
                  do_forward(scheme.prolongation_matrix_1d,
                             evaluation_data_coarse.begin(),
                             evaluation_data_fine.begin());
              }
            // -------------------------------- fine
            // -------------------------------

            if (scheme.fine_element_is_continuous)
              {
                const Number *w =
                  &scheme.weights[cell * scheme.n_cell_dofs_fine];
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

    if (schemes.front().fine_element_is_continuous)
      this->vec_fine.compress(VectorOperation::add);

    dst.copy_locally_owned_data_from(this->vec_fine);
  }

  template <int degree_fine, int degree_coarse>
  void
  do_restrict_add(const unsigned int                                from_level,
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

        for (unsigned int cell = 0; cell < scheme.n_cells_coarse;
             cell += vec_size)
          {
            const unsigned int n_lanes =
              cell + vec_size > scheme.n_cells_coarse ?
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
                const Number *w =
                  &scheme.weights[cell * scheme.n_cell_dofs_fine];
                for (unsigned int v = 0; v < n_lanes; ++v)
                  {
                    for (unsigned int i = 0; i < scheme.n_cell_dofs_fine; ++i)
                      evaluation_data_fine[i][v] *= w[i];
                    w += scheme.n_cell_dofs_fine;
                  }
              }

            // ------------------------------ fine -----------------------------
            if (scheme.n_cell_dofs_fine ==
                scheme.n_cell_dofs_coarse) // TODO: create jump table
              {
                internal::FEEvaluationImplBasisChange<
                  internal::evaluate_general,
                  dim,
                  degree_coarse + 1,
                  degree_coarse + 1,
                  1,
                  VectorizedArray<Number>,
                  VectorizedArray<Number>>::
                  do_backward(scheme.prolongation_matrix_1d,
                              false,
                              evaluation_data_fine.begin(),
                              evaluation_data_coarse.begin());
              }
            else
              {
                internal::FEEvaluationImplBasisChange<
                  internal::evaluate_general,
                  dim,
                  degree_coarse + 1,
                  degree_fine + 1,
                  1,
                  VectorizedArray<Number>,
                  VectorizedArray<Number>>::
                  do_backward(scheme.prolongation_matrix_1d,
                              false,
                              evaluation_data_fine.begin(),
                              evaluation_data_coarse.begin());
              }
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

    if (schemes.front().fine_element_is_continuous)
      this->vec_coarse.compress(VectorOperation::add);

    dst.copy_locally_owned_data_from(this->vec_coarse);
  }

  std::vector<TransferScheme<Number>> schemes;

  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_fine;
  std::shared_ptr<const Utilities::MPI::Partitioner> partitioner_coarse;

  mutable LinearAlgebra::distributed::Vector<Number> vec_fine;
  mutable LinearAlgebra::distributed::Vector<Number> vec_coarse;

  AffineConstraints<Number> constraint_coarse;
};

DEAL_II_NAMESPACE_CLOSE

#endif
