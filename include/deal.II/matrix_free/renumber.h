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


#ifndef dealii_matrix_free_renumber_h
#define dealii_matrix_free_renumber_h

#include <unordered_set>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  class FirstTouch
  {
  public:
    static void
    renumber(std::vector<unsigned int> &numbers_mf_order,
             unsigned int &             counter_dof_numbers,               //?
             std::unordered_set<dealii::types::global_dof_index> set_dofs, //?
             const unsigned int &index_within_set)
    {
      (void)set_dofs;

      if (index_within_set != dealii::numbers::invalid_unsigned_int &&
          numbers_mf_order[index_within_set] ==
            dealii::numbers::invalid_unsigned_int)
        numbers_mf_order[index_within_set] = counter_dof_numbers++;
    }
  };

  class LastTouch
  {
  public:
    static void
    renumber(std::vector<unsigned int> &numbers_mf_order,
             unsigned int &             counter_dof_numbers,               //?
             std::unordered_set<dealii::types::global_dof_index> set_dofs, //?
             const unsigned int &index_within_set)
    {
      if (index_within_set != dealii::numbers::invalid_unsigned_int &&
          set_dofs.find(index_within_set) == set_dofs.end())
        {
          numbers_mf_order[index_within_set] = counter_dof_numbers++;
          set_dofs.emplace(index_within_set);
        }
    }
  };

  class Assembly
  {
  public:
    template <unsigned int group_size, int dim, typename RunumberFunction>
    static std::vector<unsigned int>
    renumber(const DoFHandler<dim> &dof_handler, RunumberFunction renumber_algo)
    {
      const auto &local_dofs = dof_handler.locally_owned_dofs();

      std::vector<unsigned int> new_iterator_order(
        dof_handler.n_dofs(), dealii::numbers::invalid_unsigned_int);
      unsigned int counter_dof_numbers = 0;

      auto cell = dof_handler.begin_active();
      while (cell != dof_handler.end())
        {
          std::vector<std::array<types::global_dof_index, group_size>>
            dof_indices_grouped(dof_handler.get_fe().dofs_per_cell,
                                {dealii::numbers::invalid_unsigned_int});

          for (unsigned int v = 0; v < group_size && cell != dof_handler.end();
               v++, cell++)
            {
              // get indices of this cell
              std::vector<types::global_dof_index> dof_indices_local(
                dof_handler.get_fe().dofs_per_cell);
              cell->get_dof_indices(dof_indices_local);

              // store indices vectorized
              for (unsigned int i = 0; i < dof_indices_local.size(); i++)
                dof_indices_grouped[i][v] = dof_indices_local[i];
            }

          std::unordered_set<dealii::types::global_dof_index> set_dofs;

          for (const auto dof_indices : dof_indices_grouped)
            for (const auto dof_index : dof_indices)
              renumber_algo.renumber(new_iterator_order,
                                     counter_dof_numbers,
                                     set_dofs,
                                     local_dofs.is_element(dof_index) ?
                                       local_dofs.index_within_set(dof_index) :
                                       dealii::numbers::invalid_unsigned_int);
        }

      return new_iterator_order;
    }
  };

} // namespace internal

DEAL_II_NAMESPACE_CLOSE

#endif
