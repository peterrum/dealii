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
    static void
    renumber()
    {}
  };

} // namespace internal

DEAL_II_NAMESPACE_CLOSE

#endif
