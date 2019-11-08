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



// test copy constructor and copy assignment of MatrixFree::AdditionalData

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/renumber.h>

#include "../tests.h"

std::ofstream logfile("output");

template <int dim, typename RunumberFunction>
void
test(const unsigned int fe_degree, RunumberFunction renumber)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(2);

  FE_Q<dim> fe(fe_degree);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const auto &local_dofs = dof_handler.locally_owned_dofs();


  std::vector<unsigned int> numbers_mf_order(
    dof_handler.n_dofs(), dealii::numbers::invalid_unsigned_int);
  unsigned int counter_dof_numbers = 0;

  deallog << std::endl << "DoFs per cell:" << std::endl;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      std::unordered_set<dealii::types::global_dof_index> set_dofs;
      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      for (const auto i : dof_indices)
        deallog << std::right << std::setw(5) << i;
      deallog << std::endl;


      for (const auto i : dof_indices)
        renumber(numbers_mf_order,
                 counter_dof_numbers,
                 set_dofs,
                 local_dofs.is_element(i) ?
                   local_dofs.index_within_set(i) :
                   dealii::numbers::invalid_unsigned_int);
    }
  deallog << std::endl << "New order:" << std::endl;

  for (const auto i : numbers_mf_order)
    deallog << std::right << std::setw(5) << i << std::endl;
  deallog << std::endl;
}

int
main()
{
  initlog();

  const unsigned int fe_degree = 1;

  {
    deallog.push("1d-f");
    test<1>(fe_degree, internal::first_touch_renumber);
    deallog.pop();
  }

  {
    deallog.push("1d-l");
    test<1>(fe_degree, internal::last_touch_renumber);
    deallog.pop();
  }

  {
    deallog.push("2d-f");
    test<2>(fe_degree, internal::first_touch_renumber);
    deallog.pop();
  }

  {
    deallog.push("2d-l");
    test<2>(fe_degree, internal::last_touch_renumber);
    deallog.pop();
  }
}
