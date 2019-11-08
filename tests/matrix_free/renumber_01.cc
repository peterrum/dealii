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

template <int dim, unsigned int group_size, typename RunumberFunction>
void
do_test(const unsigned int fe_degree, RunumberFunction renumber_algo)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(2);

  FE_Q<dim> fe(fe_degree);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  const auto &local_dofs = dof_handler.locally_owned_dofs();


  std::vector<unsigned int> new_iterator_order(
    dof_handler.n_dofs(), dealii::numbers::invalid_unsigned_int);
  unsigned int counter_dof_numbers = 0;

  deallog << std::endl << "DoFs per cell:" << std::endl;

  auto cell = dof_handler.begin_active();

  while (cell != dof_handler.end())
    {
      std::vector<std::array<types::global_dof_index, group_size>>
        dof_indices_grouped(fe.dofs_per_cell,
                            {dealii::numbers::invalid_unsigned_int});

      for (unsigned int v = 0; v < group_size && cell != dof_handler.end();
           v++, cell++)
        {
          // get indices of this cell
          std::vector<types::global_dof_index> dof_indices_local(
            fe.dofs_per_cell);
          cell->get_dof_indices(dof_indices_local);

          // store indices vectorized
          for (unsigned int i = 0; i < dof_indices_local.size(); i++)
            dof_indices_grouped[i][v] = dof_indices_local[i];

          // print indices for
          for (const auto i : dof_indices_local)
            deallog << std::right << std::setw(5) << i;
          deallog << std::endl;
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
  deallog << std::endl << "New order:" << std::endl;

  for (const auto i : new_iterator_order)
    deallog << std::right << std::setw(5) << i << std::endl;
  deallog << std::endl;
}

template <int dim, typename RunumberFunction>
void
test(const unsigned int fe_degree, RunumberFunction renumber)
{
  do_test<dim, 1, RunumberFunction>(fe_degree, renumber);

  do_test<dim, 4, RunumberFunction>(fe_degree, renumber);
}

int
main()
{
  initlog();

  const unsigned int fe_degree = 2;

  deallog.push("1d");
  {
    deallog.push("first");
    test<1>(fe_degree, internal::FirstTouch());
    deallog.pop();
  }

  {
    deallog.push("last");
    test<1>(fe_degree, internal::LastTouch());
    deallog.pop();
  }
  deallog.pop();

  deallog.push("2d");
  {
    deallog.push("first");
    test<2>(fe_degree, internal::FirstTouch());
    deallog.pop();
  }

  {
    deallog.push("last");
    test<2>(fe_degree, internal::LastTouch());
    deallog.pop();
  }
  deallog.pop();
}
