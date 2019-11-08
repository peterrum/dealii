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
  // initialize system
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(2);
  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  deallog << std::endl << "DoFs per cell:" << std::endl;
  for (const auto cell : dof_handler.active_cell_iterators())
    {
      // get indices of this cell
      std::vector<types::global_dof_index> dof_indices_local(fe.dofs_per_cell);
      cell->get_dof_indices(dof_indices_local);

      // print indices for
      for (const auto i : dof_indices_local)
        deallog << std::right << std::setw(5) << i;
      deallog << std::endl;
    }

  const auto new_iterator_order =
    internal::Assembly::renumber<group_size>(dof_handler, renumber_algo);


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
