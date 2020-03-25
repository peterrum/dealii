// ---------------------------------------------------------------------
//
// Copyright (C) 1999 - 2019 by the deal.II authors
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

#ifndef dealii_dof_handler_base_templates_h
#define dealii_dof_handler_base_templates_h

#include <deal.II/dofs/dof_handler_base.h>

DEAL_II_NAMESPACE_OPEN


template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::set_fe(const FiniteElement<dim, spacedim> &fe)
{
  this->set_fe(hp::FECollection<dim, spacedim>(fe));
}

template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::set_fe(
  const hp::FECollection<dim, spacedim> &fe)
{
  static_cast<T *>(this)->set_fe_impl(hp::FECollection<dim, spacedim>(fe));
}


DEAL_II_NAMESPACE_CLOSE

#endif
