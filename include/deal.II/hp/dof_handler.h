// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2020 by the deal.II authors
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

#ifndef dealii_hp_dof_handler_h
#define dealii_hp_dof_handler_h

#include <deal.II/dofs/dof_handler.h>

DEAL_II_NAMESPACE_OPEN

namespace hp
{
  template <int dim, int spacedim = dim>
  class DoFHandler : public dealii::DoFHandler<dim, spacedim>
  {
  public:
    DoFHandler();

    DoFHandler(const Triangulation<dim, spacedim> &tria);

    DoFHandler(const DoFHandler &) = delete;

    DoFHandler &
    operator=(const DoFHandler &) = delete;
  };

} // namespace hp

DEAL_II_NAMESPACE_CLOSE

#endif
