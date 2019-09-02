// ---------------------------------------------------------------------
//
// Copyright (C) 2008 - 2019 by the deal.II authors
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

#ifndef dealii_fully_distributed_tria_util_h
#define dealii_fully_distributed_tria_util_h

#include <deal.II/distributed/fully_distributed_tria.h>

DEAL_II_NAMESPACE_OPEN

namespace parallel
{
  namespace fullydistributed
  {
    namespace Utilities
    {
      template <int dim, int spacedim = dim>
      ConstructionData<dim, spacedim>
      copy_from_triangulation(
        const dealii::Triangulation<dim, spacedim> &tria,
        const Triangulation<dim, spacedim> &        tria_pft,
        const unsigned int my_rank_in = numbers::invalid_unsigned_int);

    } // namespace Utilities
  }   // namespace fullydistributed
} // namespace parallel


DEAL_II_NAMESPACE_CLOSE

#endif
