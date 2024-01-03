// ---------------------------------------------------------------------
//
// Copyright (C) 2009 - 2022 by the deal.II authors
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

#ifndef dealii_distributed_solution_transfer_h
#define dealii_distributed_solution_transfer_h

#include <deal.II/base/config.h>

#include <deal.II/numerics/solution_transfer.h>


DEAL_II_NAMESPACE_OPEN

namespace parallel
{
  namespace distributed
  {
    DEAL_II_DEPRECATED_EARLY
    template <int dim, typename VectorType, int spacedim = dim>
    using SolutionTransfer =
      dealii::SolutionTransfer<dim, VectorType, spacedim>;
  } // namespace distributed
} // namespace parallel


DEAL_II_NAMESPACE_CLOSE

#endif
