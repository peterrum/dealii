// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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

#ifndef dealii_tria_geometric_entity_h
#define dealii_tria_geometric_entity_h

#include <deal.II/base/config.h>

#include <deal.II/base/types.h>


DEAL_II_NAMESPACE_OPEN


/**
 * Supported geometric entity types.
 */
enum class GeometricEntityType : types::geometric_entity_type
{
  VERTEX  = 0,
  LINE    = 1,
  TRI     = 2,
  QUAD    = 3,
  TET     = 4,
  PYRAMID = 5,
  WEDGE   = 6,
  HEX     = 7,
  INVALID = static_cast<std::uint8_t>(-1)
};


DEAL_II_NAMESPACE_CLOSE

#endif
