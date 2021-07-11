// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#ifndef dealii_matrix_free_hanging_nodes_internal_h
#define dealii_matrix_free_hanging_nodes_internal_h

#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  // Here is the system for how we store constraint types in a binary mask.
  // This is not a complete contradiction-free system, i.e., there are
  // invalid states that we just assume that we never get.

  // If the mask is zero, there are no constraints. Then, there are three
  // different fields with one bit per dimension. The first field determines
  // the type, or the position of an element along each direction. The
  // second field determines if there is a constrained face with that
  // direction as normal. The last field determines if there is a
  // constrained edge of a given pair of coordinate planes, but where
  // neither of the corresponding faces are constrained (only valid in 3D).

  // The element is placed in the 'first position' along *-axis. These also
  // determine which face is constrained. For example, in 2D, if
  // constr_face_x and constr_type_x are set, then x = 0 is constrained.
  constexpr unsigned int constr_type_x = 1 << 0;
  constexpr unsigned int constr_type_y = 1 << 1;
  constexpr unsigned int constr_type_z = 1 << 2;

  // Element has as a constraint at * = 0 or * = fe_degree face
  constexpr unsigned int constr_face_x = 1 << 3;
  constexpr unsigned int constr_face_y = 1 << 4;
  constexpr unsigned int constr_face_z = 1 << 5;

  // Element has as a constraint at * = 0 or * = fe_degree edge
  constexpr unsigned int constr_edge_xy = 1 << 6;
  constexpr unsigned int constr_edge_yz = 1 << 7;
  constexpr unsigned int constr_edge_zx = 1 << 8;

} // namespace internal

DEAL_II_NAMESPACE_CLOSE

#endif
