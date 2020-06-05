// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2020 by the deal.II authors
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

#ifndef dealii_tria_entity_h
#define dealii_tria_entity_h


#include <deal.II/base/config.h>

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/tria_iterator_base.h>
#include <deal.II/grid/tria_iterator_selector.h>

#include <utility>


DEAL_II_NAMESPACE_OPEN

struct DynamicGeometryInfo
{
  virtual unsigned int
  n_vertices() const = 0;

  virtual unsigned int
  n_lines() const = 0;

  virtual unsigned int
  n_faces() const = 0;

  virtual std::array<unsigned int, 2>
  standard_quad_vertex_to_line_vertex_index(const unsigned int vertex) const
  {
    Assert(false, ExcNotImplemented());

    (void)vertex;

    return {0u, 0u};
  }

  virtual unsigned int
  standard_to_real_line_vertex(const unsigned int vertex,
                               const bool         line_orientation) const
  {
    Assert(false, ExcNotImplemented());

    (void)vertex;
    (void)line_orientation;

    return 0;
  }


  virtual std::array<unsigned int, 2>
  standard_hex_line_to_quad_line_index(const unsigned int line) const
  {
    Assert(false, ExcNotImplemented());

    (void)line;

    return {0, 0};
  }

  virtual unsigned int
  standard_to_real_face_line(const unsigned int line,
                             const unsigned int face_orientation) const
  {
    Assert(false, ExcNotImplemented());

    (void)line;
    (void)face_orientation;

    return 0;
  }
};



template <int dim>
struct DynamicGeometryInfoTensor : DynamicGeometryInfo
{
  unsigned int
  n_vertices() const override
  {
    return GeometryInfo<dim>::vertices_per_cell;
  }

  unsigned int
  n_lines() const override
  {
    return GeometryInfo<dim>::lines_per_cell;
  }

  unsigned int
  n_faces() const override
  {
    return GeometryInfo<dim>::faces_per_cell;
  }
};



/*
 * VERTEX
 */
struct DynamicGeometryInfoVertex : public DynamicGeometryInfoTensor<0>
{};



/*
 * LINE
 */
struct DynamicGeometryInfoLine : public DynamicGeometryInfoTensor<1>
{};



/**
 * TRI
 */
struct DynamicGeometryInfoTri : DynamicGeometryInfo
{
  unsigned int
  n_vertices() const override
  {
    return 3;
  }

  unsigned int
  n_lines() const override
  {
    return 3;
  }

  unsigned int
  n_faces() const override
  {
    return this->n_lines();
  }

  std::array<unsigned int, 2>
  standard_quad_vertex_to_line_vertex_index(
    const unsigned int vertex) const override
  {
    AssertIndexRange(vertex, 3);

    static const std::array<unsigned int, 2> table[3] = {{0, 0},
                                                         {0, 1},
                                                         {1, 1}}; // TODO

    return table[vertex];
  }

  unsigned int
  standard_to_real_line_vertex(const unsigned int vertex,
                               const bool line_orientation) const override
  {
    (void)line_orientation; // TODO

    return vertex;
  }
};



/**
 * QUAD
 */
struct DynamicGeometryInfoQuad : public DynamicGeometryInfoTensor<2>
{
  std::array<unsigned int, 2>
  standard_quad_vertex_to_line_vertex_index(
    const unsigned int vertex) const override
  {
    return GeometryInfo<2>::standard_quad_vertex_to_line_vertex_index(vertex);
  }

  unsigned int
  standard_to_real_line_vertex(const unsigned int vertex,
                               const bool line_orientation) const override
  {
    return GeometryInfo<2>::standard_to_real_line_vertex(vertex,
                                                         line_orientation);
  }
};



/**
 * TET
 */
struct DynamicGeometryInfoTet : DynamicGeometryInfo
{
  unsigned int
  n_vertices() const override
  {
    return 4;
  }

  unsigned int
  n_lines() const override
  {
    return 6;
  }

  unsigned int
  n_faces() const override
  {
    return 4;
  }
};



/**
 * HEX
 */
struct DynamicGeometryInfoHex : public DynamicGeometryInfoTensor<3>
{
  std::array<unsigned int, 2>
  standard_hex_line_to_quad_line_index(const unsigned int line) const override
  {
    return GeometryInfo<3>::standard_hex_line_to_quad_line_index(line);
  }

  unsigned int
  standard_to_real_face_line(const unsigned int line,
                             const unsigned int face_orientation) const override
  {
    return GeometryInfo<3>::standard_to_real_face_line(
      line,
      get_bit(face_orientation, 0),
      get_bit(face_orientation, 1),
      get_bit(face_orientation, 2));
  }

private:
  /**
   * Check if the bit at position @p n in @p number is set.
   */
  inline static bool
  get_bit(const unsigned char number, const unsigned int n)
  {
    AssertIndexRange(n, 8);

    // source:
    // https://stackoverflow.com/questions/47981/how-do-you-set-clear-and-toggle-a-single-bit
    // "Checking a bit"
    return (number >> n) & 1U;
  }
};



DEAL_II_NAMESPACE_CLOSE

#endif
