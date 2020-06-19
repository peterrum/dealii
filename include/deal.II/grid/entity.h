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

  inline std_cxx20::ranges::iota_view<unsigned int, unsigned int>
  vertex_indices() const
  {
    return {0U, n_vertices()};
  }

  /**
   * TODO
   */
  inline std_cxx20::ranges::iota_view<unsigned int, unsigned int>
  line_indices() const
  {
    return {0U, n_lines()};
  }

  /**
   * @note Only implemented for cells.
   */
  inline std_cxx20::ranges::iota_view<unsigned int, unsigned int>
  face_indices() const
  {
    return {0U, n_faces()};
  }

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
  standard_to_real_face_line(const unsigned int  line,
                             const unsigned char face_orientation) const
  {
    Assert(false, ExcNotImplemented());

    (void)line;
    (void)face_orientation;

    return 0;
  }

  virtual bool
  combine_quad_and_line_orientation(const unsigned int  line,
                                    const unsigned char face_orientation,
                                    const bool          line_orientation) const
  {
    Assert(false, ExcNotImplemented());

    (void)line;
    (void)face_orientation;
    (void)line_orientation;

    return true;
  }

  virtual std::array<unsigned int, 2>
  standard_hex_vertex_to_quad_vertex_index(const unsigned int vertex) const
  {
    Assert(false, ExcNotImplemented());

    (void)vertex;

    return {0, 0};
  }

  virtual unsigned int
  standard_to_real_face_vertex(const unsigned int  vertex,
                               const unsigned char face_orientation) const
  {
    Assert(false, ExcNotImplemented());

    (void)vertex;
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

    static const std::array<std::array<unsigned int, 2>, 3> table = {
      {{0, 0}, {0, 1}, {1, 1}}};

    return table[vertex];
  }

  unsigned int
  standard_to_real_line_vertex(const unsigned int vertex,
                               const bool line_orientation) const override
  {
    static const std::array<std::array<unsigned int, 2>, 2> table = {
      {{0, 1}, {1, 0}}};

    return table[line_orientation][vertex];
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
  std::array<unsigned int, 2>
  standard_hex_line_to_quad_line_index(const unsigned int line) const override
  {
    static const std::array<unsigned int, 2> table[6] = {
      {0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 1}};
    // static const std::array<unsigned int, 2> table[6] = {
    //  {0, 0}, {0, 2}, {1, 2}, {0, 1}, {1, 1}, {2, 1}};

    return table[line];
  }

  unsigned int
  standard_to_real_face_line(
    const unsigned int  line,
    const unsigned char face_orientation) const override
  {
    static const std::array<std::array<unsigned int, 3>, 6> table = {
      {{0, 1, 2}, {2, 1, 0}, {0, 2, 1}, {1, 2, 0}, {2, 0, 1}, {1, 0, 2}}};

    return table[face_orientation][line];
  }

  bool
  combine_quad_and_line_orientation(const unsigned int  line,
                                    const unsigned char face_orientation,
                                    const bool line_orientation) const override
  {
    (void)line;
    (void)face_orientation;

    deallog << line << " " << static_cast<unsigned int>(face_orientation) << " "
            << static_cast<unsigned int>(line_orientation) << std::endl;

    return line_orientation; // TODO
  }

  std::array<unsigned int, 2>
  standard_hex_vertex_to_quad_vertex_index(
    const unsigned int vertex) const override
  {
    AssertIndexRange(vertex, 4);

    static const std::array<unsigned int, 2> table[4] = {{0, 0},
                                                         {0, 1},
                                                         {0, 2},
                                                         {1, 2}};

    return table[vertex];
  }

  unsigned int
  standard_to_real_face_vertex(
    const unsigned int  vertex,
    const unsigned char face_orientation) const override
  {
    AssertIndexRange(face_orientation, 6);

    // static const std::array<std::array<unsigned int, 3>, 6> table = {
    //  {{0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}, {0, 1, 2}}};

    static const std::array<std::array<unsigned int, 3>, 6> table = {
      {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}}};

    return table[face_orientation][vertex];
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
  standard_to_real_face_line(
    const unsigned int  line,
    const unsigned char face_orientation) const override
  {
    return GeometryInfo<3>::standard_to_real_face_line(
      line,
      get_bit(face_orientation, 0),
      get_bit(face_orientation, 1),
      get_bit(face_orientation, 2));
  }

  bool
  combine_quad_and_line_orientation(const unsigned int  line,
                                    const unsigned char face_orientation_raw,
                                    const bool line_orientation) const override
  {
    static const bool bool_table[2][2][2][2] = {
      {{{true, false},    // lines 0/1, face_orientation=false,
                          // face_flip=false, face_rotation=false and true
        {false, true}},   // lines 0/1, face_orientation=false,
                          // face_flip=true, face_rotation=false and true
       {{true, true},     // lines 0/1, face_orientation=true, face_flip=false,
                          // face_rotation=false and true
        {false, false}}}, // lines 0/1, face_orientation=true,
                          // face_flip=true, face_rotation=false and true

      {{{true, true}, // lines 2/3 ...
        {false, false}},
       {{true, false}, {false, true}}}};

    const bool face_orientation = get_bit(face_orientation_raw, 0);
    const bool face_flip        = get_bit(face_orientation_raw, 1);
    const bool face_rotation    = get_bit(face_orientation_raw, 2);

    return (line_orientation ==
            bool_table[line / 2][face_orientation][face_flip][face_rotation]);
  }

  std::array<unsigned int, 2>
  standard_hex_vertex_to_quad_vertex_index(
    const unsigned int vertex) const override
  {
    return GeometryInfo<3>::standard_hex_vertex_to_quad_vertex_index(vertex);
  }

  unsigned int
  standard_to_real_face_vertex(
    const unsigned int  vertex,
    const unsigned char face_orientation) const override
  {
    return GeometryInfo<3>::standard_to_real_face_vertex(
      vertex,
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

enum EntityType
{
  none,
  vertex,
  line,
  tri,
  quad,
  tet,
  hex
};

inline const DynamicGeometryInfo &
create_dynamic_geometry_info(const EntityType label)
{
  static DynamicGeometryInfoVertex o_vertex;
  static DynamicGeometryInfoLine   o_line;
  static DynamicGeometryInfoTri    o_tri;
  static DynamicGeometryInfoQuad   o_quad;
  static DynamicGeometryInfoTet    o_tet;
  static DynamicGeometryInfoHex    o_hex;

  if (label == EntityType::vertex)
    return o_vertex;
  if (label == EntityType::line)
    return o_line;
  if (label == EntityType::tri)
    return o_tri;
  if (label == EntityType::quad)
    return o_quad;
  if (label == EntityType::tet)
    return o_tet;
  if (label == EntityType::hex)
    return o_hex;

  Assert(false, ExcNotImplemented())

    return o_hex;
}

inline const DynamicGeometryInfo &
create_dynamic_geometry_info(const std::string &label)
{
  if (label == "vertex")
    return create_dynamic_geometry_info(EntityType::vertex);
  if (label == "line")
    return create_dynamic_geometry_info(EntityType::line);
  if (label == "tri")
    return create_dynamic_geometry_info(EntityType::tri);
  if (label == "quad")
    return create_dynamic_geometry_info(EntityType::quad);
  if (label == "tet")
    return create_dynamic_geometry_info(EntityType::tet);
  if (label == "hex")
    return create_dynamic_geometry_info(EntityType::hex);

  Assert(false, ExcNotImplemented())

    return create_dynamic_geometry_info(EntityType::none);
}

DEAL_II_NAMESPACE_CLOSE

#endif
