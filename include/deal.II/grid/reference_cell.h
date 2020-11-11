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

#ifndef dealii_tria_reference_cell_h
#define dealii_tria_reference_cell_h

#include <deal.II/base/config.h>

#include <deal.II/base/geometry_info.h>


DEAL_II_NAMESPACE_OPEN


/**
 * A namespace for reference cells.
 */
namespace ReferenceCell
{
  /**
   * Supported reference cell types.
   */
  enum class Type : std::uint8_t
  {
    Vertex  = 0,
    Line    = 1,
    Tri     = 2,
    Quad    = 3,
    Tet     = 4,
    Pyramid = 5,
    Wedge   = 6,
    Hex     = 7,
    Invalid = static_cast<std::uint8_t>(-1)
  };

  /**
   * Return the correct simplex reference cell type for the given dimension
   * @p dim.
   */
  inline Type
  get_simplex(const unsigned int dim)
  {
    switch (dim)
      {
        case 0:
          return Type::Vertex;
        case 1:
          return Type::Line;
        case 2:
          return Type::Tri;
        case 3:
          return Type::Tet;
        default:
          Assert(false, ExcNotImplemented());
          return Type::Invalid;
      }
  }

  /**
   * Return the correct hypercube reference cell type for the given dimension
   * @p dim.
   */
  inline Type
  get_hypercube(const unsigned int dim)
  {
    switch (dim)
      {
        case 0:
          return Type::Vertex;
        case 1:
          return Type::Line;
        case 2:
          return Type::Quad;
        case 3:
          return Type::Hex;
        default:
          Assert(false, ExcNotImplemented());
          return Type::Invalid;
      }
  }

  /**
   * Retrieve the correct ReferenceCell::Type for a given structural dimension
   * and number of vertices.
   */
  inline Type
  n_vertices_to_type(const int dim, const unsigned int n_vertices)
  {
    AssertIndexRange(dim, 4);
    AssertIndexRange(n_vertices, 9);
    const auto X = Type::Invalid;

    static constexpr std::array<std::array<ReferenceCell::Type, 9>, 4> table = {
      {// dim 0
       {{X, Type::Vertex, X, X, X, X, X, X, X}},
       // dim 1
       {{X, X, Type::Line, X, X, X, X, X, X}},
       // dim 2
       {{X, X, X, Type::Tri, Type::Quad, X, X, X, X}},
       // dim 3
       {{X, X, X, X, Type::Tet, Type::Pyramid, Type::Wedge, X, Type::Hex}}}};
    Assert(table[dim][n_vertices] != Type::Invalid,
           ExcMessage("The combination of dim = " + std::to_string(dim) +
                      " and n_vertices = " + std::to_string(n_vertices) +
                      " does not correspond to a known reference cell type."));
    return table[dim][n_vertices];
  }

  namespace internal
  {
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



    /**
     * Set the bit at position @p n in @p number to value @p x.
     */
    inline static void
    set_bit(unsigned char &number, const unsigned int n, const bool x)
    {
      AssertIndexRange(n, 8);

      // source:
      // https://stackoverflow.com/questions/47981/how-do-you-set-clear-and-toggle-a-single-bit
      // "Changing the nth bit to x"
      number ^= (-static_cast<unsigned char>(x) ^ number) & (1U << n);
    }

    /**
     * A namespace for geometric information on reference cells.
     */
    namespace Info
    {
      /**
       * Interface to be used in TriaAccessor/TriaCellAccessor to access
       * sub-entities of dimension d' of geometric entities of dimension d, with
       * 0<=d'<d<=3.
       */
      struct Base
      {
        /**
         * Destructor.
         */
        virtual ~Base() = default;

        /**
         * Number of vertices.
         */
        virtual unsigned int
        n_vertices() const
        {
          Assert(false, ExcNotImplemented());
          return 0;
        }

        /**
         * Number of lines.
         */
        virtual unsigned int
        n_lines() const
        {
          Assert(false, ExcNotImplemented());
          return 0;
        }


        /**
         * Number of faces.
         */
        virtual unsigned int
        n_faces() const
        {
          Assert(false, ExcNotImplemented());
          return 0;
        }

        /**
         * Return an object that can be thought of as an array containing all
         * indices from zero to n_vertices().
         */
        inline std_cxx20::ranges::iota_view<unsigned int, unsigned int>
        vertex_indices() const
        {
          return {0U, n_vertices()};
        }

        /**
         * Return an object that can be thought of as an array containing all
         * indices from zero to n_lines().
         */
        inline std_cxx20::ranges::iota_view<unsigned int, unsigned int>
        line_indices() const
        {
          return {0U, n_lines()};
        }

        /**
         * Return an object that can be thought of as an array containing all
         * indices from zero to n_faces().
         */
        inline std_cxx20::ranges::iota_view<unsigned int, unsigned int>
        face_indices() const
        {
          return {0U, n_faces()};
        }

        /**
         * Standard decomposition of vertex index into face and face-vertex
         * index.
         */
        virtual std::array<unsigned int, 2>
        standard_vertex_to_face_and_vertex_index(
          const unsigned int vertex) const
        {
          Assert(false, ExcNotImplemented());

          (void)vertex;

          return {{0u, 0u}};
        }

        /**
         * Standard decomposition of line index into face and face-line index.
         */
        virtual std::array<unsigned int, 2>
        standard_line_to_face_and_line_index(const unsigned int line) const
        {
          Assert(false, ExcNotImplemented());

          (void)line;

          return {{0, 0}};
        }

        /**
         * Correct vertex index depending on face orientation.
         */
        virtual unsigned int
        standard_to_real_face_vertex(const unsigned int  vertex,
                                     const unsigned int  face,
                                     const unsigned char face_orientation) const
        {
          Assert(false, ExcNotImplemented());

          (void)vertex;
          (void)face;
          (void)face_orientation;

          return 0;
        }

        /**
         * Correct line index depending on face orientation.
         */
        virtual unsigned int
        standard_to_real_face_line(const unsigned int  line,
                                   const unsigned int  face,
                                   const unsigned char face_orientation) const
        {
          Assert(false, ExcNotImplemented());

          (void)line;
          (void)face;
          (void)face_orientation;

          return 0;
        }

        /**
         * Combine face and line orientation.
         */
        virtual bool
        combine_face_and_line_orientation(
          const unsigned int  line,
          const unsigned char face_orientation,
          const unsigned char line_orientation) const
        {
          Assert(false, ExcNotImplemented());

          (void)line;
          (void)face_orientation;
          (void)line_orientation;

          return true;
        }

        /**
         * Return reference-cell type of face @p face_no.
         */
        virtual ReferenceCell::Type
        face_reference_cell_type(const unsigned int face_no) const
        {
          Assert(false, ExcNotImplemented());
          (void)face_no;

          return ReferenceCell::Type::Invalid;
        }

        /**
         * Map face line number to cell line number.
         */
        virtual unsigned int
        face_to_cell_lines(const unsigned int  face,
                           const unsigned int  line,
                           const unsigned char face_orientation) const
        {
          Assert(false, ExcNotImplemented());
          (void)face;
          (void)line;
          (void)face_orientation;

          return 0;
        }

        /**
         * Map face vertex number to cell vertex number.
         */
        virtual unsigned int
        face_to_cell_vertices(const unsigned int  face,
                              const unsigned int  vertex,
                              const unsigned char face_orientation) const
        {
          Assert(false, ExcNotImplemented());
          (void)face;
          (void)vertex;
          (void)face_orientation;

          return 0;
        }

        /**
         * Map an ExodusII vertex number to a deal.II vertex number.
         */
        virtual unsigned int
        exodusii_vertex_to_deal_vertex(const unsigned int vertex_n) const
        {
          Assert(false, ExcNotImplemented());
          (void)vertex_n;

          return 0;
        }

        /**
         * Map an ExodusII face number to a deal.II face number.
         */
        virtual unsigned int
        exodusii_face_to_deal_face(const unsigned int face_n) const
        {
          Assert(false, ExcNotImplemented());
          (void)face_n;

          return 0;
        }
      };


      /**
       * Base class for tensor-product geometric entities.
       */
      template <int dim>
      struct TensorProductBase : Base
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

        unsigned int
        face_to_cell_lines(const unsigned int  face,
                           const unsigned int  line,
                           const unsigned char face_orientation) const override
        {
          return GeometryInfo<dim>::face_to_cell_lines(
            face,
            line,
            get_bit(face_orientation, 0),
            get_bit(face_orientation, 2),
            get_bit(face_orientation, 1));
        }

        unsigned int
        face_to_cell_vertices(
          const unsigned int  face,
          const unsigned int  vertex,
          const unsigned char face_orientation) const override
        {
          return GeometryInfo<dim>::face_to_cell_vertices(
            face,
            vertex,
            get_bit(face_orientation, 0),
            get_bit(face_orientation, 2),
            get_bit(face_orientation, 1));
        }
      };



      /*
       * Vertex.
       */
      struct Vertex : public TensorProductBase<0>
      {
        ReferenceCell::Type
        face_reference_cell_type(const unsigned int face_no) const override
        {
          (void)face_no;
          return ReferenceCell::Type::Invalid;
        }

        virtual unsigned int
        exodusii_face_to_deal_face(const unsigned int face_n) const override
        {
          (void)face_n;
          AssertIndexRange(face_n, n_faces());

          return 0;
        }
      };



      /*
       * Line.
       */
      struct Line : public TensorProductBase<1>
      {
        ReferenceCell::Type
        face_reference_cell_type(const unsigned int face_no) const override
        {
          (void)face_no;
          return ReferenceCell::Type::Vertex;
        }

        virtual unsigned int
        exodusii_vertex_to_deal_vertex(
          const unsigned int vertex_n) const override
        {
          AssertIndexRange(vertex_n, n_vertices());
          return vertex_n;
        }

        virtual unsigned int
        exodusii_face_to_deal_face(const unsigned int face_n) const override
        {
          AssertIndexRange(face_n, n_faces());
          return face_n;
        }
      };



      /**
       * Triangle.
       */
      struct Tri : public Base
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
        standard_vertex_to_face_and_vertex_index(
          const unsigned int vertex) const override
        {
          AssertIndexRange(vertex, 3);

          static const std::array<std::array<unsigned int, 2>, 3> table = {
            {{{0, 0}}, {{0, 1}}, {{1, 1}}}};

          return table[vertex];
        }

        unsigned int
        standard_to_real_face_vertex(
          const unsigned int  vertex,
          const unsigned int  face,
          const unsigned char line_orientation) const override
        {
          (void)face;

          static const std::array<std::array<unsigned int, 2>, 2> table = {
            {{{1, 0}}, {{0, 1}}}};

          return table[line_orientation][vertex];
        }

        ReferenceCell::Type
        face_reference_cell_type(const unsigned int face_no) const override
        {
          (void)face_no;

          AssertIndexRange(face_no, n_faces());

          return ReferenceCell::Type::Line;
        }

        unsigned int
        face_to_cell_lines(const unsigned int  face,
                           const unsigned int  line,
                           const unsigned char face_orientation) const override
        {
          AssertIndexRange(face, n_faces());
          AssertDimension(line, 0);

          (void)line;
          (void)face_orientation;

          return face;
        }

        unsigned int
        face_to_cell_vertices(
          const unsigned int  face,
          const unsigned int  vertex,
          const unsigned char face_orientation) const override
        {
          static const std::array<std::array<unsigned int, 2>, 3> table = {
            {{{0, 1}}, {{1, 2}}, {{2, 0}}}};

          return table[face][face_orientation ? vertex : (1 - vertex)];
        }

        virtual unsigned int
        exodusii_vertex_to_deal_vertex(
          const unsigned int vertex_n) const override
        {
          AssertIndexRange(vertex_n, n_vertices());
          return vertex_n;
        }

        virtual unsigned int
        exodusii_face_to_deal_face(const unsigned int face_n) const override
        {
          AssertIndexRange(face_n, n_faces());
          return face_n;
        }
      };



      /**
       * Quad.
       */
      struct Quad : public TensorProductBase<2>
      {
        std::array<unsigned int, 2>
        standard_vertex_to_face_and_vertex_index(
          const unsigned int vertex) const override
        {
          return GeometryInfo<2>::standard_quad_vertex_to_line_vertex_index(
            vertex);
        }

        unsigned int
        standard_to_real_face_vertex(
          const unsigned int  vertex,
          const unsigned int  face,
          const unsigned char line_orientation) const override
        {
          (void)face;

          return GeometryInfo<2>::standard_to_real_line_vertex(
            vertex, line_orientation);
        }

        ReferenceCell::Type
        face_reference_cell_type(const unsigned int face_no) const override
        {
          (void)face_no;
          return ReferenceCell::Type::Line;
        }

        virtual unsigned int
        exodusii_vertex_to_deal_vertex(
          const unsigned int vertex_n) const override
        {
          AssertIndexRange(vertex_n, n_vertices());
          constexpr std::array<unsigned int, 4> exodus_to_deal{{0, 1, 3, 2}};
          return exodus_to_deal[vertex_n];
        }

        virtual unsigned int
        exodusii_face_to_deal_face(const unsigned int face_n) const override
        {
          AssertIndexRange(face_n, n_faces());
          constexpr std::array<unsigned int, 4> exodus_to_deal{{2, 1, 3, 0}};
          return exodus_to_deal[face_n];
        }
      };



      /**
       * Tet.
       */
      struct Tet : public TensorProductBase<3>
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
        standard_line_to_face_and_line_index(
          const unsigned int line) const override
        {
          static const std::array<unsigned int, 2> table[6] = {
            {{0, 0}}, {{0, 1}}, {{0, 2}}, {{1, 1}}, {{1, 2}}, {{2, 1}}};

          return table[line];
        }

        unsigned int
        standard_to_real_face_line(
          const unsigned int  line,
          const unsigned int  face,
          const unsigned char face_orientation) const override
        {
          (void)face;

          static const std::array<std::array<unsigned int, 3>, 6> table = {
            {{{2, 1, 0}},
             {{0, 1, 2}},
             {{1, 2, 0}},
             {{0, 2, 1}},
             {{1, 0, 2}},
             {{2, 0, 1}}}};

          return table[face_orientation][line];
        }

        bool
        combine_face_and_line_orientation(
          const unsigned int  line,
          const unsigned char face_orientation_raw,
          const unsigned char line_orientation) const override
        {
          (void)line;
          (void)face_orientation_raw;

          return line_orientation;
        }

        std::array<unsigned int, 2>
        standard_vertex_to_face_and_vertex_index(
          const unsigned int vertex) const override
        {
          AssertIndexRange(vertex, 4);

          static const std::array<unsigned int, 2> table[4] = {{{0, 0}},
                                                               {{0, 1}},
                                                               {{0, 2}},
                                                               {{1, 2}}};

          return table[vertex];
        }

        unsigned int
        standard_to_real_face_vertex(
          const unsigned int  vertex,
          const unsigned int  face,
          const unsigned char face_orientation) const override
        {
          AssertIndexRange(face_orientation, 6);
          (void)face;

          static const std::array<std::array<unsigned int, 3>, 6> table = {
            {{{0, 2, 1}},
             {{0, 1, 2}},
             {{1, 2, 0}},
             {{1, 0, 2}},
             {{2, 1, 0}},
             {{2, 0, 1}}}};

          return table[face_orientation][vertex];
        }

        ReferenceCell::Type
        face_reference_cell_type(const unsigned int face_no) const override
        {
          (void)face_no;

          AssertIndexRange(face_no, n_faces());

          return ReferenceCell::Type::Tri;
        }

        unsigned int
        face_to_cell_lines(const unsigned int  face,
                           const unsigned int  line,
                           const unsigned char face_orientation) const override
        {
          AssertIndexRange(face, n_faces());

          const static std::array<std::array<unsigned int, 3>, 4> table = {
            {{{0, 1, 2}}, {{0, 3, 4}}, {{2, 5, 3}}, {{1, 4, 5}}}};

          return table[face][standard_to_real_face_line(
            line, face, face_orientation)];
        }

        unsigned int
        face_to_cell_vertices(
          const unsigned int  face,
          const unsigned int  vertex,
          const unsigned char face_orientation) const override
        {
          static const std::array<std::array<unsigned int, 3>, 4> table = {
            {{{0, 1, 2}}, {{1, 0, 3}}, {{0, 2, 3}}, {{2, 1, 3}}}};

          return table[face][standard_to_real_face_vertex(
            vertex, face, face_orientation)];
        }

        virtual unsigned int
        exodusii_vertex_to_deal_vertex(
          const unsigned int vertex_n) const override
        {
          AssertIndexRange(vertex_n, n_vertices());
          return vertex_n;
        }

        virtual unsigned int
        exodusii_face_to_deal_face(const unsigned int face_n) const override
        {
          AssertIndexRange(face_n, n_faces());
          constexpr std::array<unsigned int, 4> exodus_to_deal{{1, 3, 2, 0}};
          return exodus_to_deal[face_n];
        }
      };



      /**
       * Pyramid.
       */
      struct Pyramid : public TensorProductBase<3>
      {
        unsigned int
        n_vertices() const override
        {
          return 5;
        }

        unsigned int
        n_lines() const override
        {
          return 8;
        }

        unsigned int
        n_faces() const override
        {
          return 5;
        }

        std::array<unsigned int, 2>
        standard_line_to_face_and_line_index(
          const unsigned int line) const override
        {
          Assert(false, ExcNotImplemented());

          static const std::array<unsigned int, 2> table[6] = {
            {{0, 0}}, {{0, 1}}, {{0, 2}}, {{1, 1}}, {{1, 2}}, {{2, 1}}};

          return table[line];
        }

        unsigned int
        standard_to_real_face_line(
          const unsigned int  line,
          const unsigned int  face,
          const unsigned char face_orientation) const override
        {
          Assert(false, ExcNotImplemented());

          (void)face;

          static const std::array<std::array<unsigned int, 3>, 6> table = {
            {{{2, 1, 0}},
             {{0, 1, 2}},
             {{1, 2, 0}},
             {{0, 2, 1}},
             {{1, 0, 2}},
             {{2, 0, 1}}}};

          return table[face_orientation][line];
        }

        bool
        combine_face_and_line_orientation(
          const unsigned int  line,
          const unsigned char face_orientation_raw,
          const unsigned char line_orientation) const override
        {
          (void)line;
          (void)face_orientation_raw;

          return line_orientation;
        }

        std::array<unsigned int, 2>
        standard_vertex_to_face_and_vertex_index(
          const unsigned int vertex) const override
        {
          static const std::array<unsigned int, 2> table[5] = {
            {{0, 0}}, {{0, 1}}, {{0, 2}}, {{0, 3}}, {{1, 2}}};

          return table[vertex];
        }

        unsigned int
        standard_to_real_face_vertex(
          const unsigned int  vertex,
          const unsigned int  face,
          const unsigned char face_orientation) const override
        {
          if (face == 0) // Quad
            {
              return GeometryInfo<3>::standard_to_real_face_vertex(
                vertex,
                get_bit(face_orientation, 0),
                get_bit(face_orientation, 2),
                get_bit(face_orientation, 1));
            }
          else // Tri
            {
              static const std::array<std::array<unsigned int, 3>, 6> table = {
                {{{0, 2, 1}},
                 {{0, 1, 2}},
                 {{1, 2, 0}},
                 {{1, 0, 2}},
                 {{2, 1, 0}},
                 {{2, 0, 1}}}};

              return table[face_orientation][vertex];
            }
        }

        ReferenceCell::Type
        face_reference_cell_type(const unsigned int face_no) const override
        {
          AssertIndexRange(face_no, n_faces());

          if (face_no == 0)
            return ReferenceCell::Type::Quad;
          else
            return ReferenceCell::Type::Tri;
        }

        unsigned int
        face_to_cell_vertices(
          const unsigned int  face,
          const unsigned int  vertex,
          const unsigned char face_orientation) const override
        {
          AssertIndexRange(face, n_faces());
          if (face == 0)
            {
              AssertIndexRange(vertex, 4);
            }
          else
            {
              AssertIndexRange(vertex, 3);
            }
          constexpr auto X = numbers::invalid_unsigned_int;
          static const std::array<std::array<unsigned int, 4>, 5> table = {
            {{{0, 1, 2, 3}},
             {{0, 2, 4, X}},
             {{3, 1, 4, X}},
             {{1, 0, 4, X}},
             {{2, 3, 4, X}}}};

          return table[face][standard_to_real_face_vertex(
            vertex, face, face_orientation)];
        }

        virtual unsigned int
        exodusii_vertex_to_deal_vertex(
          const unsigned int vertex_n) const override
        {
          AssertIndexRange(vertex_n, n_vertices());
          constexpr std::array<unsigned int, 5> exodus_to_deal{{0, 1, 3, 2, 4}};
          return exodus_to_deal[vertex_n];
        }

        virtual unsigned int
        exodusii_face_to_deal_face(const unsigned int face_n) const override
        {
          AssertIndexRange(face_n, n_faces());
          constexpr std::array<unsigned int, 5> exodus_to_deal{{3, 2, 4, 1, 0}};
          return exodus_to_deal[face_n];
        }
      };



      /**
       * Wedge.
       */
      struct Wedge : public TensorProductBase<3>
      {
        unsigned int
        n_vertices() const override
        {
          return 6;
        }

        unsigned int
        n_lines() const override
        {
          return 9;
        }

        unsigned int
        n_faces() const override
        {
          return 5;
        }

        std::array<unsigned int, 2>
        standard_line_to_face_and_line_index(
          const unsigned int line) const override
        {
          Assert(false, ExcNotImplemented());

          static const std::array<unsigned int, 2> table[6] = {
            {{0, 0}}, {{0, 1}}, {{0, 2}}, {{1, 1}}, {{1, 2}}, {{2, 1}}};

          return table[line];
        }

        unsigned int
        standard_to_real_face_line(
          const unsigned int  line,
          const unsigned int  face,
          const unsigned char face_orientation) const override
        {
          Assert(false, ExcNotImplemented());

          (void)face;

          static const std::array<std::array<unsigned int, 3>, 6> table = {
            {{{2, 1, 0}},
             {{0, 1, 2}},
             {{1, 2, 0}},
             {{0, 2, 1}},
             {{1, 0, 2}},
             {{2, 0, 1}}}};

          return table[face_orientation][line];
        }

        bool
        combine_face_and_line_orientation(
          const unsigned int  line,
          const unsigned char face_orientation_raw,
          const unsigned char line_orientation) const override
        {
          (void)line;
          (void)face_orientation_raw;

          return line_orientation;
        }

        std::array<unsigned int, 2>
        standard_vertex_to_face_and_vertex_index(
          const unsigned int vertex) const override
        {
          static const std::array<std::array<unsigned int, 2>, 6> table = {
            {{{0, 1}}, {{0, 0}}, {{0, 2}}, {{1, 0}}, {{1, 1}}, {{1, 2}}}};

          return table[vertex];
        }

        unsigned int
        standard_to_real_face_vertex(
          const unsigned int  vertex,
          const unsigned int  face,
          const unsigned char face_orientation) const override
        {
          if (face > 1) // QUAD
            {
              return GeometryInfo<3>::standard_to_real_face_vertex(
                vertex,
                get_bit(face_orientation, 0),
                get_bit(face_orientation, 2),
                get_bit(face_orientation, 1));
            }
          else // TRI
            {
              static const std::array<std::array<unsigned int, 3>, 6> table = {
                {{{0, 2, 1}},
                 {{0, 1, 2}},
                 {{1, 2, 0}},
                 {{1, 0, 2}},
                 {{2, 1, 0}},
                 {{2, 0, 1}}}};

              return table[face_orientation][vertex];
            }
        }

        ReferenceCell::Type
        face_reference_cell_type(const unsigned int face_no) const override
        {
          AssertIndexRange(face_no, n_faces());

          if (face_no > 1)
            return ReferenceCell::Type::Quad;
          else
            return ReferenceCell::Type::Tri;
        }

        unsigned int
        face_to_cell_vertices(
          const unsigned int  face,
          const unsigned int  vertex,
          const unsigned char face_orientation) const override
        {
          AssertIndexRange(face, n_faces());
          if (face < 2)
            {
              AssertIndexRange(vertex, 3);
            }
          else
            {
              AssertIndexRange(vertex, 4);
            }
          constexpr auto X = numbers::invalid_unsigned_int;
          static const std::array<std::array<unsigned int, 4>, 6> table = {
            {{{1, 0, 2, X}},
             {{3, 4, 5, X}},
             {{0, 1, 3, 4}},
             {{1, 2, 4, 5}},
             {{2, 0, 5, 3}}}};

          return table[face][standard_to_real_face_vertex(
            vertex, face, face_orientation)];
        }

        virtual unsigned int
        exodusii_vertex_to_deal_vertex(
          const unsigned int vertex_n) const override
        {
          AssertIndexRange(vertex_n, n_vertices());
          constexpr std::array<unsigned int, 6> exodus_to_deal{
            {2, 1, 0, 5, 4, 3}};
          return exodus_to_deal[vertex_n];
        }

        virtual unsigned int
        exodusii_face_to_deal_face(const unsigned int face_n) const override
        {
          AssertIndexRange(face_n, n_faces());
          constexpr std::array<unsigned int, 6> exodus_to_deal{{3, 4, 2, 0, 1}};
          return exodus_to_deal[face_n];
        }
      };



      /**
       * Hex.
       */
      struct Hex : public TensorProductBase<3>
      {
        std::array<unsigned int, 2>
        standard_line_to_face_and_line_index(
          const unsigned int line) const override
        {
          return GeometryInfo<3>::standard_hex_line_to_quad_line_index(line);
        }

        unsigned int
        standard_to_real_face_line(
          const unsigned int  line,
          const unsigned int  face,
          const unsigned char face_orientation) const override
        {
          (void)face;

          return GeometryInfo<3>::standard_to_real_face_line(
            line,
            get_bit(face_orientation, 0),
            get_bit(face_orientation, 2),
            get_bit(face_orientation, 1));
        }

        bool
        combine_face_and_line_orientation(
          const unsigned int  line,
          const unsigned char face_orientation_raw,
          const unsigned char line_orientation) const override
        {
          static const bool bool_table[2][2][2][2] = {
            {{{true, false},    // lines 0/1, face_orientation=false,
                                // face_flip=false, face_rotation=false and true
              {false, true}},   // lines 0/1, face_orientation=false,
                                // face_flip=true, face_rotation=false and true
             {{true, true},     // lines 0/1, face_orientation=true,
                                // face_flip=false, face_rotation=false and true
              {false, false}}}, // lines 0/1, face_orientation=true,
                                // face_flip=true, face_rotation=false and true

            {{{true, true}, // lines 2/3 ...
              {false, false}},
             {{true, false}, {false, true}}}};

          const bool face_orientation = get_bit(face_orientation_raw, 0);
          const bool face_flip        = get_bit(face_orientation_raw, 2);
          const bool face_rotation    = get_bit(face_orientation_raw, 1);

          return (
            static_cast<bool>(line_orientation) ==
            bool_table[line / 2][face_orientation][face_flip][face_rotation]);
        }

        std::array<unsigned int, 2>
        standard_vertex_to_face_and_vertex_index(
          const unsigned int vertex) const override
        {
          return GeometryInfo<3>::standard_hex_vertex_to_quad_vertex_index(
            vertex);
        }

        unsigned int
        standard_to_real_face_vertex(
          const unsigned int  vertex,
          const unsigned int  face,
          const unsigned char face_orientation) const override
        {
          (void)face;

          return GeometryInfo<3>::standard_to_real_face_vertex(
            vertex,
            get_bit(face_orientation, 0),
            get_bit(face_orientation, 2),
            get_bit(face_orientation, 1));
        }

        ReferenceCell::Type
        face_reference_cell_type(const unsigned int face_no) const override
        {
          (void)face_no;
          return ReferenceCell::Type::Quad;
        }

        virtual unsigned int
        exodusii_vertex_to_deal_vertex(
          const unsigned int vertex_n) const override
        {
          AssertIndexRange(vertex_n, n_vertices());
          constexpr std::array<unsigned int, 8> exodus_to_deal{
            {0, 1, 3, 2, 4, 5, 7, 6}};
          return exodus_to_deal[vertex_n];
        }

        virtual unsigned int
        exodusii_face_to_deal_face(const unsigned int face_n) const override
        {
          AssertIndexRange(face_n, n_faces());
          constexpr std::array<unsigned int, 6> exodus_to_deal{
            {2, 1, 3, 0, 4, 5}};
          return exodus_to_deal[face_n];
        }
      };

      /**
       * Return for a given reference-cell type the right Info.
       */
      inline const ReferenceCell::internal::Info::Base &
      get_cell(const ReferenceCell::Type &type)
      {
        static const std::
          array<std::unique_ptr<ReferenceCell::internal::Info::Base>, 8>
            gei{{std::make_unique<ReferenceCell::internal::Info::Vertex>(),
                 std::make_unique<ReferenceCell::internal::Info::Line>(),
                 std::make_unique<ReferenceCell::internal::Info::Tri>(),
                 std::make_unique<ReferenceCell::internal::Info::Quad>(),
                 std::make_unique<ReferenceCell::internal::Info::Tet>(),
                 std::make_unique<ReferenceCell::internal::Info::Pyramid>(),
                 std::make_unique<ReferenceCell::internal::Info::Wedge>(),
                 std::make_unique<ReferenceCell::internal::Info::Hex>()}};
        AssertIndexRange(static_cast<std::uint8_t>(type), 8);
        return *gei[static_cast<std::uint8_t>(type)];
      }

      /**
       * Return for a given reference-cell type @p and face number @p face_no the
       * right Info of the @p face_no-th face.
       */
      inline const ReferenceCell::internal::Info::Base &
      get_face(const ReferenceCell::Type &type, const unsigned int face_no)
      {
        return get_cell(get_cell(type).face_reference_cell_type(face_no));
      }

    } // namespace Info
  }   // namespace internal
} // namespace ReferenceCell


DEAL_II_NAMESPACE_CLOSE

#endif
