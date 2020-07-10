// ---------------------------------------------------------------------
//
// Copyright (C) 2001 - 2018 by the deal.II authors
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

#include <deal.II/base/geometry_info.h>

#include <deal.II/fe/fe.h>

DEAL_II_NAMESPACE_OPEN

namespace
{
  inline unsigned int
  vertices_per_cell(const DynamicGeometryInfo &geometry_info)
  {
    if (dynamic_cast<const DynamicGeometryInfoVertex *>(&geometry_info))
      return GeometryInfo<0>::vertices_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoLine *>(&geometry_info))
      return GeometryInfo<1>::vertices_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoTri *>(&geometry_info))
      return 3;
    if (dynamic_cast<const DynamicGeometryInfoQuad *>(&geometry_info))
      return GeometryInfo<2>::vertices_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoTet *>(&geometry_info))
      return 4;
    if (dynamic_cast<const DynamicGeometryInfoHex *>(&geometry_info))
      return GeometryInfo<3>::vertices_per_cell;

    return 0;
  }

  inline unsigned int
  lines_per_cell(const DynamicGeometryInfo &geometry_info)
  {
    if (dynamic_cast<const DynamicGeometryInfoVertex *>(&geometry_info))
      return GeometryInfo<0>::lines_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoLine *>(&geometry_info))
      return GeometryInfo<1>::lines_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoTri *>(&geometry_info))
      return 3;
    if (dynamic_cast<const DynamicGeometryInfoQuad *>(&geometry_info))
      return GeometryInfo<2>::lines_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoTet *>(&geometry_info))
      return 6;
    if (dynamic_cast<const DynamicGeometryInfoHex *>(&geometry_info))
      return GeometryInfo<3>::lines_per_cell;

    return 0;
  }

  inline unsigned int
  quads_per_cell(const DynamicGeometryInfo &geometry_info)
  {
    if (dynamic_cast<const DynamicGeometryInfoVertex *>(&geometry_info))
      return GeometryInfo<0>::quads_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoLine *>(&geometry_info))
      return GeometryInfo<1>::quads_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoTri *>(&geometry_info))
      return 1;
    if (dynamic_cast<const DynamicGeometryInfoQuad *>(&geometry_info))
      return GeometryInfo<2>::quads_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoTet *>(&geometry_info))
      return 4;
    if (dynamic_cast<const DynamicGeometryInfoHex *>(&geometry_info))
      return GeometryInfo<3>::quads_per_cell;

    return 0;
  }

  inline unsigned int
  hexes_per_cell(const DynamicGeometryInfo &geometry_info)
  {
    if (dynamic_cast<const DynamicGeometryInfoVertex *>(&geometry_info))
      return GeometryInfo<0>::hexes_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoLine *>(&geometry_info))
      return GeometryInfo<1>::hexes_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoTri *>(&geometry_info))
      return 0;
    if (dynamic_cast<const DynamicGeometryInfoQuad *>(&geometry_info))
      return GeometryInfo<2>::hexes_per_cell;
    if (dynamic_cast<const DynamicGeometryInfoTet *>(&geometry_info))
      return 1;
    if (dynamic_cast<const DynamicGeometryInfoHex *>(&geometry_info))
      return GeometryInfo<3>::hexes_per_cell;

    return 0;
  }

  inline unsigned int
  vertices_per_face(const DynamicGeometryInfo &geometry_info)
  {
    if (dynamic_cast<const DynamicGeometryInfoVertex *>(&geometry_info))
      return GeometryInfo<0>::vertices_per_face;
    if (dynamic_cast<const DynamicGeometryInfoLine *>(&geometry_info))
      return GeometryInfo<1>::vertices_per_face;
    if (dynamic_cast<const DynamicGeometryInfoTri *>(&geometry_info))
      return 2;
    if (dynamic_cast<const DynamicGeometryInfoQuad *>(&geometry_info))
      return GeometryInfo<2>::vertices_per_face;
    if (dynamic_cast<const DynamicGeometryInfoTet *>(&geometry_info))
      return 3;
    if (dynamic_cast<const DynamicGeometryInfoHex *>(&geometry_info))
      return GeometryInfo<3>::vertices_per_face;

    return 0;
  }

  inline unsigned int
  lines_per_face(const DynamicGeometryInfo &geometry_info)
  {
    if (dynamic_cast<const DynamicGeometryInfoVertex *>(&geometry_info))
      return GeometryInfo<0>::lines_per_face;
    if (dynamic_cast<const DynamicGeometryInfoLine *>(&geometry_info))
      return GeometryInfo<1>::lines_per_face;
    if (dynamic_cast<const DynamicGeometryInfoTri *>(&geometry_info))
      return 1;
    if (dynamic_cast<const DynamicGeometryInfoQuad *>(&geometry_info))
      return GeometryInfo<2>::lines_per_face;
    if (dynamic_cast<const DynamicGeometryInfoTet *>(&geometry_info))
      return 3;
    if (dynamic_cast<const DynamicGeometryInfoHex *>(&geometry_info))
      return GeometryInfo<3>::lines_per_face;

    return 0;
  }

  inline unsigned int
  quads_per_face(const DynamicGeometryInfo &geometry_info)
  {
    if (dynamic_cast<const DynamicGeometryInfoVertex *>(&geometry_info))
      return GeometryInfo<0>::quads_per_face;
    if (dynamic_cast<const DynamicGeometryInfoLine *>(&geometry_info))
      return GeometryInfo<1>::quads_per_face;
    if (dynamic_cast<const DynamicGeometryInfoTri *>(&geometry_info))
      return 0;
    if (dynamic_cast<const DynamicGeometryInfoQuad *>(&geometry_info))
      return GeometryInfo<2>::quads_per_face;
    if (dynamic_cast<const DynamicGeometryInfoTet *>(&geometry_info))
      return 1;
    if (dynamic_cast<const DynamicGeometryInfoHex *>(&geometry_info))
      return GeometryInfo<3>::quads_per_face;

    return 0;
  }
} // namespace

template <int dim>
FiniteElementData<dim>::FiniteElementData(
  const std::vector<unsigned int> &dofs_per_object,
  const unsigned int               n_components,
  const unsigned int               degree,
  const Conformity                 conformity,
  const BlockIndices &             block_indices)
  : FiniteElementData(
      dofs_per_object,
      dim == 0 ?
        create_dynamic_geometry_info(EntityType::vertex) :
        (dim == 1 ? create_dynamic_geometry_info(EntityType::line) :
                    (dim == 2 ? create_dynamic_geometry_info(EntityType::quad) :
                                create_dynamic_geometry_info(EntityType::hex))),
      n_components,
      degree,
      conformity,
      block_indices)
{}

template <int dim>
FiniteElementData<dim>::FiniteElementData(
  const std::vector<unsigned int> &dofs_per_object,
  const DynamicGeometryInfo &      geometry_info,
  const unsigned int               n_components,
  const unsigned int               degree,
  const Conformity                 conformity,
  const BlockIndices &             block_indices)
  : dofs_per_vertex(dofs_per_object[0])
  , dofs_per_line(dofs_per_object[1])
  , dofs_per_quad(dim > 1 ? dofs_per_object[2] : 0)
  , dofs_per_hex(dim > 2 ? dofs_per_object[3] : 0)
  , first_line_index(vertices_per_cell(geometry_info) * dofs_per_vertex)
  , first_quad_index(first_line_index +
                     lines_per_cell(geometry_info) * dofs_per_line)
  , first_hex_index(first_quad_index +
                    quads_per_cell(geometry_info) * dofs_per_quad)
  , first_face_line_index(vertices_per_face(geometry_info) * dofs_per_vertex)
  , first_face_quad_index(
      (dim == 3 ? vertices_per_face(geometry_info) * dofs_per_vertex :
                  vertices_per_cell(geometry_info) * dofs_per_vertex) +
      lines_per_face(geometry_info) * dofs_per_line)
  , dofs_per_face(vertices_per_face(geometry_info) * dofs_per_vertex +
                  lines_per_face(geometry_info) * dofs_per_line +
                  quads_per_face(geometry_info) * dofs_per_quad)
  , dofs_per_cell(vertices_per_cell(geometry_info) * dofs_per_vertex +
                  lines_per_cell(geometry_info) * dofs_per_line +
                  quads_per_cell(geometry_info) * dofs_per_quad +
                  hexes_per_cell(geometry_info) * dofs_per_hex)
  , components(n_components)
  , degree(degree)
  , conforming_space(conformity)
  , block_indices_data(block_indices.size() == 0 ?
                         BlockIndices(1, dofs_per_cell) :
                         block_indices)
{
  Assert(dofs_per_object.size() == dim + 1,
         ExcDimensionMismatch(dofs_per_object.size() - 1, dim));
}



template <int dim>
bool
FiniteElementData<dim>::operator==(const FiniteElementData<dim> &f) const
{
  return ((dofs_per_vertex == f.dofs_per_vertex) &&
          (dofs_per_line == f.dofs_per_line) &&
          (dofs_per_quad == f.dofs_per_quad) &&
          (dofs_per_hex == f.dofs_per_hex) && (components == f.components) &&
          (degree == f.degree) && (conforming_space == f.conforming_space));
}


template class FiniteElementData<1>;
template class FiniteElementData<2>;
template class FiniteElementData<3>;

DEAL_II_NAMESPACE_CLOSE
