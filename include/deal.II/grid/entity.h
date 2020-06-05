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
    return 3;
  }
};



/**
 * QUAD
 */
struct DynamicGeometryInfoQuad : public DynamicGeometryInfoTensor<2>
{};



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
{};



DEAL_II_NAMESPACE_CLOSE

#endif
