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


#ifndef dealii_matrix_free_util_h
#define dealii_matrix_free_util_h


#include <deal.II/base/config.h>

#include <deal.II/base/quadrature.h>

#include <deal.II/simplex/quadrature_lib.h>

DEAL_II_NAMESPACE_OPEN


namespace internal
{
  namespace MatrixFreeFunctions
  {
    template <int dim>
    inline Quadrature<dim - 1>
    get_face_quadrature(const Quadrature<dim> &quad)
    {
      if (dim == 2 || dim == 3)
        for (unsigned int i = 1; i <= 3; ++i)
          if (quad == Simplex::QGauss<dim>(i))
            return Simplex::QGauss<dim - 1>(i);

      AssertThrow(false, ExcNotImplemented());

      return Quadrature<dim - 1>();
    }
    
    template <int dim>
    inline std::pair<ReferenceCell::Type, dealii::hp::QCollection<dim - 1>>
    get_face_quadrature_collection(const Quadrature<dim> &quad,
                                   const bool             do_assert = true)
    {
      if (dim == 2 || dim == 3)
        for (unsigned int i = 1; i <= 3; ++i)
          if (quad == Simplex::QGauss<dim>(i))
            {
              Simplex::QGauss<dim - 1> tri(i);

              if (dim == 2)
                return {ReferenceCell::Type::Tri,
                        dealii::hp::QCollection<dim - 1>(tri, tri, tri)};
              else
                return {ReferenceCell::Type::Tet,
                        dealii::hp::QCollection<dim - 1>(tri, tri, tri, tri)};
            }

      if (dim == 3)
        for (unsigned int i = 1; i <= 3; ++i)
          if (quad == Simplex::QGaussWedge<dim>(i))
            {
              QGauss<dim - 1>          quad(i);
              Simplex::QGauss<dim - 1> tri(i);

              return {
                ReferenceCell::Type::Wedge,
                dealii::hp::QCollection<dim - 1>(tri, tri, quad, quad, quad)};
            }

      if (dim == 3)
        for (unsigned int i = 1; i <= 2; ++i)
          if (quad == Simplex::QGaussPyramid<dim>(i))
            {
              QGauss<dim - 1>          quad(i);
              Simplex::QGauss<dim - 1> tri(i);

              return {
                ReferenceCell::Type::Pyramid,
                dealii::hp::QCollection<dim - 1>(quad, tri, tri, tri, tri)};
            }

      if (do_assert)
        AssertThrow(false, ExcNotImplemented());

      return {ReferenceCell::Type::Invalid, dealii::hp::QCollection<dim - 1>()};
    }

  } // end of namespace MatrixFreeFunctions
} // end of namespace internal

DEAL_II_NAMESPACE_CLOSE

#endif
