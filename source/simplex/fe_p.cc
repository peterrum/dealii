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

#ifndef dealii_fe_dgp_monomial_h
#define dealii_fe_dgp_monomial_h

#include <deal.II/base/config.h>

#include <deal.II/simplex/fe_p.h>

DEAL_II_NAMESPACE_OPEN

namespace Simplex
{
  template <int dim, int spacedim>
  FE_P<dim, spacedim>::FE_P(const unsigned int degree)
    : FE_Poly<dim, spacedim>(
        Simplex::ScalarPolynomial<dim>(degree),
        FiniteElementData<dim>(get_dpo_vector(degree),
                               dim == 2 ? ReferenceCell::Type::Tri :
                                          ReferenceCell::Type::Tet,
                               1,
                               degree,
                               FiniteElementData<dim>::L2),
        std::vector<bool>(FiniteElementData<dim>(get_dpo_vector(degree),
                                                 dim == 2 ?
                                                   ReferenceCell::Type::Tri :
                                                   ReferenceCell::Type::Tet,
                                                 1,
                                                 degree)
                            .dofs_per_cell,
                          true),
        std::vector<ComponentMask>(
          FiniteElementData<dim>(get_dpo_vector(degree),
                                 dim == 2 ? ReferenceCell::Type::Tri :
                                            ReferenceCell::Type::Tet,
                                 1,
                                 degree)
            .dofs_per_cell,
          std::vector<bool>(1, true)))
  {
    this->unit_support_points.clear();

    if (dim == 2)
      {
        if (degree == 1)
          {
            this->unit_support_points.emplace_back(1.0, 0.0);
            this->unit_support_points.emplace_back(0.0, 1.0);
            this->unit_support_points.emplace_back(0.0, 0.0);

            // TODO
            this->unit_face_support_points.emplace_back(0.0);
            this->unit_face_support_points.emplace_back(1.0);
          }
        else if (degree == 2)
          {
            this->unit_support_points.emplace_back(1.0, 0.0);
            this->unit_support_points.emplace_back(0.0, 1.0);
            this->unit_support_points.emplace_back(0.0, 0.0);
            this->unit_support_points.emplace_back(0.5, 0.5);
            this->unit_support_points.emplace_back(0.0, 0.5);
            this->unit_support_points.emplace_back(0.5, 0.0);

            // TODO
            this->unit_face_support_points.emplace_back(0.0);
            this->unit_face_support_points.emplace_back(1.0);
            this->unit_face_support_points.emplace_back(0.5);
          }
        else
          {
            Assert(false, ExcNotImplemented());
          }
      }
    else if (dim == 3)
      {
        if (degree == 1)
          {
            this->unit_support_points.emplace_back(0.0, 0.0, 0.0);
            this->unit_support_points.emplace_back(1.0, 0.0, 0.0);
            this->unit_support_points.emplace_back(0.0, 1.0, 0.0);
            this->unit_support_points.emplace_back(0.0, 0.0, 1.0);

            // TODO
            this->unit_face_support_points.emplace_back(1.0, 0.0);
            this->unit_face_support_points.emplace_back(0.0, 1.0);
            this->unit_face_support_points.emplace_back(0.0, 0.0);
          }
        else if (degree == 2)
          {
            this->unit_support_points.emplace_back(0.0, 0.0, 0.0);
            this->unit_support_points.emplace_back(1.0, 0.0, 0.0);
            this->unit_support_points.emplace_back(0.0, 1.0, 0.0);
            this->unit_support_points.emplace_back(0.0, 0.0, 1.0);
            this->unit_support_points.emplace_back(0.5, 0.0, 0.0);
            this->unit_support_points.emplace_back(0.5, 0.5, 0.0);
            this->unit_support_points.emplace_back(0.0, 0.5, 0.0);
            this->unit_support_points.emplace_back(0.0, 0.0, 0.5);
            this->unit_support_points.emplace_back(0.5, 0.0, 0.5);
            this->unit_support_points.emplace_back(0.0, 0.5, 0.5);

            // TODO
            this->unit_face_support_points.emplace_back(1.0, 0.0);
            this->unit_face_support_points.emplace_back(0.0, 1.0);
            this->unit_face_support_points.emplace_back(0.0, 0.0);
            this->unit_face_support_points.emplace_back(0.5, 0.5);
            this->unit_face_support_points.emplace_back(0.0, 0.5);
            this->unit_face_support_points.emplace_back(0.5, 0.0);
          }
        else
          {
            Assert(false, ExcNotImplemented());
          }
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
  }

  template <int dim, int spacedim>
  std::unique_ptr<FiniteElement<dim, dim>>
  FE_P<dim, spacedim>::clone() const
  {
    return std::make_unique<FE_P<dim, spacedim>>(*this);
  }

  template <int dim, int spacedim>
  std::string
  FE_P<dim, spacedim>::get_name() const
  {
    std::ostringstream namebuf;
    namebuf << "FE_P<" << dim << ">(" << this->degree << ")";

    return namebuf.str();
  }

} // namespace Simplex

// explicit instantiations
#include "fe_p.inst"

DEAL_II_NAMESPACE_CLOSE

#endif