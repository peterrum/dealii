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

#ifndef dealii_simplex_fe_lib_h
#define dealii_simplex_fe_lib_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_poly.h>

#include <deal.II/simplex/polynomials.h>

DEAL_II_NAMESPACE_OPEN

namespace Simplex
{
  /**
   * Base class of FE_P and FE_DGP.
   *
   * @note Only implemented for 2D and 3D.
   *
   * @ingroup simplex
   */
  template <int dim, int spacedim = dim>
  class FE_Poly : public dealii::FE_Poly<dim, spacedim>
  {
  public:
    /**
     * Constructor.
     */
    FE_Poly(const unsigned int               degree,
            const std::vector<unsigned int> &dpo_vector);

  private:
    /**
     * @copydoc dealii::FiniteElement::convert_generalized_support_point_values_to_dof_values()
     */
    void
    convert_generalized_support_point_values_to_dof_values(
      const std::vector<Vector<double>> &support_point_values,
      std::vector<double> &              nodal_values) const override;
  };



  /**
   * Implementation of a scalar Lagrange finite element Pp that yields
   * the finite element space of continuous, piecewise polynomials of
   * degree p.
   *
   * @ingroup simplex
   */
  template <int dim, int spacedim = dim>
  class FE_P : public FE_Poly<dim, spacedim>
  {
  public:
    /**
     * Constructor.
     */
    FE_P(const unsigned int degree);

    /**
     * @copydoc dealii::FiniteElement::clone()
     */
    std::unique_ptr<FiniteElement<dim, spacedim>>
    clone() const override;

    /**
     * Return a string that uniquely identifies a finite element. This class
     * returns <tt>Simplex::FE_P<dim>(degree)</tt>, with @p dim and @p degree
     * replaced by appropriate values.
     */
    std::string
    get_name() const override;
  };



  /**
   * Implementation of a scalar Lagrange finite element Pp that yields
   * the finite element space of discontinuous, piecewise polynomials of
   * degree p.
   *
   * @ingroup simplex
   */
  template <int dim, int spacedim = dim>
  class FE_DGP : public FE_Poly<dim, spacedim>
  {
  public:
    /**
     * Constructor.
     */
    FE_DGP(const unsigned int degree);

    /**
     * @copydoc dealii::FiniteElement::clone()
     */
    std::unique_ptr<FiniteElement<dim, spacedim>>
    clone() const override;

    /**
     * Return a string that uniquely identifies a finite element. This class
     * returns <tt>Simplex::FE_DGP<dim>(degree)</tt>, with @p dim and @p degree
     * replaced by appropriate values.
     */
    std::string
    get_name() const override;
  };

} // namespace Simplex

DEAL_II_NAMESPACE_CLOSE

#endif
