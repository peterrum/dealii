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

#include <deal.II/base/pprojector.h>

#include <deal.II/fe/fe_poly.h>

#include <deal.II/simplex/polynomials.h>

DEAL_II_NAMESPACE_OPEN

namespace Simplex
{
  /**
   * Base class of FE_P and FE_DGP.
   *
   * @note Only implemented for 2D and 3D.
   */
  template <int dim, int spacedim = dim>
  class FE_Poly : public dealii::FE_Poly<dim>
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

  protected:
    /**
     * Create InternalData.
     */
    virtual std::unique_ptr<
      typename FiniteElement<dim, spacedim>::InternalDataBase>
    get_data(const UpdateFlags             update_flags,
             const Mapping<dim, spacedim> &mapping,
             const Quadrature<dim> &       quadrature,
             dealii::internal::FEValuesImplementation::FiniteElementRelatedData<
               dim,
               spacedim> &output_data) const override;

    /**
     * Create InternalData for face.
     */
    virtual std::unique_ptr<
      typename FiniteElement<dim, spacedim>::InternalDataBase>
    get_face_data(
      const UpdateFlags             flags,
      const Mapping<dim, spacedim> &mapping,
      const Quadrature<dim - 1> &   quadrature,
      dealii::internal::FEValuesImplementation::
        FiniteElementRelatedData<dim, spacedim> &output_data) const override;

    /**
     * @copydoc dealii::FiniteElement::fill_fe_values()
     */
    virtual void
    fill_fe_values(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const CellSimilarity::Similarity                         cell_similarity,
      const Quadrature<dim> &                                  quadrature,
      const Mapping<dim, spacedim> &                           mapping,
      const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
      const dealii::internal::FEValuesImplementation::
        MappingRelatedData<dim, spacedim> &mapping_data,
      const typename FiniteElement<dim, spacedim>::InternalDataBase
        &fe_internal,
      dealii::internal::FEValuesImplementation::
        FiniteElementRelatedData<dim, spacedim> &output_data) const override;

    /**
     * @copydoc dealii::FiniteElement::fill_fe_face_values()
     */
    void
    fill_fe_face_values(
      const typename Triangulation<dim, spacedim>::cell_iterator &cell,
      const unsigned int                                          face_no,
      const Quadrature<dim - 1> &                                 quadrature,
      const Mapping<dim, spacedim> &                              mapping,
      const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
      const dealii::internal::FEValuesImplementation::
        MappingRelatedData<dim, spacedim> &mapping_data,
      const typename FiniteElement<dim, spacedim>::InternalDataBase
        &fe_internal,
      dealii::internal::FEValuesImplementation::
        FiniteElementRelatedData<dim, spacedim> &output_data) const override;

    /**
     * Internal data. (TODO: needed?)
     */
    class InternalData : public FiniteElement<dim, dim>::InternalDataBase
    {
    public:
      /**
       * Array with shape function values in quadrature points. There is one row
       * for each shape function, containing values for each quadrature point.
       *
       * In this array, we store the values of the shape function in the
       * quadrature points on the unit cell. Since these values do not change
       * under transformation to the real cell, we only need to copy them over
       * when visiting a concrete cell.
       */
      Table<2, double> shape_values;

      /**
       * Array with shape function gradients in quadrature points. There is one
       * row for each shape function, containing values for each quadrature
       * point.
       *
       * We store the gradients in the quadrature points on the unit cell. We
       * then only have to apply the transformation (which is a matrix-vector
       * multiplication) when visiting an actual cell.
       */
      Table<2, Tensor<1, dim>> shape_gradients;

      /**
       * Array with shape function hessians in quadrature points. There is one
       * row for each shape function, containing values for each quadrature
       * point.
       *
       * We store the hessians in the quadrature points on the unit cell. We
       * then only have to apply the transformation when visiting an actual
       * cell.
       */
      Table<2, Tensor<2, dim>> shape_hessians;

      /**
       * Array with shape function third derivatives in quadrature points. There
       * is one row for each shape function, containing values for each
       * quadrature point.
       *
       * We store the third derivatives in the quadrature points on the unit
       * cell. We then only have to apply the transformation when visiting an
       * actual cell.
       */
      Table<2, Tensor<3, dim>> shape_3rd_derivatives;
    };
  };



  /**
   * Implementation of a scalar Lagrange finite element Pp that yields
   * the finite element space of continuous, piecewise polynomials of
   * degree p.
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
    std::unique_ptr<FiniteElement<dim, dim>>
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
    std::unique_ptr<FiniteElement<dim, dim>>
    clone() const override;

    /**
     * Return a string that uniquely identifies a finite element. This class
     * returns <tt>Simplex::FE_DGP<dim>(degree)</tt>, with @p dim and @p degree
     * replaced by appropriate values.
     */
    std::string
    get_name() const override;
  };



  template <int dim, int spacedim>
  std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
  FE_Poly<dim, spacedim>::get_data(
    const UpdateFlags update_flags,
    const Mapping<dim, spacedim> & /*mapping*/,
    const Quadrature<dim> &quadrature,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const
  {
    // generate a new data object and
    // initialize some fields
    std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
          data_ptr = std::make_unique<InternalData>();
    auto &data     = dynamic_cast<InternalData &>(*data_ptr);
    // data.update_each = requires_update_flags(update_flags); // TODO
    data.update_each = update_flags;

    const unsigned int n_q_points    = quadrature.size();
    const unsigned int dofs_per_cell = this->dofs_per_cell;

    // initialize some scratch arrays. we need them for the underlying
    // polynomial to put the values and derivatives of shape functions
    // to put there, depending on what the user requested
    std::vector<double> values(update_flags & update_values ? dofs_per_cell :
                                                              0);
    std::vector<Tensor<1, dim>> grads(
      update_flags & update_gradients ? dofs_per_cell : 0);
    std::vector<Tensor<2, dim>> grad_grads(
      update_flags & update_hessians ? dofs_per_cell : 0);
    std::vector<Tensor<3, dim>> third_derivatives(
      update_flags & update_3rd_derivatives ? dofs_per_cell : 0);
    std::vector<Tensor<4, dim>>
      fourth_derivatives; // won't be needed, so leave empty

    // now also initialize fields the fields of this class's own
    // temporary storage, depending on what we need for the given
    // update flags.
    //
    // there is one exception from the rule: if we are dealing with
    // cells (i.e., if this function is not called via
    // get_(sub)face_data()), then we can already store things in the
    // final location where FEValues::reinit() later wants to see
    // things. we then don't need the intermediate space. we determine
    // whether we are on a cell by asking whether the number of
    // elements in the output array equals the number of quadrature
    // points (yes, it's a cell) or not (because in that case the
    // number of quadrature points we use here equals the number of
    // quadrature points summed over *all* faces or subfaces, whereas
    // the number of output slots equals the number of quadrature
    // points on only *one* face)
    if ((update_flags & update_values) &&
        !((output_data.shape_values.n_rows() > 0) &&
          (output_data.shape_values.n_cols() == n_q_points)))
      data.shape_values.reinit(dofs_per_cell, n_q_points);

    if (update_flags & update_gradients)
      data.shape_gradients.reinit(dofs_per_cell, n_q_points);

    if (update_flags & update_hessians)
      data.shape_hessians.reinit(dofs_per_cell, n_q_points);

    if (update_flags & update_3rd_derivatives)
      data.shape_3rd_derivatives.reinit(dofs_per_cell, n_q_points);

    // next already fill those fields of which we have information by
    // now. note that the shape gradients are only those on the unit
    // cell, and need to be transformed when visiting an actual cell
    if (update_flags & (update_values | update_gradients | update_hessians |
                        update_3rd_derivatives))
      for (unsigned int i = 0; i < n_q_points; ++i)
        {
          this->poly_space->evaluate(quadrature.point(i),
                                     values,
                                     grads,
                                     grad_grads,
                                     third_derivatives,
                                     fourth_derivatives);

          // the values of shape functions at quadrature points don't change.
          // consequently, write these values right into the output array if
          // we can, i.e., if the output array has the correct size. this is
          // the case on cells. on faces, we already precompute data on *all*
          // faces and subfaces, but we later on copy only a portion of it
          // into the output object; in that case, copy the data from all
          // faces into the scratch object
          if (update_flags & update_values)
            if (output_data.shape_values.n_rows() > 0)
              {
                if (output_data.shape_values.n_cols() == n_q_points)
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    output_data.shape_values[k][i] = values[k];
                else
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    data.shape_values[k][i] = values[k];
              }

          // for everything else, derivatives need to be transformed,
          // so we write them into our scratch space and only later
          // copy stuff into where FEValues wants it
          if (update_flags & update_gradients)
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                data.shape_gradients[k][i] = grads[k];
              }

          if (update_flags & update_hessians)
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              data.shape_hessians[k][i] = grad_grads[k];

          if (update_flags & update_3rd_derivatives)
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              data.shape_3rd_derivatives[k][i] = third_derivatives[k];
        }
    return data_ptr;
  }


  template <int dim, int spacedim>
  std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
  FE_Poly<dim, spacedim>::get_face_data(
    const UpdateFlags             flags,
    const Mapping<dim, spacedim> &mapping,
    const Quadrature<dim - 1> &   quadrature,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const
  {
    return get_data(flags,
                    mapping,
                    PProjector<dim>::project_to_all_faces(quadrature),
                    output_data);
  }

} // namespace Simplex

DEAL_II_NAMESPACE_CLOSE

#endif
