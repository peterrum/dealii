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
    get_face_quadrature_collection(const Quadrature<dim> &quad)
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

      AssertThrow(false, ExcNotImplemented());

      return {ReferenceCell::Type::Invalid, dealii::hp::QCollection<dim - 1>()};
    }

    template <int dim>
    inline std::pair<Quadrature<dim - 1>, Quadrature<dim - 1>>
    get_unique_face_quadratures(const Quadrature<dim> &quad)
    {
      if (dim == 2 || dim == 3)
        for (unsigned int i = 1; i <= 3; ++i)
          if (quad == Simplex::QGauss<dim>(i))
            {
              if (dim == 2)
                return {Simplex::QGauss<dim - 1>(i), Quadrature<dim - 1>()};
              else
                return {Quadrature<dim - 1>(), Simplex::QGauss<dim - 1>(i)};
            }

      if (dim == 3)
        for (unsigned int i = 1; i <= 3; ++i)
          if (quad == Simplex::QGaussWedge<dim>(i))
            return {QGauss<dim - 1>(i), Simplex::QGauss<dim - 1>(i)};

      if (dim == 3)
        for (unsigned int i = 1; i <= 2; ++i)
          if (quad == Simplex::QGaussPyramid<dim>(i))
            return {QGauss<dim - 1>(i), Simplex::QGauss<dim - 1>(i)};

      AssertThrow(false, ExcNotImplemented());

      return {QGauss<dim - 1>(1), QGauss<dim - 1>(1)};
    }



    template <int dim, int spacedim = dim>
    class FE_Dummy : public FiniteElement<dim, spacedim>
    {
    public:
      FE_Dummy(const ReferenceCell::Type &type);

      virtual std::unique_ptr<FiniteElement<dim, spacedim>>
      clone() const override;

      virtual std::string
      get_name() const override;

      virtual UpdateFlags
      requires_update_flags(const UpdateFlags update_flags) const override;

      virtual std::unique_ptr<
        typename FiniteElement<dim, spacedim>::InternalDataBase>
      get_data(
        const UpdateFlags             update_flags,
        const Mapping<dim, spacedim> &mapping,
        const Quadrature<dim> &       quadrature,
        dealii::internal::FEValuesImplementation::
          FiniteElementRelatedData<dim, spacedim> &output_data) const override;

      virtual void
      fill_fe_values(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell,
        const CellSimilarity::Similarity cell_similarity,
        const Quadrature<dim> &          quadrature,
        const Mapping<dim, spacedim> &   mapping,
        const typename Mapping<dim, spacedim>::InternalDataBase
          &mapping_internal,
        const dealii::internal::FEValuesImplementation::
          MappingRelatedData<dim, spacedim> &mapping_data,
        const typename FiniteElement<dim, spacedim>::InternalDataBase
          &fe_internal,
        dealii::internal::FEValuesImplementation::
          FiniteElementRelatedData<dim, spacedim> &output_data) const override;

      using FiniteElement<dim, spacedim>::fill_fe_face_values;

      virtual void
      fill_fe_face_values(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell,
        const unsigned int                                          face_no,
        const dealii::hp::QCollection<dim - 1> &                    quadrature,
        const Mapping<dim, spacedim> &                              mapping,
        const typename Mapping<dim, spacedim>::InternalDataBase
          &mapping_internal,
        const dealii::internal::FEValuesImplementation::
          MappingRelatedData<dim, spacedim> &mapping_data,
        const typename FiniteElement<dim, spacedim>::InternalDataBase
          &fe_internal,
        dealii::internal::FEValuesImplementation::
          FiniteElementRelatedData<dim, spacedim> &output_data) const override;

      virtual void
      fill_fe_subface_values(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell,
        const unsigned int                                          face_no,
        const unsigned int                                          sub_no,
        const Quadrature<dim - 1> &                                 quadrature,
        const Mapping<dim, spacedim> &                              mapping,
        const typename Mapping<dim, spacedim>::InternalDataBase
          &mapping_internal,
        const dealii::internal::FEValuesImplementation::
          MappingRelatedData<dim, spacedim> &mapping_data,
        const typename FiniteElement<dim, spacedim>::InternalDataBase
          &fe_internal,
        dealii::internal::FEValuesImplementation::
          FiniteElementRelatedData<dim, spacedim> &output_data) const override;
    };


    template <int dim, int spacedim>
    inline FE_Dummy<dim, spacedim>::FE_Dummy(const ReferenceCell::Type &type)
      : FiniteElement<dim, spacedim>(
          FiniteElementData<dim>(std::vector<unsigned>(dim + 1, 0),
                                 type,
                                 1 /*n_components*/,
                                 0,
                                 FiniteElementData<dim>::unknown),
          std::vector<bool>(),
          std::vector<ComponentMask>())
    {}


    template <int dim, int spacedim>
    inline std::unique_ptr<FiniteElement<dim, spacedim>>
    FE_Dummy<dim, spacedim>::clone() const
    {
      return std::make_unique<FE_Dummy<dim, spacedim>>(*this);
    }



    template <int dim, int spacedim>
    inline std::string
    FE_Dummy<dim, spacedim>::get_name() const
    {
      std::ostringstream namebuf;
      namebuf << "FE_Dummy<" << dim << ">()";
      return namebuf.str();
    }



    template <int dim, int spacedim>
    std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
    FE_Dummy<dim, spacedim>::get_data(
      const UpdateFlags /*update_flags*/,
      const Mapping<dim, spacedim> & /*mapping*/,
      const Quadrature<dim> & /*quadrature*/,
      dealii::internal::FEValuesImplementation::
        FiniteElementRelatedData<dim, spacedim> & /*output_data*/) const
    {
      // Create a default data object.  Normally we would then
      // need to resize things to hold the appropriate numbers
      // of dofs, but in this case all data fields are empty.
      return std::make_unique<
        typename FiniteElement<dim, spacedim>::InternalDataBase>();
    }



    template <int dim, int spacedim>
    UpdateFlags
    FE_Dummy<dim, spacedim>::requires_update_flags(
      const UpdateFlags flags) const
    {
      return flags;
    }



    template <int dim, int spacedim>
    void
    FE_Dummy<dim, spacedim>::fill_fe_values(
      const typename Triangulation<dim, spacedim>::cell_iterator &,
      const CellSimilarity::Similarity,
      const Quadrature<dim> &,
      const Mapping<dim, spacedim> &,
      const typename Mapping<dim, spacedim>::InternalDataBase &,
      const dealii::internal::FEValuesImplementation::
        MappingRelatedData<dim, spacedim> &,
      const typename FiniteElement<dim, spacedim>::InternalDataBase &,
      dealii::internal::FEValuesImplementation::
        FiniteElementRelatedData<dim, spacedim> &) const
    {
      // leave data fields empty
    }



    template <int dim, int spacedim>
    void
    FE_Dummy<dim, spacedim>::fill_fe_face_values(
      const typename Triangulation<dim, spacedim>::cell_iterator &,
      const unsigned int,
      const dealii::hp::QCollection<dim - 1> &,
      const Mapping<dim, spacedim> &,
      const typename Mapping<dim, spacedim>::InternalDataBase &,
      const dealii::internal::FEValuesImplementation::
        MappingRelatedData<dim, spacedim> &,
      const typename FiniteElement<dim, spacedim>::InternalDataBase &,
      dealii::internal::FEValuesImplementation::
        FiniteElementRelatedData<dim, spacedim> &) const
    {
      // leave data fields empty
    }



    template <int dim, int spacedim>
    void
    FE_Dummy<dim, spacedim>::fill_fe_subface_values(
      const typename Triangulation<dim, spacedim>::cell_iterator &,
      const unsigned int,
      const unsigned int,
      const Quadrature<dim - 1> &,
      const Mapping<dim, spacedim> &,
      const typename Mapping<dim, spacedim>::InternalDataBase &,
      const dealii::internal::FEValuesImplementation::
        MappingRelatedData<dim, spacedim> &,
      const typename FiniteElement<dim, spacedim>::InternalDataBase &,
      dealii::internal::FEValuesImplementation::
        FiniteElementRelatedData<dim, spacedim> &) const
    {
      // leave data fields empty
    }


  } // end of namespace MatrixFreeFunctions
} // end of namespace internal

DEAL_II_NAMESPACE_CLOSE

#endif
