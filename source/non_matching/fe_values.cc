// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2021 by the deal.II authors
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

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/trilinos_epetra_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_tpetra_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/non_matching/fe_values.h>

#include <deal.II/numerics/fe_field_function.h>

#include "../fe/fe_values.cc"

DEAL_II_NAMESPACE_OPEN

namespace NonMatching
{
  RegionUpdateFlags::RegionUpdateFlags()
    : inside(update_default)
    , outside(update_default)
    , surface(update_default)
  {}


  namespace internal
  {
    namespace FEValuesImplementation
    {
      template <int dim, class VECTOR>
      class RefSpaceFEFieldWrapper : public LevelSetDescription<dim>
      {
      public:
        RefSpaceFEFieldWrapper(const DoFHandler<dim> &dof_handler,
                               const VECTOR &         level_set);


        const Function<dim> &
        get_ref_space_level_set(
          const typename Triangulation<dim>::active_cell_iterator &cell)
          override;

      private:
        Functions::RefSpaceFEFieldFunction<dim, VECTOR> level_set;
      };



      template <int dim, class VECTOR>
      RefSpaceFEFieldWrapper<dim, VECTOR>::RefSpaceFEFieldWrapper(
        const DoFHandler<dim> &dof_handler,
        const VECTOR &         level_set)
        : level_set(dof_handler, level_set)
      {}



      template <int dim, class VECTOR>
      const Function<dim> &
      RefSpaceFEFieldWrapper<dim, VECTOR>::get_ref_space_level_set(
        const typename Triangulation<dim>::active_cell_iterator &cell)
      {
        level_set.set_active_cell(cell);
        return level_set;
      }
    } // namespace FEValuesImplementation
  }   // namespace internal



  template <int dim>
  template <class VECTOR>
  FEValues<dim>::FEValues(const hp::MappingCollection<dim> &mapping_collection,
                          const hp::FECollection<dim> &     fe_collection,
                          const hp::QCollection<dim> &      q_collection,
                          const hp::QCollection<1> &        q_collection_1D,
                          const RegionUpdateFlags           update_flags,
                          const MeshClassifier<dim> &       mesh_classifier,
                          const DoFHandler<dim> &           dof_handler,
                          const VECTOR &                    level_set,
                          const AdditionalData &            additional_data)
    : mapping_collection(&mapping_collection)
    , fe_collection(&fe_collection)
    , q_collection_1D(q_collection_1D)
    , region_update_flags(update_flags)
    , mesh_classifier(&mesh_classifier)
    , fe_values_inside_full_quadrature(mapping_collection,
                                       fe_collection,
                                       q_collection,
                                       region_update_flags.inside)
    , fe_values_outside_full_quadrature(mapping_collection,
                                        fe_collection,
                                        q_collection,
                                        region_update_flags.outside)
    , quadrature_generator(q_collection_1D, additional_data)
    , unit_box(create_unit_bounding_box<dim>())
    , level_set_description(
        std::make_unique<internal::FEValuesImplementation::
                           RefSpaceFEFieldWrapper<dim, VECTOR>>(dof_handler,
                                                                level_set))
  {
    current_cell_location = LocationToLevelSet::unassigned;
  }



  template <int dim>
  template <bool level_dof_access>
  void
  FEValues<dim>::reinit(
    const TriaIterator<DoFCellAccessor<dim, dim, level_dof_access>> &cell)
  {
    current_cell_location = mesh_classifier->location_to_level_set(cell);

    // These objects were created with a quadrature based on the previous cell
    // and are thus no longer valid.
    fe_values_inside.reset();
    fe_values_surface.reset();
    fe_values_outside.reset();

    switch (current_cell_location)
      {
        case LocationToLevelSet::inside:
          {
            fe_values_inside_full_quadrature.reinit(cell);
            break;
          }
        case LocationToLevelSet::outside:
          {
            fe_values_outside_full_quadrature.reinit(cell);
            break;
          }
        case LocationToLevelSet::intersected:
          {
            const unsigned int fe_index = cell->active_fe_index();

            const unsigned int mapping_index =
              mapping_collection->size() > 1 ? fe_index : 0;

            const unsigned int q1D_index =
              q_collection_1D.size() > 1 ? fe_index : 0;

            quadrature_generator.set_1D_quadrature(q1D_index);

            const Function<dim> &level_set =
              level_set_description->get_ref_space_level_set(cell);

            quadrature_generator.generate(level_set, unit_box);

            const Quadrature<dim> &inside_quadrature =
              quadrature_generator.get_inside_quadrature();
            const Quadrature<dim> &outside_quadrature =
              quadrature_generator.get_outside_quadrature();
            const ImmersedSurfaceQuadrature<dim> &surface_quadrature =
              quadrature_generator.get_surface_quadrature();

            // Even if a cell is formally intersected the number of created
            // quadrature points can be 0. Avoid creating an FEValues object if
            // that is the case.
            if (inside_quadrature.size() > 0)
              {
                fe_values_inside = std::make_unique<dealii::FEValues<dim>>(
                  (*mapping_collection)[mapping_index],
                  (*fe_collection)[fe_index],
                  inside_quadrature,
                  region_update_flags.inside);

                fe_values_inside->reinit(cell);
              }

            if (outside_quadrature.size() > 0)
              {
                fe_values_outside = std::make_unique<dealii::FEValues<dim>>(
                  (*mapping_collection)[mapping_index],
                  (*fe_collection)[fe_index],
                  outside_quadrature,
                  region_update_flags.outside);

                fe_values_outside->reinit(cell);
              }

            if (surface_quadrature.size() > 0)
              {
                fe_values_surface =
                  std::make_unique<FEImmersedSurfaceValues<dim>>(
                    (*mapping_collection)[mapping_index],
                    (*fe_collection)[fe_index],
                    surface_quadrature,
                    region_update_flags.surface);
                fe_values_surface->reinit(cell);
              }

            break;
          }
        default:
          {
            Assert(false, ExcInternalError());
            break;
          }
      }
  }



  template <int dim>
  boost::optional<const dealii::FEValues<dim> &>
  FEValues<dim>::get_inside_fe_values() const
  {
    boost::optional<const dealii::FEValues<dim> &> fe_values;

    if (current_cell_location == LocationToLevelSet::intersected)
      {
        // If the cut was to small for the fe_values object to be created the
        // smart pointer won't be set. Return it if it exists.
        if (fe_values_inside)
          fe_values.reset(*fe_values_inside);
      }
    else if (current_cell_location == LocationToLevelSet::inside)
      fe_values.reset(fe_values_inside_full_quadrature.get_present_fe_values());

    return fe_values;
  }



  template <int dim>
  boost::optional<const dealii::FEValues<dim> &>
  FEValues<dim>::get_outside_fe_values() const
  {
    boost::optional<const dealii::FEValues<dim> &> fe_values;

    if (current_cell_location == LocationToLevelSet::intersected)
      {
        // If the cut was to small for the fe_values object to be created the
        // smart pointer won't be set. Return it if it exists.
        if (fe_values_outside)
          fe_values.reset(*fe_values_outside);
      }
    else if (current_cell_location == LocationToLevelSet::outside)
      fe_values.reset(
        fe_values_outside_full_quadrature.get_present_fe_values());

    return fe_values;
  }



  template <int dim>
  boost::optional<const FEImmersedSurfaceValues<dim> &>
  FEValues<dim>::get_surface_fe_values() const
  {
    boost::optional<const FEImmersedSurfaceValues<dim> &> fe_values;

    if (current_cell_location == LocationToLevelSet::intersected)
      // If the cut was to small for the fe_values object to be created the
      // smart pointer won't be set. Return it if it exists.
      if (fe_values_surface)
        fe_values.reset(*fe_values_surface);

    return fe_values;
  }

} // namespace NonMatching



template <int dim>
FEImmersedSurfaceValues<dim>::FEImmersedSurfaceValues(
  const Mapping<dim> &                               mapping,
  const FiniteElement<dim> &                         element,
  const NonMatching::ImmersedSurfaceQuadrature<dim> &quadrature,
  const UpdateFlags                                  update_flags)
  : FEValuesBase<dim, dim>(quadrature.size(),
                           element.dofs_per_cell,
                           update_default,
                           mapping,
                           element)
  , quadrature(quadrature)
{
  initialize(update_flags);
}



template <int dim>
template <bool level_dof_access>
void
FEImmersedSurfaceValues<dim>::reinit(
  const TriaIterator<DoFCellAccessor<dim, dim, level_dof_access>> &cell)
{
  // Assert that the finite elements passed to the constructor and used by the
  // DoFHandler used by this cell, are the same
  Assert(static_cast<const FiniteElementData<dim> &>(*this->fe) ==
           static_cast<const FiniteElementData<dim> &>(cell->get_fe()),
         (typename FEValuesBase<dim>::ExcFEDontMatch()));

  this->maybe_invalidate_previous_present_cell(cell);
  this->check_cell_similarity(cell);

  reset_pointer_in_place_if_possible<
    typename FEValuesBase<dim, dim>::template CellIterator<
      TriaIterator<DoFCellAccessor<dim, dim, level_dof_access>>>>(
    this->present_cell, cell);

  // This was the part of the work that is dependent on the actual data type of
  // the iterator. Now pass on to the function doing the real work.
  do_reinit();
}



template <int dim>
void
FEImmersedSurfaceValues<dim>::do_reinit()
{
  // First call the mapping and let it generate the data specific to the
  // mapping.
  if (this->update_flags & update_mapping)
    {
      this->get_mapping().fill_fe_immersed_surface_values(*this->present_cell,
                                                          quadrature,
                                                          *this->mapping_data,
                                                          this->mapping_output);
    }

  // Call the finite element and, with the data already filled by the mapping,
  // let it compute the data for the mapped shape function values, gradients
  // etc.
  this->get_fe().fill_fe_values(*this->present_cell,
                                CellSimilarity::none,
                                this->quadrature,
                                this->get_mapping(),
                                *this->mapping_data,
                                this->mapping_output,
                                *this->fe_data,
                                this->finite_element_output);
}



template <int dim>
Tensor<1, dim>
FEImmersedSurfaceValues<dim>::shape_surface_grad(
  const unsigned int function_no,
  const unsigned int quadrature_point) const
{
  const unsigned int component = 0;
  return shape_surface_grad_component(function_no, quadrature_point, component);
}



template <int dim>
Tensor<1, dim>
FEImmersedSurfaceValues<dim>::shape_surface_grad_component(
  const unsigned int function_no,
  const unsigned int quadrature_point,
  const unsigned int component) const
{
  const Tensor<1, dim> gradient =
    this->shape_grad_component(function_no, quadrature_point, component);
  const Tensor<1, dim> &normal = this->normal_vector(quadrature_point);

  return gradient - (normal * gradient) * normal;
}



template <int dim>
const NonMatching::ImmersedSurfaceQuadrature<dim> &
FEImmersedSurfaceValues<dim>::get_quadrature() const
{
  return quadrature;
}



template <int dim>
inline void
FEImmersedSurfaceValues<dim>::initialize(const UpdateFlags update_flags)
{
  const UpdateFlags flags = this->compute_update_flags(update_flags);

  // Initialize the base classes.
  if (flags & update_mapping)
    this->mapping_output.initialize(this->n_quadrature_points, flags);
  this->finite_element_output.initialize(this->n_quadrature_points,
                                         *this->fe,
                                         flags);

  // Then get objects into which the FE and the Mapping can store
  // intermediate data used across calls to reinit. We can do this in parallel.
  Threads::Task<
    std::unique_ptr<typename FiniteElement<dim, dim>::InternalDataBase>>
    fe_get_data = Threads::new_task(&FiniteElement<dim, dim>::get_data,
                                    *this->fe,
                                    flags,
                                    *this->mapping,
                                    this->quadrature,
                                    this->finite_element_output);

  Threads::Task<std::unique_ptr<typename Mapping<dim>::InternalDataBase>>
    mapping_get_data;
  if (flags & update_mapping)
    mapping_get_data = Threads::new_task(&Mapping<dim>::get_data,
                                         *this->mapping,
                                         flags,
                                         this->quadrature);

  this->update_flags = flags;

  // Then collect answers from the two task above.
  this->fe_data = std::move(fe_get_data.return_value());
  if (flags & update_mapping)
    this->mapping_data = std::move(mapping_get_data.return_value());
  else
    this->mapping_data =
      std::make_unique<typename Mapping<dim>::InternalDataBase>();
}

#include "fe_values.inst"

DEAL_II_NAMESPACE_CLOSE
