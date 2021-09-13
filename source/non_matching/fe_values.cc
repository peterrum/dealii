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

#include "fe_values.inst"

DEAL_II_NAMESPACE_CLOSE
