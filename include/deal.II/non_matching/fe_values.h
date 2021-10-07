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

#ifndef dealii_non_matching_fe_values
#define dealii_non_matching_fe_values

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria_iterator.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/non_matching/mesh_classifier.h>
#include <deal.II/non_matching/quadrature_generator.h>

#include <boost/optional.hpp>

DEAL_II_NAMESPACE_OPEN

namespace NonMatching
{
  namespace internal
  {
    namespace FEValuesImplementation
    {
      template <int dim>
      class LevelSetDescription;
    }
  } // namespace internal


  /**
   * Struct storing UpdateFlags for the 3 regions of an cell, $K$, that is
   * defined by the sign of a level set function, $\psi$:
   * @f[
   * N = \{x \in K : \psi(x) < 0 \}, \\
   * P = \{x \in K : \psi(x) > 0 \}, \\
   * S = \{x \in K : \psi(x) = 0 \}.
   * @f]
   * As in the QuadratureGenerator class, we refer to $N$, $P$ and $S$ as the
   * inside, outside, and surface region. RegionUpdateFlags is used to describe
   * how the FEValues objects which are created by NonMatching::FEValues
   * should be updated.
   */
  struct RegionUpdateFlags
  {
    /**
     * Constructor, sets the UpdateFlags for each region to update_default.
     */
    RegionUpdateFlags();

    /**
     * Flags for the region $\{x \in K : \psi(x) < 0 \}$
     */
    UpdateFlags inside;

    /**
     * Flags for the region $\{x \in K : \psi(x) > 0 \}$
     */
    UpdateFlags outside;

    /**
     * Flags for the region $\{x \in K : \psi(x) = 0 \}$
     */
    UpdateFlags surface;
  };


  /**
   * This class intended to facilitate assembling in immersed (in the sense of
   * cut) finite element methods, when the domain is described by a level set
   * function, $\psi : \mathbb{R}^{dim} \to \mathbb{R}$. In this type of
   * methods, we typically need to integrate over 3 different regions of each
   * cell, $K$:
   * @f[
   * N = \{x \in K : \psi(x) < 0 \}, \\
   * P = \{x \in K : \psi(x) > 0 \}, \\
   * S = \{x \in K : \psi(x) = 0 \}.
   * @f]
   * Thus we need quadrature rules for these 3 regions:
   * @image html immersed_quadratures.svg
   * As in the QuadratureGenerator class, we refer to $N$, $P$ and $S$ as the
   * inside, outside, and surface region. The constructor of this class takes a
   * discrete level set function (a DoFHandler and a Vector). When the reinit()
   * function is called the QuadratureGenerator will be called in the background
   * to create these immersed quadrature rules. This class then creates
   * dealii::FEValues objects for the inside/outside region and an
   * FEImmersedSurface object for the surface region. These objects can then be
   * accessed through one of the functions:
   * get_inside_fe_values(),
   * get_outside_fe_values(), and
   * get_surface_fe_values().
   * Since a cut between a cell and the domain can be arbitrarily small, it can
   * happen that the underlying algorithm generates a quadrature rule with 0
   * points. This can, for example, happen if the cut is of floating point size.
   * Since the FEValues-like objects aren't allowed to contain 0 points, the
   * object that get_inside/outside/surface_fe_values() returns is wrapped in a
   * boost::optional. This requires us to check if the returned FEValues-like
   * object contains a value before we use it:
   * @code
   * NonMatching::FEValues<dim> fe_values(...);
   *
   * for (const auto &cell : dof_handler.active_cell_iterators())
   *  {
   *    fe_values.reinit(cell);
   *
   *    const boost::optional<const FEValues<dim> &> fe_values_inside =
   *      fe_values.get_inside_fe_values();
   *
   *    if (fe_values_inside)
   *      {
   *        // Assemble locally
   *        for (const unsigned int q_index :
   *             fe_values_inside->quadrature_point_indices())
   *          {
   *            // ...
   *          }
   *      }
   *  }
   * @endcode
   *
   * Of course, it is somewhat expensive to generate the immersed quadrature
   * rules and create FEValues objects with the generated quadratures. To reduce
   * the amount of work, the reinit() function of this class uses the
   * MeshClassifier passed to the constructor to check how the incoming cell
   * relates to the level set function. It only generates the immersed
   * quadrature rules if the cell is intersected. If the cell is completely
   * inside or outside it returns a cached FEValues object created with a
   * quadrature over the reference cell: $[0, 1]^{dim}$.
   */
  template <int dim>
  class FEValues
  {
  public:
    using AdditionalData = typename QuadratureGenerator<dim>::AdditionalData;

    /**
     * Constructor.
     *
     * @param mapping_collection Collection of Mappings to be used.
     * @param fe_collection Collection of FiniteElements to be used.
     * @param q_collection Collection of Quadrature rules over $[0, 1]^{dim}$
     * that should be used when a cell isn't intersected and we don't need to
     * generate immersed quadrature rules.
     * @param q_collection_1D Collection of 1-dimensional quadrature rules used
     * to generate the immersed quadrature rules. See the QuadratureGenerator
     * class.
     * @param mesh_classifier Object used to determine when the immersed
     * quadrature rules need to be generated.
     * @param region_update_flags Struct storing UpdateFlags for the
     * inside/outside/surface region of the cell.
     * @param dof_handler The DoFHandler associated with the discrete level set
     * function.
     * @param level_set The degrees of freedom of the discrete level set function.
     * @param additional_data Additional data passed to the QuadratureGenerator
     * class.
     *
     * @note Pointers to @p mapping_collection, @p fe_collection,
     * @p mesh_classifier, @p dof_handler, and @p level_set are stored
     * internally, so these need to have a longer life span than the instance of
     * this class.
     *
     * @note Only the case when @p mapping_collection contains MappingCartesian
     * is presently implemented.
     */
    template <class VECTOR>
    FEValues(const hp::MappingCollection<dim> &mapping_collection,
             const hp::FECollection<dim> &     fe_collection,
             const hp::QCollection<dim> &      q_collection,
             const hp::QCollection<1> &        q_collection_1D,
             const RegionUpdateFlags           region_update_flags,
             const MeshClassifier<dim> &       mesh_classifier,
             const DoFHandler<dim> &           dof_handler,
             const VECTOR &                    level_set,
             const AdditionalData &additional_data = AdditionalData());

    /**
     * Reinitialize the various FEValues-like objects for the 3 different
     * regions of the cell. After calling this function an FEValues-like object
     * can be retrieved for each region using the functions
     * get_inside_fe_values(),
     * get_outside_fe_values(),
     * get_surface_fe_values().
     */
    template <bool level_dof_access>
    void
    reinit(
      const TriaIterator<DoFCellAccessor<dim, dim, level_dof_access>> &cell);

    /**
     * Return an dealii::FEValues object reinitialized with a quadrature for the
     * inside region of the cell: $\{x \in K : \psi(x) < 0 \}$.
     *
     * @note If the quadrature rule over the region is empty the returned
     * optional will not contain a value.
     */
    boost::optional<const dealii::FEValues<dim> &>
    get_inside_fe_values() const;

    /**
     * Return an dealii::FEValues object reinitialized with a quadrature for the
     * outside region of the cell: $\{x \in K : \psi(x) > 0 \}$.
     *
     * @note If the quadrature rule over the region is empty the returned
     * optional will not contain a value.
     */
    boost::optional<const dealii::FEValues<dim> &>
    get_outside_fe_values() const;

    /**
     * Return an dealii::FEValues object reinitialized with a quadrature for the
     * surface region of the cell: $\{x \in K : \psi(x) = 0 \}$.
     *
     * @note If the quadrature rule over the region is empty the returned
     * optional will not contain a value.
     */
    boost::optional<const FEImmersedSurfaceValues<dim> &>
    get_surface_fe_values() const;

  private:
    /**
     * A pointer to the collection of mappings to be used.
     */
    const SmartPointer<const hp::MappingCollection<dim>> mapping_collection;

    /**
     * A pointer to the collection of finite elements to be used.
     */
    const SmartPointer<const hp::FECollection<dim>> fe_collection;

    /**
     * Collection of 1-dimensional quadrature rules that are used by
     * QuadratureGenerator as base for generating the immersed quadrature rules.
     */
    const hp::QCollection<1> q_collection_1D;

    /**
     * Location of the last cell that reinit was called with.
     */
    LocationToLevelSet current_cell_location;

    /**
     * The update flags passed to the constructor.
     */
    const RegionUpdateFlags region_update_flags;

    /**
     * Pointer to the MeshClassifier passed to the constructor.
     */
    const SmartPointer<const MeshClassifier<dim>> mesh_classifier;

    /**
     * hp::FEValues object created with the UpdateFlags for the inside region
     * and quadrature rules over the full reference cell: $[0, 1]^dim$, i.e.,
     * the QCollection passed to the constructor.
     *
     * This object is used to get dealii::FEValues objects when the cell is not
     * intersected and we don't need to generate immersed quadrature rules.
     */
    hp::FEValues<dim> fe_values_inside_full_quadrature;

    /**
     * hp::FEValues object created with the UpdateFlags for the outside region
     * and quadrature rules over the full reference cell: $[0, 1]^dim$, i.e.,
     * the QCollection passed to the constructor.
     *
     * This object is used to get dealii::FEValues objects when the cell is not
     * intersected and we don't need to generate immersed quadrature rules.
     */
    hp::FEValues<dim> fe_values_outside_full_quadrature;

    /**
     * Pointer to an FEValues object created with a quadrature integrating over
     * the inside region, $\{x \in B : \psi(x) < 0 \}$, that was generated in
     * the last call to reinit(..). If the cell in the last call was not
     * intersected or if 0 quadrature points were generated, the unique_ptr will
     * we empty.
     */
    std::unique_ptr<dealii::FEValues<dim>> fe_values_inside;

    /**
     * Pointer to an FEValues object created with a quadrature integrating over
     * the inside region, $\{x \in B : \psi(x) > 0 \}$, that was generated in
     * the last call to reinit(..). If the cell in the last call was not
     * intersected or if 0 quadrature points were generated, the unique_ptr will
     * we empty.
     */
    std::unique_ptr<dealii::FEValues<dim>> fe_values_outside;

    /**
     * Pointer to an FEImmersedSurfaceValues object created with a quadrature
     * integrating over the surface region, $\{x \in B : \psi(x) = 0 \}$, that
     * was generated in the last call to reinit(..). If the cell in the last
     * call was not intersected or if 0 quadrature points were generated, the
     * unique_ptr will we empty.
     */
    std::unique_ptr<FEImmersedSurfaceValues<dim>> fe_values_surface;

    /**
     * Object that generates the immersed quadrature rules.
     */
    QuadratureGenerator<dim> quadrature_generator;

    /**
     * A box corresponding to the reference cell $[0, 1]^{dim}$.
     */
    const BoundingBox<dim> unit_box;

    /**
     * Function that describes our level set function in reference space.
     */
    const std::unique_ptr<
      internal::FEValuesImplementation::LevelSetDescription<dim>>
      level_set_description;
  };


  namespace internal
  {
    namespace FEValuesImplementation
    {
      template <int dim>
      class LevelSetDescription
      {
      public:
        virtual ~LevelSetDescription() = default;

        virtual const Function<dim> &
        get_ref_space_level_set(
          const typename Triangulation<dim>::active_cell_iterator &cell) = 0;
      };
    } // namespace FEValuesImplementation
  }   // namespace internal

} // namespace NonMatching



/**
 * Finite element evaluated in the quadrature points of an
 * ImmersedSurfaceQuadrature of a cell.
 *
 * The shape functions values and their derivatives are the same as for an
 * FEValues-object, but the JxW-values are computed with the transformation
 * described in the documentation of ImmersedSurfaceQuadrature. Further, the
 * normal_vector-function returns the normal to the immersed surface.
 *
 * The reinit-function of this class exist mostly to be consistent with the
 * other FEValues-like classes. The immersed quadrature rule will typically vary
 * between each cell of the triangulation. Thus, an FEImmersedSurfaceValues
 * object can, typically, not be reused for different cells.
 *
 * See also documentation in FEValuesBase.
 *
 * @ingroup feaccess
 */
template <int dim>
class FEImmersedSurfaceValues : public FEValuesBase<dim, dim>
{
public:
  /**
   * Constructor. Gets cell independent data from mapping and finite element
   * objects, matching the quadrature rule and update flags.
   *
   * @note Currently this class is only implemented for MappingCartesian.
   */
  FEImmersedSurfaceValues(
    const Mapping<dim> &                               mapping,
    const FiniteElement<dim> &                         element,
    const NonMatching::ImmersedSurfaceQuadrature<dim> &quadrature,
    const UpdateFlags                                  update_flags);

  /**
   * Reinitialize the gradients, Jacobi determinants, etc for the incoming @p cell.
   */
  template <bool level_dof_access>
  void
  reinit(const TriaIterator<DoFCellAccessor<dim, dim, level_dof_access>> &cell);


  /**
   * Returns the surface gradient of the shape function with index
   * @p function_no at the quadrature point with index @p quadrature_point.
   *
   * The surface gradient is defined as the projection of the gradient to the
   * tangent plane of the surface:
   * $ \nabla u - (n \cdot \nabla u) n $,
   * where $n$ is the unit normal to the surface.
   *
   * @dealiiRequiresUpdateFlags{update_gradients | update_normal_vectors}
   */
  Tensor<1, dim>
  shape_surface_grad(const unsigned int function_no,
                     const unsigned int quadrature_point) const;

  /**
   * Return one vector component of the surface gradient of the shape function
   * at a quadrature point. See the definition of the surface gradient in the
   * shape_surface_grad function.
   *
   * @p function_no Index of the shape function to be evaluated.
   *
   * @p point_no Index of the quadrature point at which the function is to be
   * evaluated.
   *
   * @p component Vector component to be evaluated.
   *
   * @dealiiRequiresUpdateFlags{update_gradients | update_normal_vectors}
   */
  Tensor<1, dim>
  shape_surface_grad_component(const unsigned int function_no,
                               const unsigned int quadrature_point,
                               const unsigned int component) const;

  /**
   * Return a reference to the copy of the quadrature formula stored by this
   * object.
   */
  const NonMatching::ImmersedSurfaceQuadrature<dim> &
  get_quadrature() const;

protected:
  /**
   * Do work common to the constructors.
   */
  void
  initialize(const UpdateFlags update_flags);

  /**
   * The reinit() functions do only that part of the work that requires
   * knowledge of the type of iterator. After setting present_cell(), they
   * pass on to this function, which does the real work, and which is
   * independent of the actual type of the cell iterator.
   */
  void
  do_reinit();

  /**
   * Copy of the quadrature formula that was passed to the constructor.
   */
  const NonMatching::ImmersedSurfaceQuadrature<dim> quadrature;
};

DEAL_II_NAMESPACE_CLOSE

#endif
