// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2019 by the deal.II authors
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

#ifndef dealii_hp_dof_handler_h
#define dealii_hp_dof_handler_h



#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/template_constraints.h>

#include <deal.II/dofs/deprecated_function_map.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler_base.h>
#include <deal.II/dofs/dof_iterator_selector.h>
#include <deal.II/dofs/number_cache.h>

#include <deal.II/hp/dof_faces.h>
#include <deal.II/hp/dof_level.h>
#include <deal.II/hp/fe_collection.h>

#include <map>
#include <memory>
#include <set>
#include <vector>

DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
template <int dim, int spacedim>
class Triangulation;

namespace parallel
{
  namespace distributed
  {
    template <int dim, int spacedim, typename VectorType>
    class CellDataTransfer;
  }
} // namespace parallel

namespace internal
{
  namespace DoFHandlerImplementation
  {
    struct Implementation;

    namespace Policy
    {
      template <int dim, int spacedim>
      class PolicyBase;
      struct Implementation;
    } // namespace Policy
  }   // namespace DoFHandlerImplementation

  namespace hp
  {
    class DoFLevel;

    namespace DoFHandlerImplementation
    {
      struct Implementation;
    }
  } // namespace hp
} // namespace internal

namespace internal
{
  namespace DoFAccessorImplementation
  {
    struct Implementation;
  }

  namespace DoFCellAccessorImplementation
  {
    struct Implementation;
  }
} // namespace internal
#endif


namespace hp
{
  template <int dim, int spacedim = dim>
  class DoFHandler
    : public DoFHandlerBase<dim, spacedim, DoFHandler<dim, spacedim>>
  {
    using Base = DoFHandlerBase<dim, spacedim, DoFHandler<dim, spacedim>>;

  public:
    static const unsigned int dimension         = Base::dimension;
    static const unsigned int space_dimension   = Base::space_dimension;
    static const bool         is_hp_dof_handler = true; // TODO
    static const unsigned int default_fe_index  = Base::default_fe_index;


    DoFHandler();

    DoFHandler(const Triangulation<dim, spacedim> &tria);

    DoFHandler(const DoFHandler &) = delete;

    DoFHandler &
    operator=(const DoFHandler &) = delete;

  private:
    // Make accessor objects friends.
    template <int, class, bool>
    friend class dealii::DoFAccessor;
    template <class, bool>
    friend class dealii::DoFCellAccessor;
    friend struct dealii::internal::DoFAccessorImplementation::Implementation;
    friend struct dealii::internal::DoFCellAccessorImplementation::
      Implementation;

    // Likewise for DoFLevel objects since they need to access the vertex dofs
    // in the functions that set and retrieve vertex dof indices.
    template <int>
    friend class dealii::internal::hp::DoFIndicesOnFacesOrEdges;
    friend struct dealii::internal::hp::DoFHandlerImplementation::
      Implementation;
    friend struct dealii::internal::DoFHandlerImplementation::Policy::
      Implementation;
  };

} // namespace hp

DEAL_II_NAMESPACE_CLOSE

#endif
