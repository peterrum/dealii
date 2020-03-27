// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2019 by the deal.II authors
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

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/std_cxx14/memory.h>

#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_handler_base.templates.h>
#include <deal.II/dofs/dof_handler_policy.h>
#include <deal.II/dofs/dof_levels.h>

#include <deal.II/fe/fe.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_levels.h>

#include <algorithm>
#include <set>
#include <unordered_set>

DEAL_II_NAMESPACE_OPEN

// reference the invalid_dof_index variable explicitly to work around
// a bug in the icc8 compiler
namespace internal
{
  template <int dim, int spacedim>
  const types::global_dof_index *
  dummy()
  {
    return &dealii::numbers::invalid_dof_index;
  }
} // namespace internal



template <int dim, int spacedim>
const unsigned int DoFHandler<dim, spacedim>::dimension;

template <int dim, int spacedim>
const unsigned int DoFHandler<dim, spacedim>::space_dimension;

template <int dim, int spacedim>
const types::global_dof_index DoFHandler<dim, spacedim>::invalid_dof_index;

template <int dim, int spacedim>
const unsigned int DoFHandler<dim, spacedim>::default_fe_index;



template <int dim, int spacedim>
DoFHandler<dim, spacedim>::DoFHandler()
  : Base()
{}



template <int dim, int spacedim>
DoFHandler<dim, spacedim>::DoFHandler(const Triangulation<dim, spacedim> &tria)
  : Base(tria)
{}


template <int dim, int spacedim>
DoFHandler<dim, spacedim>::~DoFHandler()
{
  // release allocated memory
  // virtual functions called in constructors and destructors never use the
  // override in a derived class
  // for clarity be explicit on which function is called
  DoFHandler<dim, spacedim>::clear();

  // also release the policy. this needs to happen before the
  // current object disappears because the policy objects
  // store references to the DoFhandler object they work on
  this->policy.reset();
}



template <int dim, int spacedim>
void
DoFHandler<dim, spacedim>::distribute_dofs_impl(
  const hp::FECollection<dim, spacedim> &ff)
{
  // first, assign the finite_element
  this->set_fe(ff);

  // delete all levels and set them up newly. note that we still have to
  // allocate space for all degrees of freedom on this mesh (including ghost and
  // cells that are entirely stored on different processors), though we may not
  // assign numbers to some of them (i.e. they will remain at
  // invalid_dof_index). We need to allocate the space because we will want to
  // be able to query the dof_indices on each cell, and simply be told that we
  // don't know them on some cell (i.e. get back invalid_dof_index)
  clear_space();
  internal::DoFHandlerImplementation::Implementation::reserve_space(*this);

  // hand things off to the policy
  this->number_cache = this->policy->distribute_dofs();

  // initialize the block info object only if this is a sequential
  // triangulation. it doesn't work correctly yet if it is parallel
  if (dynamic_cast<const parallel::DistributedTriangulationBase<dim, spacedim>
                     *>(&*this->tria) == nullptr)
    this->block_info_object.initialize(*this, false, true);
}



template <int dim, int spacedim>
void
DoFHandler<dim, spacedim>::distribute_mg_dofs_impl()
{
  Assert(
    this->levels.size() > 0,
    ExcMessage(
      "Distribute active DoFs using distribute_dofs() before calling distribute_mg_dofs()."));

  Assert(
    ((this->tria->get_mesh_smoothing() &
      Triangulation<dim, spacedim>::limit_level_difference_at_vertices) !=
     Triangulation<dim, spacedim>::none),
    ExcMessage(
      "The mesh smoothing requirement 'limit_level_difference_at_vertices' has to be set for using multigrid!"));

  this->clear_mg_space();

  internal::DoFHandlerImplementation::Implementation::reserve_space_mg(*this);
  this->mg_number_cache = this->policy->distribute_mg_dofs();

  // initialize the block info object
  // only if this is a sequential
  // triangulation. it doesn't work
  // correctly yet if it is parallel
  if (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
        &*this->tria) == nullptr)
    this->block_info_object.initialize(*this, true, false);
}


template <int dim, int spacedim>
void
DoFHandler<dim, spacedim>::initialize_local_block_info()
{
  this->block_info_object.initialize_local(*this);
}



template <int dim, int spacedim>
void
DoFHandler<dim, spacedim>::clear()
{
  // release memory
  clear_space();
  this->clear_mg_space();
}



template <int dim, int spacedim>
unsigned int
DoFHandler<dim, spacedim>::max_couplings_between_dofs() const
{
  return internal::DoFHandlerImplementation::Implementation::
    max_couplings_between_dofs(*this);
}



template <int dim, int spacedim>
void
DoFHandler<dim, spacedim>::clear_space()
{
  this->levels.clear();
  this->faces.reset();

  std::vector<types::global_dof_index> tmp;
  std::swap(this->vertex_dofs, tmp);

  this->number_cache.clear();
}


/*-------------- Explicit Instantiations -------------------------------*/
#include "dof_handler.inst"


DEAL_II_NAMESPACE_CLOSE
