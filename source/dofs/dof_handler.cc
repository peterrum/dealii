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
{
  this->setup_policy();
}


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
DoFHandler<dim, spacedim>::initialize_impl(
  const Triangulation<dim, spacedim> &   t,
  const hp::FECollection<dim, spacedim> &fe)
{
  this->tria                       = &t;
  this->faces                      = nullptr;
  this->number_cache.n_global_dofs = 0;

  this->setup_policy();

  this->distribute_dofs(fe);
}



template <int dim, int spacedim>
std::size_t
DoFHandler<dim, spacedim>::memory_consumption() const
{
  std::size_t mem =
    (MemoryConsumption::memory_consumption(this->tria) +
     MemoryConsumption::memory_consumption(this->fe_collection) +
     MemoryConsumption::memory_consumption(this->block_info_object) +
     MemoryConsumption::memory_consumption(this->levels) +
     MemoryConsumption::memory_consumption(*this->faces) +
     MemoryConsumption::memory_consumption(this->faces) +
     sizeof(this->number_cache) +
     MemoryConsumption::memory_consumption(this->n_dofs()) +
     MemoryConsumption::memory_consumption(this->vertex_dofs));
  for (unsigned int i = 0; i < this->levels.size(); ++i)
    mem += MemoryConsumption::memory_consumption(*this->levels[i]);

  for (unsigned int level = 0; level < this->mg_levels.size(); ++level)
    mem += this->mg_levels[level]->memory_consumption();

  if (this->mg_faces != nullptr)
    mem += MemoryConsumption::memory_consumption(*this->mg_faces);

  for (unsigned int i = 0; i < this->mg_vertex_dofs.size(); ++i)
    mem += sizeof(typename Base::MGVertexDoFs) +
           (1 + this->mg_vertex_dofs[i].get_finest_level() -
            this->mg_vertex_dofs[i].get_coarsest_level()) *
             sizeof(types::global_dof_index);

  return mem;
}



template <int dim, int spacedim>
void
DoFHandler<dim, spacedim>::set_fe_impl(
  const hp::FECollection<dim, spacedim> &ff)
{
  Assert(
    this->tria != nullptr,
    ExcMessage(
      "You need to set the Triangulation in the DoFHandler using initialize() or "
      "in the constructor before you can distribute DoFs."));
  Assert(this->tria->n_levels() > 0,
         ExcMessage("The Triangulation you are using is empty!"));
  Assert(ff.size() > 0, ExcMessage("The hp::FECollection given is empty!"));

  // don't create a new object if the one we have is already appropriate
  if (this->fe_collection != ff)
    this->fe_collection = hp::FECollection<dim, spacedim>(ff);
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
void
DoFHandler<dim, spacedim>::renumber_dofs(
  const std::vector<types::global_dof_index> &new_numbers)
{
  Assert(this->levels.size() > 0,
         ExcMessage(
           "You need to distribute DoFs before you can renumber them."));

#ifdef DEBUG
  if (dynamic_cast<const parallel::shared::Triangulation<dim, spacedim> *>(
        &*this->tria) != nullptr)
    {
      Assert(new_numbers.size() == this->n_dofs() ||
               new_numbers.size() == this->n_locally_owned_dofs(),
             ExcMessage("Incorrect size of the input array."));
    }
  else if (dynamic_cast<
             const parallel::DistributedTriangulationBase<dim, spacedim> *>(
             &*this->tria) != nullptr)
    {
      AssertDimension(new_numbers.size(), this->n_locally_owned_dofs());
    }
  else
    {
      AssertDimension(new_numbers.size(), this->n_dofs());
    }

  // assert that the new indices are
  // consecutively numbered if we are
  // working on a single
  // processor. this doesn't need to
  // hold in the case of a parallel
  // mesh since we map the interval
  // [0...n_dofs()) into itself but
  // only globally, not on each
  // processor
  if (this->n_locally_owned_dofs() == this->n_dofs())
    {
      std::vector<types::global_dof_index> tmp(new_numbers);
      std::sort(tmp.begin(), tmp.end());
      std::vector<types::global_dof_index>::const_iterator p = tmp.begin();
      types::global_dof_index                              i = 0;
      for (; p != tmp.end(); ++p, ++i)
        Assert(*p == i, ExcNewNumbersNotConsecutive(i));
    }
  else
    for (const auto new_number : new_numbers)
      Assert(new_number < this->n_dofs(),
             ExcMessage(
               "New DoF index is not less than the total number of dofs."));
#endif

  this->number_cache = this->policy->renumber_dofs(new_numbers);
}


template <int dim, int spacedim>
void
DoFHandler<dim, spacedim>::renumber_dofs(
  const unsigned int                          level,
  const std::vector<types::global_dof_index> &new_numbers)
{
  Assert(
    this->mg_levels.size() > 0 && this->levels.size() > 0,
    ExcMessage(
      "You need to distribute active and level DoFs before you can renumber level DoFs."));
  AssertIndexRange(level, this->get_triangulation().n_global_levels());
  AssertDimension(new_numbers.size(),
                  this->locally_owned_mg_dofs(level).n_elements());

#ifdef DEBUG
  // assert that the new indices are consecutively numbered if we are working
  // on a single processor. this doesn't need to hold in the case of a
  // parallel mesh since we map the interval [0...n_dofs(level)) into itself
  // but only globally, not on each processor
  if (this->n_locally_owned_dofs() == this->n_dofs())
    {
      std::vector<types::global_dof_index> tmp(new_numbers);
      std::sort(tmp.begin(), tmp.end());
      std::vector<types::global_dof_index>::const_iterator p = tmp.begin();
      types::global_dof_index                              i = 0;
      for (; p != tmp.end(); ++p, ++i)
        Assert(*p == i, ExcNewNumbersNotConsecutive(i));
    }
  else
    for (const auto new_number : new_numbers)
      Assert(new_number < this->n_dofs(level),
             ExcMessage(
               "New DoF index is not less than the total number of dofs."));
#endif

  this->mg_number_cache[level] =
    this->policy->renumber_mg_dofs(level, new_numbers);
}



template <int dim, int spacedim>
unsigned int
DoFHandler<dim, spacedim>::max_couplings_between_dofs() const
{
  return internal::DoFHandlerImplementation::Implementation::
    max_couplings_between_dofs(*this);
}



template <int dim, int spacedim>
unsigned int
DoFHandler<dim, spacedim>::max_couplings_between_boundary_dofs() const
{
  switch (dim)
    {
      case 1:
        return this->get_fe().dofs_per_vertex;
      case 2:
        return (3 * this->get_fe().dofs_per_vertex +
                2 * this->get_fe().dofs_per_line);
      case 3:
        // we need to take refinement of
        // one boundary face into
        // consideration here; in fact,
        // this function returns what
        // #max_coupling_between_dofs<2>
        // returns
        //
        // we assume here, that only four
        // faces meet at the boundary;
        // this assumption is not
        // justified and needs to be
        // fixed some time. fortunately,
        // omitting it for now does no
        // harm since the matrix will cry
        // foul if its requirements are
        // not satisfied
        return (19 * this->get_fe().dofs_per_vertex +
                28 * this->get_fe().dofs_per_line +
                8 * this->get_fe().dofs_per_quad);
      default:
        Assert(false, ExcNotImplemented());
        return numbers::invalid_unsigned_int;
    }
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
