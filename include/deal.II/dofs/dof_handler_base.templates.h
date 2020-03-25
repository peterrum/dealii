// ---------------------------------------------------------------------
//
// Copyright (C) 1999 - 2019 by the deal.II authors
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

#ifndef dealii_dof_handler_base_templates_h
#define dealii_dof_handler_base_templates_h

#include <deal.II/dofs/dof_handler_base.h>

DEAL_II_NAMESPACE_OPEN


template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::set_fe(const FiniteElement<dim, spacedim> &fe)
{
  this->set_fe(hp::FECollection<dim, spacedim>(fe));
}

template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::set_fe(
  const hp::FECollection<dim, spacedim> &fe)
{
  static_cast<T *>(this)->set_fe_impl(hp::FECollection<dim, spacedim>(fe));
}


/*------------------------ Cell iterator functions ------------------------*/

template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::cell_iterator
DoFHandlerBase<dim, spacedim, T>::begin(const unsigned int level) const
{
  typename Triangulation<dim, spacedim>::cell_iterator cell =
    this->get_triangulation().begin(level);
  if (cell == this->get_triangulation().end(level))
    return end(level);
  return cell_iterator(*cell, static_cast<const T *>(this));
}



template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator
DoFHandlerBase<dim, spacedim, T>::begin_active(const unsigned int level) const
{
  // level is checked in begin
  cell_iterator i = begin(level);
  if (i.state() != IteratorState::valid)
    return i;
  while (i->has_children())
    if ((++i).state() != IteratorState::valid)
      return i;
  return i;
}



template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::cell_iterator
DoFHandlerBase<dim, spacedim, T>::end() const
{
  return cell_iterator(&this->get_triangulation(),
                       -1,
                       -1,
                       static_cast<const T *>(this));
}


template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::cell_iterator
DoFHandlerBase<dim, spacedim, T>::end(const unsigned int level) const
{
  typename Triangulation<dim, spacedim>::cell_iterator cell =
    this->get_triangulation().end(level);
  if (cell.state() != IteratorState::valid)
    return end();
  return cell_iterator(*cell, static_cast<const T *>(this));
}


template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator
DoFHandlerBase<dim, spacedim, T>::end_active(const unsigned int level) const
{
  typename Triangulation<dim, spacedim>::cell_iterator cell =
    this->get_triangulation().end_active(level);
  if (cell.state() != IteratorState::valid)
    return active_cell_iterator(end());
  return active_cell_iterator(*cell, static_cast<const T *>(this));
}



template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator
DoFHandlerBase<dim, spacedim, T>::begin_mg(const unsigned int level) const
{
  // Assert(this->has_level_dofs(), ExcMessage("You can only iterate over mg "
  //     "levels if mg dofs got distributed."));
  typename Triangulation<dim, spacedim>::cell_iterator cell =
    this->get_triangulation().begin(level);
  if (cell == this->get_triangulation().end(level))
    return end_mg(level);
  return level_cell_iterator(*cell, static_cast<const T *>(this));
}


template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator
DoFHandlerBase<dim, spacedim, T>::end_mg(const unsigned int level) const
{
  // Assert(this->has_level_dofs(), ExcMessage("You can only iterate over mg "
  //     "levels if mg dofs got distributed."));
  typename Triangulation<dim, spacedim>::cell_iterator cell =
    this->get_triangulation().end(level);
  if (cell.state() != IteratorState::valid)
    return end();
  return level_cell_iterator(*cell, static_cast<const T *>(this));
}


template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator
DoFHandlerBase<dim, spacedim, T>::end_mg() const
{
  return level_cell_iterator(&this->get_triangulation(),
                             -1,
                             -1,
                             static_cast<const T *>(this));
}



template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::cell_iterator>
DoFHandlerBase<dim, spacedim, T>::cell_iterators() const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::cell_iterator>(begin(), end());
}


template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator>
DoFHandlerBase<dim, spacedim, T>::active_cell_iterators() const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator>(
    begin_active(), end());
}



template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator>
DoFHandlerBase<dim, spacedim, T>::mg_cell_iterators() const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator>(begin_mg(),
                                                                    end_mg());
}



template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::cell_iterator>
DoFHandlerBase<dim, spacedim, T>::cell_iterators_on_level(
  const unsigned int level) const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::cell_iterator>(begin(level),
                                                              end(level));
}



template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator>
DoFHandlerBase<dim, spacedim, T>::active_cell_iterators_on_level(
  const unsigned int level) const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator>(
    begin_active(level), end_active(level));
}



template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator>
DoFHandlerBase<dim, spacedim, T>::mg_cell_iterators_on_level(
  const unsigned int level) const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator>(
    begin_mg(level), end_mg(level));
}



//---------------------------------------------------------------------------


template <int dim, int spacedim, typename T>
inline const Triangulation<dim, spacedim> &
DoFHandlerBase<dim, spacedim, T>::get_triangulation() const
{
  Assert(tria != nullptr,
         ExcMessage("This DoFHandler object has not been associated "
                    "with a triangulation."));
  return *tria;
}


DEAL_II_NAMESPACE_CLOSE

#endif
