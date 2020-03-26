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
DoFHandlerBase<dim, spacedim, T>::initialize(
  const Triangulation<dim, spacedim> &tria,
  const FiniteElement<dim, spacedim> &fe)
{
  this->initialize(tria, hp::FECollection<dim, spacedim>(fe));
}

template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::initialize(
  const Triangulation<dim, spacedim> &   tria,
  const hp::FECollection<dim, spacedim> &fe)
{
  static_cast<T *>(this)->initialize_impl(tria,
                                          hp::FECollection<dim, spacedim>(fe));
}


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


template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::distribute_dofs(
  const FiniteElement<dim, spacedim> &fe)
{
  this->distribute_dofs(hp::FECollection<dim, spacedim>(fe));
}

template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::distribute_dofs(
  const hp::FECollection<dim, spacedim> &fe)
{
  static_cast<T *>(this)->distribute_dofs_impl(fe);
}


template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::distribute_mg_dofs(
  const FiniteElement<dim, spacedim> &fe)
{
  (void)fe;
  this->distribute_mg_dofs();
}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::distribute_mg_dofs(
  const hp::FECollection<dim, spacedim> &fe)
{
  (void)fe;
  this->distribute_mg_dofs();
}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::distribute_mg_dofs()
{
  static_cast<T *>(this)->distribute_mg_dofs_impl();
}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::set_active_fe_indices(
  const std::vector<unsigned int> &active_fe_indices)
{
  Assert(active_fe_indices.size() == this->get_triangulation().n_active_cells(),
         ExcDimensionMismatch(active_fe_indices.size(),
                              this->get_triangulation().n_active_cells()));

  static_cast<T *>(this)->create_active_fe_table();
  // we could set the values directly, since they are stored as
  // protected data of this object, but for simplicity we use the
  // cell-wise access. this way we also have to pass some debug-mode
  // tests which we would have to duplicate ourselves otherwise
  for (const auto &cell : this->active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(active_fe_indices[cell->active_cell_index()]);
}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::get_active_fe_indices(
  std::vector<unsigned int> &active_fe_indices) const
{
  active_fe_indices.resize(this->get_triangulation().n_active_cells());

  // we could try to extract the values directly, since they are
  // stored as protected data of this object, but for simplicity we
  // use the cell-wise access.
  for (const auto &cell : this->active_cell_iterators())
    if (!cell->is_artificial())
      active_fe_indices[cell->active_cell_index()] = cell->active_fe_index();
}


DEAL_II_NAMESPACE_CLOSE

#endif
