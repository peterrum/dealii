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

#ifndef dealii_dof_handler_base_h
#define dealii_dof_handler_base_h



#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/block_info.h>
#include <deal.II/dofs/deprecated_function_map.h>
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_iterator_selector.h>
#include <deal.II/dofs/dof_levels.h>
#include <deal.II/dofs/number_cache.h>

#include <deal.II/hp/fe_collection.h>

#include <boost/serialization/split_member.hpp>

#include <map>
#include <memory>
#include <set>
#include <vector>

DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
template <int dim, int spacedim>
class FiniteElement;
template <int dim, int spacedim>
class Triangulation;

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

template <int dim, int spacedim, typename T>
class DoFHandlerBase : public Subscriptor
{
  using ActiveSelector =
    dealii::internal::DoFHandlerImplementation::Iterators<T, false>;
  using LevelSelector =
    dealii::internal::DoFHandlerImplementation::Iterators<T, true>;

public:
  using cell_accessor = typename ActiveSelector::CellAccessor;

  using face_accessor = typename ActiveSelector::FaceAccessor;

  using line_iterator = typename ActiveSelector::line_iterator;

  using active_line_iterator = typename ActiveSelector::active_line_iterator;

  using quad_iterator = typename ActiveSelector::quad_iterator;

  using active_quad_iterator = typename ActiveSelector::active_quad_iterator;

  using hex_iterator = typename ActiveSelector::hex_iterator;

  using active_hex_iterator = typename ActiveSelector::active_hex_iterator;

  using active_cell_iterator = typename ActiveSelector::active_cell_iterator;

  using cell_iterator = typename ActiveSelector::cell_iterator;

  using face_iterator = typename ActiveSelector::face_iterator;

  using active_face_iterator = typename ActiveSelector::active_face_iterator;

  using level_cell_accessor = typename LevelSelector::CellAccessor;
  using level_face_accessor = typename LevelSelector::FaceAccessor;

  using level_cell_iterator = typename LevelSelector::cell_iterator;
  using level_face_iterator = typename LevelSelector::face_iterator;


  static const unsigned int dimension = dim;

  static const unsigned int space_dimension = spacedim;

  static const bool is_hp_dof_handler = false;

  DEAL_II_DEPRECATED
  static const types::global_dof_index invalid_dof_index =
    numbers::invalid_dof_index;

  static const unsigned int default_fe_index = 0;

  DoFHandlerBase()
    : tria(nullptr, typeid(*this).name())
  {}

  DoFHandlerBase(const Triangulation<dim, spacedim> &tria)
    : tria(&tria, typeid(*this).name())
  {}

  virtual void
  initialize(const Triangulation<dim, spacedim> &tria,
             const FiniteElement<dim, spacedim> &fe) = 0;
  virtual void
  initialize(const Triangulation<dim, spacedim> &   tria,
             const hp::FECollection<dim, spacedim> &fe) = 0;

  virtual void
  set_fe(const FiniteElement<dim, spacedim> &fe);

  virtual void
  set_fe(const hp::FECollection<dim, spacedim> &fe);

  virtual void
  distribute_dofs(const FiniteElement<dim, spacedim> &fe) = 0;

  virtual void
  distribute_dofs(const hp::FECollection<dim, spacedim> &fe) = 0;

  virtual void
  set_active_fe_indices(const std::vector<unsigned int> &active_fe_indices) = 0;

  virtual void
  get_active_fe_indices(std::vector<unsigned int> &active_fe_indices) const = 0;

  DEAL_II_DEPRECATED
  virtual void
  distribute_mg_dofs(const FiniteElement<dim, spacedim> &fe) = 0;

  virtual void
  distribute_mg_dofs() = 0;

  virtual bool
  has_level_dofs() const = 0;

  virtual bool
  has_active_dofs() const = 0;

  virtual void
  initialize_local_block_info() = 0;

  virtual void
  clear() = 0;

  virtual void
  renumber_dofs(const std::vector<types::global_dof_index> &new_numbers) = 0;

  virtual void
  renumber_dofs(const unsigned int                          level,
                const std::vector<types::global_dof_index> &new_numbers) = 0;

  virtual unsigned int
  max_couplings_between_dofs() const = 0;

  virtual unsigned int
  max_couplings_between_boundary_dofs() const = 0;

  virtual cell_iterator
  begin(const unsigned int level = 0) const;

  virtual active_cell_iterator
  begin_active(const unsigned int level = 0) const;

  virtual cell_iterator
  end() const;

  virtual cell_iterator
  end(const unsigned int level) const;

  virtual active_cell_iterator
  end_active(const unsigned int level) const;

  virtual level_cell_iterator
  begin_mg(const unsigned int level = 0) const;

  virtual level_cell_iterator
  end_mg(const unsigned int level) const;

  virtual level_cell_iterator
  end_mg() const;

  virtual IteratorRange<cell_iterator>
  cell_iterators() const;

  virtual IteratorRange<active_cell_iterator>
  active_cell_iterators() const;

  virtual IteratorRange<level_cell_iterator>
  mg_cell_iterators() const;

  virtual IteratorRange<cell_iterator>
  cell_iterators_on_level(const unsigned int level) const;

  virtual IteratorRange<active_cell_iterator>
  active_cell_iterators_on_level(const unsigned int level) const;

  virtual IteratorRange<level_cell_iterator>
  mg_cell_iterators_on_level(const unsigned int level) const;

  virtual types::global_dof_index
  n_dofs() const = 0;

  virtual types::global_dof_index
  n_dofs(const unsigned int level) const = 0;

  virtual types::global_dof_index
  n_boundary_dofs() const = 0;

  // virtual template <typename number>
  // types::global_dof_index
  // n_boundary_dofs(
  //  const std::map<types::boundary_id, const Function<spacedim, number> *>
  //    &boundary_ids) const = 0;

  virtual types::global_dof_index
  n_boundary_dofs(const std::set<types::boundary_id> &boundary_ids) const = 0;

  virtual const BlockInfo &
  block_info() const;

  virtual types::global_dof_index
  n_locally_owned_dofs() const = 0;

  virtual const IndexSet &
  locally_owned_dofs() const = 0;

  virtual const IndexSet &
  locally_owned_mg_dofs(const unsigned int level) const = 0;

  virtual std::vector<IndexSet>
  compute_locally_owned_dofs_per_processor() const = 0;

  virtual std::vector<types::global_dof_index>
  compute_n_locally_owned_dofs_per_processor() const = 0;

  virtual std::vector<IndexSet>
  compute_locally_owned_mg_dofs_per_processor(
    const unsigned int level) const = 0;

  DEAL_II_DEPRECATED virtual const std::vector<IndexSet> &
  locally_owned_dofs_per_processor() const = 0;

  DEAL_II_DEPRECATED virtual const std::vector<types::global_dof_index> &
  n_locally_owned_dofs_per_processor() const = 0;

  DEAL_II_DEPRECATED virtual const std::vector<IndexSet> &
  locally_owned_mg_dofs_per_processor(const unsigned int level) const = 0;

  virtual const FiniteElement<dim, spacedim> &
  get_fe(const unsigned int index = 0) const;

  virtual const hp::FECollection<dim, spacedim> &
  get_fe_collection() const;

  virtual const Triangulation<dim, spacedim> &
  get_triangulation() const;

  virtual std::size_t
  memory_consumption() const = 0;

  virtual void
  prepare_for_serialization_of_active_fe_indices() = 0;

  virtual void
  deserialize_active_fe_indices() = 0;

protected:
  BlockInfo block_info_object;

  SmartPointer<const Triangulation<dim, spacedim>, DoFHandler<dim, spacedim>>
    tria;

  hp::FECollection<dim, spacedim> fe_collection;

  std::unique_ptr<dealii::internal::DoFHandlerImplementation::Policy::
                    PolicyBase<dim, spacedim>>
    policy;
};



template <int dim, int spacedim, typename T>
inline const FiniteElement<dim, spacedim> &
DoFHandlerBase<dim, spacedim, T>::get_fe(const unsigned int number) const
{
  Assert(fe_collection.size() > 0,
         ExcMessage("No finite element collection is associated with "
                    "this DoFHandler"));
  return fe_collection[number];
}



template <int dim, int spacedim, typename T>
inline const hp::FECollection<dim, spacedim> &
DoFHandlerBase<dim, spacedim, T>::get_fe_collection() const
{
  Assert(fe_collection.size() > 0,
         ExcMessage("No finite element collection is associated with "
                    "this DoFHandler"));
  return fe_collection;
}



template <int dim, int spacedim, typename T>
inline const BlockInfo &
DoFHandlerBase<dim, spacedim, T>::block_info() const
{
  Assert(T::is_hp_dof_handler == false, ExcNotImplemented());

  return block_info_object;
}

DEAL_II_NAMESPACE_CLOSE

#endif
