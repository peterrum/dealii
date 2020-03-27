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
#include <deal.II/base/std_cxx14/memory.h>

#include <deal.II/distributed/cell_data_transfer.templates.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/block_info.h>
#include <deal.II/dofs/deprecated_function_map.h>
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_iterator_selector.h>
#include <deal.II/dofs/dof_levels.h>
#include <deal.II/dofs/number_cache.h>

#include <deal.II/hp/dof_faces.h>
#include <deal.II/hp/dof_level.h>
#include <deal.II/hp/fe_collection.h>

#include <boost/serialization/split_member.hpp>

#include <map>
#include <memory>
#include <set>
#include <unordered_set>
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

  namespace hp
  {
    class DoFLevel;

    namespace DoFHandlerImplementation
    {
      struct Implementation;
    }
  } // namespace hp
} // namespace internal

namespace parallel
{
  namespace distributed
  {
    template <int dim, int spacedim, typename VectorType>
    class CellDataTransfer;
  }
} // namespace parallel
#endif

/**
 * @author Wolfgang Bangerth, Markus Buerg, Marc Fehling, Timo Heister,
 *   Oliver Kayser-Herold, Guido Kanschat, Peter Munch, 1998, 1999, 2000, 2003,
 *   2004, 2012, 2017, 2018, 2020
 */
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

  static const bool is_hp_dof_handler = T::is_hp_dof_handler;

  static const unsigned int default_fe_index =
    is_hp_dof_handler ? numbers::invalid_unsigned_int : 0;

  DoFHandlerBase();

  DoFHandlerBase(const Triangulation<dim, spacedim> &tria);

  DoFHandlerBase(const DoFHandlerBase &) = delete;

  DoFHandlerBase &
  operator=(const DoFHandlerBase &) = delete;

  virtual ~DoFHandlerBase();

  void
  initialize(const Triangulation<dim, spacedim> &tria,
             const FiniteElement<dim, spacedim> &fe);
  void
  initialize(const Triangulation<dim, spacedim> &   tria,
             const hp::FECollection<dim, spacedim> &fe);

  void
  set_fe(const FiniteElement<dim, spacedim> &fe);

  void
  set_fe(const hp::FECollection<dim, spacedim> &fe);

  void
  distribute_dofs(const FiniteElement<dim, spacedim> &fe);

  void
  distribute_dofs(const hp::FECollection<dim, spacedim> &fe);

  void
  set_active_fe_indices(const std::vector<unsigned int> &active_fe_indices);

  void
  get_active_fe_indices(std::vector<unsigned int> &active_fe_indices) const;

  DEAL_II_DEPRECATED
  void
  distribute_mg_dofs(const FiniteElement<dim, spacedim> &fe);

  DEAL_II_DEPRECATED
  void
  distribute_mg_dofs(const hp::FECollection<dim, spacedim> &fe);

  void
  distribute_mg_dofs();

  bool
  has_level_dofs() const;

  bool
  has_active_dofs() const;

  void
  initialize_local_block_info();

  void
  clear();

  void
  renumber_dofs(const std::vector<types::global_dof_index> &new_numbers);

  void
  renumber_dofs(const unsigned int                          level,
                const std::vector<types::global_dof_index> &new_numbers);

  unsigned int
  max_couplings_between_dofs() const;

  unsigned int
  max_couplings_between_boundary_dofs() const;

  cell_iterator
  begin(const unsigned int level = 0) const;

  active_cell_iterator
  begin_active(const unsigned int level = 0) const;

  cell_iterator
  end() const;

  cell_iterator
  end(const unsigned int level) const;

  active_cell_iterator
  end_active(const unsigned int level) const;

  level_cell_iterator
  begin_mg(const unsigned int level = 0) const;

  level_cell_iterator
  end_mg(const unsigned int level) const;

  level_cell_iterator
  end_mg() const;

  IteratorRange<cell_iterator>
  cell_iterators() const;

  IteratorRange<active_cell_iterator>
  active_cell_iterators() const;

  IteratorRange<level_cell_iterator>
  mg_cell_iterators() const;

  IteratorRange<cell_iterator>
  cell_iterators_on_level(const unsigned int level) const;

  IteratorRange<active_cell_iterator>
  active_cell_iterators_on_level(const unsigned int level) const;

  IteratorRange<level_cell_iterator>
  mg_cell_iterators_on_level(const unsigned int level) const;

  types::global_dof_index
  n_dofs() const;

  types::global_dof_index
  n_dofs(const unsigned int level) const;

  types::global_dof_index
  n_boundary_dofs() const;

  template <typename number>
  types::global_dof_index
  n_boundary_dofs(
    const std::map<types::boundary_id, const Function<spacedim, number> *>
      &boundary_ids) const;

  types::global_dof_index
  n_boundary_dofs(const std::set<types::boundary_id> &boundary_ids) const;

  virtual const BlockInfo &
  block_info() const;

  types::global_dof_index
  n_locally_owned_dofs() const;

  const IndexSet &
  locally_owned_dofs() const;

  const IndexSet &
  locally_owned_mg_dofs(const unsigned int level) const;

  std::vector<IndexSet>
  compute_locally_owned_dofs_per_processor() const;

  std::vector<types::global_dof_index>
  compute_n_locally_owned_dofs_per_processor() const;

  std::vector<IndexSet>
  compute_locally_owned_mg_dofs_per_processor(const unsigned int level) const;

  DEAL_II_DEPRECATED virtual const std::vector<IndexSet> &
  locally_owned_dofs_per_processor() const;

  DEAL_II_DEPRECATED virtual const std::vector<types::global_dof_index> &
  n_locally_owned_dofs_per_processor() const;

  DEAL_II_DEPRECATED virtual const std::vector<IndexSet> &
  locally_owned_mg_dofs_per_processor(const unsigned int level) const;

  const FiniteElement<dim, spacedim> &
  get_fe(const unsigned int index = 0) const;

  const hp::FECollection<dim, spacedim> &
  get_fe_collection() const;

  const Triangulation<dim, spacedim> &
  get_triangulation() const;

  virtual std::size_t
  memory_consumption() const;

  void
  prepare_for_serialization_of_active_fe_indices();

  void
  deserialize_active_fe_indices();

  struct ActiveFEIndexTransfer
  {
    std::map<const cell_iterator, const unsigned int> persisting_cells_fe_index;

    std::map<const cell_iterator, const unsigned int> refined_cells_fe_index;

    std::map<const cell_iterator, const unsigned int> coarsened_cells_fe_index;

    std::vector<unsigned int> active_fe_indices;

    std::unique_ptr<
      parallel::distributed::
        CellDataTransfer<dim, spacedim, std::vector<unsigned int>>>
      cell_data_transfer;
  };

  std::unique_ptr<ActiveFEIndexTransfer> active_fe_index_transfer;

  std::vector<boost::signals2::connection> tria_listeners;

  template <class Archive>
  void
  save(Archive &ar, const unsigned int version) const;

  template <class Archive>
  void
  load(Archive &ar, const unsigned int version);

  BOOST_SERIALIZATION_SPLIT_MEMBER()

  DeclException0(ExcNoFESelected);
  DeclException0(ExcInvalidBoundaryIndicator);
  DeclException1(ExcInvalidLevel,
                 int,
                 << "The given level " << arg1
                 << " is not in the valid range!");
  DeclException1(ExcNewNumbersNotConsecutive,
                 types::global_dof_index,
                 << "The given list of new dof indices is not consecutive: "
                 << "the index " << arg1 << " does not exist.");
  DeclException2(ExcInvalidFEIndex,
                 int,
                 int,
                 << "The mesh contains a cell with an active_fe_index of "
                 << arg1 << ", but the finite element collection only has "
                 << arg2 << " elements");

protected:
  BlockInfo block_info_object;

  SmartPointer<const Triangulation<dim, spacedim>, DoFHandler<dim, spacedim>>
    tria;

  hp::FECollection<dim, spacedim> fe_collection;

  std::unique_ptr<dealii::internal::DoFHandlerImplementation::Policy::
                    PolicyBase<dim, spacedim>>
    policy;

  dealii::internal::DoFHandlerImplementation::NumberCache number_cache;

  std::vector<dealii::internal::DoFHandlerImplementation::NumberCache>
    mg_number_cache;


  void
  clear_space();


  class MGVertexDoFs
  {
  public:
    MGVertexDoFs();

    void
    init(const unsigned int coarsest_level,
         const unsigned int finest_level,
         const unsigned int dofs_per_vertex);

    unsigned int
    get_coarsest_level() const;

    unsigned int
    get_finest_level() const;

    types::global_dof_index
    get_index(const unsigned int level,
              const unsigned int dof_number,
              const unsigned int dofs_per_vertex) const;

    void
    set_index(const unsigned int            level,
              const unsigned int            dof_number,
              const unsigned int            dofs_per_vertex,
              const types::global_dof_index index);

  private:
    unsigned int coarsest_level;

    unsigned int finest_level;

    std::unique_ptr<types::global_dof_index[]> indices;
  };

  void
  setup_policy();

  void
  setup_policy_and_listeners();

  void
  pre_refinement_action();

  void
  post_refinement_action();

  void
  pre_active_fe_index_transfer();

  void
  pre_distributed_active_fe_index_transfer();

  void
  post_active_fe_index_transfer();

  void
  post_distributed_active_fe_index_transfer();

  void
  post_distributed_serialization_of_active_fe_indices();

  void
  create_active_fe_table();

  void
  clear_mg_space();

  template <int structdim>
  types::global_dof_index
  get_dof_index(const unsigned int obj_level,
                const unsigned int obj_index,
                const unsigned int fe_index,
                const unsigned int local_index) const;

  template <int structdim>
  void
  set_dof_index(const unsigned int            obj_level,
                const unsigned int            obj_index,
                const unsigned int            fe_index,
                const unsigned int            local_index,
                const types::global_dof_index global_index) const;

  std::vector<types::global_dof_index> vertex_dofs;

  std::vector<unsigned int> vertex_dof_offsets; // for hp

  std::vector<MGVertexDoFs> mg_vertex_dofs;

  std::vector<
    std::unique_ptr<dealii::internal::DoFHandlerImplementation::DoFLevel<dim>>>
    levels;

  std::vector<
    std::unique_ptr<dealii::internal::DoFHandlerImplementation::DoFLevel<dim>>>
    mg_levels;

  std::vector<std::unique_ptr<dealii::internal::hp::DoFLevel>>
    levels_hp; // TODO: rename hp_levels

  std::unique_ptr<dealii::internal::DoFHandlerImplementation::DoFFaces<dim>>
    faces;

  std::unique_ptr<dealii::internal::DoFHandlerImplementation::DoFFaces<dim>>
    mg_faces;

  std::unique_ptr<dealii::internal::hp::DoFIndicesOnFaces<dim>>
    faces_hp; // TODO: rename hp_faces


  template <int, class, bool>
  friend class dealii::DoFAccessor;
  template <class, bool>
  friend class dealii::DoFCellAccessor;
  friend struct dealii::internal::DoFAccessorImplementation::Implementation;
  friend struct dealii::internal::DoFCellAccessorImplementation::Implementation;

  // Likewise for DoFLevel objects since they need to access the vertex dofs
  // in the functions that set and retrieve vertex dof indices.
  template <int>
  friend class dealii::internal::hp::DoFIndicesOnFacesOrEdges;
  friend struct dealii::internal::hp::DoFHandlerImplementation::Implementation;
  friend struct dealii::internal::DoFHandlerImplementation::Policy::
    Implementation;

  template <int, class, bool>
  friend class DoFAccessor;
  template <class, bool>
  friend class DoFCellAccessor;
  friend struct dealii::internal::DoFAccessorImplementation::Implementation;
  friend struct dealii::internal::DoFCellAccessorImplementation::Implementation;

  friend struct dealii::internal::DoFHandlerImplementation::Implementation;
  friend struct dealii::internal::DoFHandlerImplementation::Policy::
    Implementation;
};



#ifndef DOXYGEN

/* ----------------------- Inline functions ----------------------------------
 */


template <int dim, int spacedim, typename T>
inline bool
DoFHandlerBase<dim, spacedim, T>::has_level_dofs() const
{
  return this->mg_number_cache.size() > 0;
}



template <int dim, int spacedim, typename T>
inline bool
DoFHandlerBase<dim, spacedim, T>::has_active_dofs() const
{
  return this->number_cache.n_global_dofs > 0;
}



template <int dim, int spacedim, typename T>
inline types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::n_dofs() const
{
  return number_cache.n_global_dofs;
}



template <int dim, int spacedim, typename T>
inline types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::n_dofs(const unsigned int level) const
{
  Assert(has_level_dofs(),
         ExcMessage(
           "n_dofs(level) can only be called after distribute_mg_dofs()"));
  Assert(level < mg_number_cache.size(), ExcInvalidLevel(level));
  return mg_number_cache[level].n_global_dofs;
}



template <int dim, int spacedim, typename T>
types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::n_locally_owned_dofs() const
{
  return this->number_cache.n_locally_owned_dofs;
}



template <int dim, int spacedim, typename T>
const IndexSet &
DoFHandlerBase<dim, spacedim, T>::locally_owned_dofs() const
{
  return this->number_cache.locally_owned_dofs;
}



template <int dim, int spacedim, typename T>
const IndexSet &
DoFHandlerBase<dim, spacedim, T>::locally_owned_mg_dofs(
  const unsigned int level) const
{
  Assert(level < this->get_triangulation().n_global_levels(),
         ExcMessage("The given level index exceeds the number of levels "
                    "present in the triangulation"));
  Assert(
    this->mg_number_cache.size() == this->get_triangulation().n_global_levels(),
    ExcMessage(
      "The level dofs are not set up properly! Did you call distribute_mg_dofs()?"));
  return this->mg_number_cache[level].locally_owned_dofs;
}



template <int dim, int spacedim, typename T>
const std::vector<types::global_dof_index> &
DoFHandlerBase<dim, spacedim, T>::n_locally_owned_dofs_per_processor() const
{
  if (this->number_cache.n_locally_owned_dofs_per_processor.empty() &&
      this->number_cache.n_global_dofs > 0)
    {
      const_cast<dealii::internal::DoFHandlerImplementation::NumberCache &>(
        this->number_cache)
        .n_locally_owned_dofs_per_processor =
        compute_n_locally_owned_dofs_per_processor();
    }
  return this->number_cache.n_locally_owned_dofs_per_processor;
}



template <int dim, int spacedim, typename T>
const std::vector<IndexSet> &
DoFHandlerBase<dim, spacedim, T>::locally_owned_dofs_per_processor() const
{
  if (this->number_cache.locally_owned_dofs_per_processor.empty() &&
      this->number_cache.n_global_dofs > 0)
    {
      const_cast<dealii::internal::DoFHandlerImplementation::NumberCache &>(
        this->number_cache)
        .locally_owned_dofs_per_processor =
        compute_locally_owned_dofs_per_processor();
    }
  return this->number_cache.locally_owned_dofs_per_processor;
}



template <int dim, int spacedim, typename T>
const std::vector<IndexSet> &
DoFHandlerBase<dim, spacedim, T>::locally_owned_mg_dofs_per_processor(
  const unsigned int level) const
{
  Assert(level < this->get_triangulation().n_global_levels(),
         ExcMessage("The given level index exceeds the number of levels "
                    "present in the triangulation"));
  Assert(
    this->mg_number_cache.size() == this->get_triangulation().n_global_levels(),
    ExcMessage(
      "The level dofs are not set up properly! Did you call distribute_mg_dofs()?"));
  if (this->mg_number_cache[level].locally_owned_dofs_per_processor.empty() &&
      this->mg_number_cache[level].n_global_dofs > 0)
    {
      const_cast<dealii::internal::DoFHandlerImplementation::NumberCache &>(
        this->mg_number_cache[level])
        .locally_owned_dofs_per_processor =
        compute_locally_owned_mg_dofs_per_processor(level);
    }
  return this->mg_number_cache[level].locally_owned_dofs_per_processor;
}



template <int dim, int spacedim, typename T>
std::vector<types::global_dof_index>
DoFHandlerBase<dim, spacedim, T>::compute_n_locally_owned_dofs_per_processor()
  const
{
  const parallel::TriangulationBase<dim, spacedim> *tr =
    (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
      &this->get_triangulation()));
  if (tr != nullptr)
    return this->number_cache.get_n_locally_owned_dofs_per_processor(
      tr->get_communicator());
  else
    return this->number_cache.get_n_locally_owned_dofs_per_processor(
      MPI_COMM_SELF);
}



template <int dim, int spacedim, typename T>
std::vector<IndexSet>
DoFHandlerBase<dim, spacedim, T>::compute_locally_owned_dofs_per_processor()
  const
{
  const parallel::TriangulationBase<dim, spacedim> *tr =
    (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
      &this->get_triangulation()));
  if (tr != nullptr)
    return this->number_cache.get_locally_owned_dofs_per_processor(
      tr->get_communicator());
  else
    return this->number_cache.get_locally_owned_dofs_per_processor(
      MPI_COMM_SELF);
}



template <int dim, int spacedim, typename T>
std::vector<IndexSet>
DoFHandlerBase<dim, spacedim, T>::compute_locally_owned_mg_dofs_per_processor(
  const unsigned int level) const
{
  Assert(level < this->get_triangulation().n_global_levels(),
         ExcMessage("The given level index exceeds the number of levels "
                    "present in the triangulation"));
  Assert(
    this->mg_number_cache.size() == this->get_triangulation().n_global_levels(),
    ExcMessage(
      "The level dofs are not set up properly! Did you call distribute_mg_dofs()?"));
  const parallel::TriangulationBase<dim, spacedim> *tr =
    (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
      &this->get_triangulation()));
  if (tr != nullptr)
    return this->mg_number_cache[level].get_locally_owned_dofs_per_processor(
      tr->get_communicator());
  else
    return this->mg_number_cache[level].get_locally_owned_dofs_per_processor(
      MPI_COMM_SELF);
}



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
inline const Triangulation<dim, spacedim> &
DoFHandlerBase<dim, spacedim, T>::get_triangulation() const
{
  Assert(tria != nullptr,
         ExcMessage("This DoFHandler object has not been associated "
                    "with a triangulation."));
  return *tria;
}



template <int dim, int spacedim, typename T>
inline const BlockInfo &
DoFHandlerBase<dim, spacedim, T>::block_info() const
{
  Assert(T::is_hp_dof_handler == false, ExcNotImplemented());

  return block_info_object;
}



template <int dim, int spacedim, typename T>
template <typename number>
types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::n_boundary_dofs(
  const std::map<types::boundary_id, const Function<spacedim, number> *>
    &boundary_ids) const
{
  Assert(!(dim == 2 && spacedim == 3) || T::is_hp_dof_handler == false,
         ExcNotImplemented());

  // extract the set of boundary ids and forget about the function object
  // pointers
  std::set<types::boundary_id> boundary_ids_only;
  for (typename std::map<types::boundary_id,
                         const Function<spacedim, number> *>::const_iterator p =
         boundary_ids.begin();
       p != boundary_ids.end();
       ++p)
    boundary_ids_only.insert(p->first);

  // then just hand everything over to the other function that does the work
  return n_boundary_dofs(boundary_ids_only);
}



namespace internal
{
  /**
   * Return a string representing the dynamic type of the given argument.
   * This is basically the same what typeid(...).name() does, but it turns out
   * this is broken on Intel 13+.
   *
   * Defined in dof_handler.cc.
   */
  template <int dim, int spacedim>
  std::string
  policy_to_string(const dealii::internal::DoFHandlerImplementation::Policy::
                     PolicyBase<dim, spacedim> &policy);
} // namespace internal



template <int dim, int spacedim, typename T>
template <class Archive>
void
DoFHandlerBase<dim, spacedim, T>::save(Archive &ar, const unsigned int) const
{
  if (is_hp_dof_handler)
    {
      ar & this->vertex_dofs;
      ar & this->vertex_dof_offsets;
      ar & this->number_cache;
      ar & this->mg_number_cache;

      // some versions of gcc have trouble with loading vectors of
      // std::unique_ptr objects because std::unique_ptr does not
      // have a copy constructor. do it one level at a time
      const unsigned int n_levels = this->levels_hp.size();
      ar &               n_levels;
      for (unsigned int i = 0; i < n_levels; ++i)
        ar & this->levels_hp[i];

      // boost dereferences a nullptr when serializing a nullptr
      // at least up to 1.65.1. This causes problems with clang-5.
      // Therefore, work around it.
      bool faces_is_nullptr = (this->faces_hp.get() == nullptr);
      ar & faces_is_nullptr;
      if (!faces_is_nullptr)
        ar & this->faces_hp;

      // write out the number of triangulation cells and later check during
      // loading that this number is indeed correct; same with something that
      // identifies the policy
      const unsigned int n_cells = this->tria->n_cells();
      std::string        policy_name =
        dealii::internal::policy_to_string(*this->policy);

      ar &n_cells &policy_name;
    }
  else
    {
      ar & this->block_info_object;
      ar & this->vertex_dofs;
      ar & this->number_cache;

      // some versions of gcc have trouble with loading vectors of
      // std::unique_ptr objects because std::unique_ptr does not
      // have a copy constructor. do it one level at a time
      unsigned int n_levels = this->levels.size();
      ar &         n_levels;
      for (unsigned int i = 0; i < this->levels.size(); ++i)
        ar & this->levels[i];

      // boost dereferences a nullptr when serializing a nullptr
      // at least up to 1.65.1. This causes problems with clang-5.
      // Therefore, work around it.
      bool faces_is_nullptr = (this->faces.get() == nullptr);
      ar & faces_is_nullptr;
      if (!faces_is_nullptr)
        ar & this->faces;

      // write out the number of triangulation cells and later check during
      // loading that this number is indeed correct; same with something that
      // identifies the FE and the policy
      unsigned int n_cells     = this->tria->n_cells();
      std::string  fe_name     = this->get_fe(0).get_name();
      std::string  policy_name = internal::policy_to_string(*this->policy);

      ar &n_cells &fe_name &policy_name;
    }
}



template <int dim, int spacedim, typename T>
template <class Archive>
void
DoFHandlerBase<dim, spacedim, T>::load(Archive &ar, const unsigned int)
{
  if (is_hp_dof_handler)
    {
      ar & this->vertex_dofs;
      ar & this->vertex_dof_offsets;
      ar & this->number_cache;
      ar & this->mg_number_cache;

      // boost::serialization can restore pointers just fine, but if the
      // pointer object still points to something useful, that object is not
      // destroyed and we end up with a memory leak. consequently, first delete
      // previous content before re-loading stuff
      this->levels_hp.clear();
      this->faces_hp.reset();

      // some versions of gcc have trouble with loading vectors of
      // std::unique_ptr objects because std::unique_ptr does not
      // have a copy constructor. do it one level at a time
      unsigned int size;
      ar &         size;
      this->levels_hp.resize(size);
      for (unsigned int i = 0; i < size; ++i)
        {
          std::unique_ptr<dealii::internal::hp::DoFLevel> level;
          ar &                                            level;
          this->levels_hp[i] = std::move(level);
        }

      // Workaround for nullptr, see in save().
      bool faces_is_nullptr = true;
      ar & faces_is_nullptr;
      if (!faces_is_nullptr)
        ar & this->faces_hp;

      // these are the checks that correspond to the last block in the save()
      // function
      unsigned int n_cells;
      std::string  policy_name;

      ar &n_cells &policy_name;

      AssertThrow(
        n_cells == this->tria->n_cells(),
        ExcMessage(
          "The object being loaded into does not match the triangulation "
          "that has been stored previously."));
      AssertThrow(
        policy_name == dealii::internal::policy_to_string(*this->policy),
        ExcMessage("The policy currently associated with this DoFHandler (" +
                   dealii::internal::policy_to_string(*this->policy) +
                   ") does not match the one that was associated with the "
                   "DoFHandler previously stored (" +
                   policy_name + ")."));
    }
  else
    {
      ar & this->block_info_object;
      ar & this->vertex_dofs;
      ar & this->number_cache;

      // boost::serialization can restore pointers just fine, but if the
      // pointer object still points to something useful, that object is not
      // destroyed and we end up with a memory leak. consequently, first delete
      // previous content before re-loading stuff
      this->levels.clear();
      this->faces.reset();

      // some versions of gcc have trouble with loading vectors of
      // std::unique_ptr objects because std::unique_ptr does not
      // have a copy constructor. do it one level at a time
      unsigned int size;
      ar &         size;
      this->levels.resize(size);
      for (unsigned int i = 0; i < this->levels.size(); ++i)
        {
          std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<dim>>
              level;
          ar &level;
          this->levels[i] = std::move(level);
        }

      // Workaround for nullptr, see in save().
      bool faces_is_nullptr = true;
      ar & faces_is_nullptr;
      if (!faces_is_nullptr)
        ar & this->faces;

      // these are the checks that correspond to the last block in the save()
      // function
      unsigned int n_cells;
      std::string  fe_name;
      std::string  policy_name;

      ar &n_cells &fe_name &policy_name;

      AssertThrow(
        n_cells == this->tria->n_cells(),
        ExcMessage(
          "The object being loaded into does not match the triangulation "
          "that has been stored previously."));
      AssertThrow(
        fe_name == this->get_fe(0).get_name(),
        ExcMessage(
          "The finite element associated with this DoFHandler does not match "
          "the one that was associated with the DoFHandler previously stored."));
      AssertThrow(policy_name == internal::policy_to_string(*this->policy),
                  ExcMessage(
                    "The policy currently associated with this DoFHandler (" +
                    internal::policy_to_string(*this->policy) +
                    ") does not match the one that was associated with the "
                    "DoFHandler previously stored (" +
                    policy_name + ")."));
    }
}



template <int dim, int spacedim, typename T>
inline types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::MGVertexDoFs::get_index(
  const unsigned int level,
  const unsigned int dof_number,
  const unsigned int dofs_per_vertex) const
{
  Assert((level >= coarsest_level) && (level <= finest_level),
         ExcInvalidLevel(level));
  return indices[dofs_per_vertex * (level - coarsest_level) + dof_number];
}



template <int dim, int spacedim, typename T>
inline void
DoFHandlerBase<dim, spacedim, T>::MGVertexDoFs::set_index(
  const unsigned int            level,
  const unsigned int            dof_number,
  const unsigned int            dofs_per_vertex,
  const types::global_dof_index index)
{
  Assert((level >= coarsest_level) && (level <= finest_level),
         ExcInvalidLevel(level));
  indices[dofs_per_vertex * (level - coarsest_level) + dof_number] = index;
}

#endif // DOXYGEN

DEAL_II_NAMESPACE_CLOSE

#endif
