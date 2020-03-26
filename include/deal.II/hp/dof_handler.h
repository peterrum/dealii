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
    static const unsigned int dimension = dim;

    static const unsigned int space_dimension = spacedim;

    static const bool is_hp_dof_handler = true;

    DEAL_II_DEPRECATED
    static const types::global_dof_index invalid_dof_index =
      numbers::invalid_dof_index;

    static const unsigned int default_fe_index = numbers::invalid_unsigned_int;


    DoFHandler();

    DoFHandler(const Triangulation<dim, spacedim> &tria);

    DoFHandler(const DoFHandler &) = delete;

    virtual ~DoFHandler() override;

    DoFHandler &
    operator=(const DoFHandler &) = delete;

    void
    initialize(const Triangulation<dim, spacedim> &tria,
               const FiniteElement<dim, spacedim> &fe) override;

    void
    initialize(const Triangulation<dim, spacedim> &   tria,
               const hp::FECollection<dim, spacedim> &fe) override;

    virtual void
    set_fe_impl(const hp::FECollection<dim, spacedim> &fe);

    virtual void
    distribute_dofs(const FiniteElement<dim, spacedim> &fe) override;

    virtual void
    distribute_dofs(const hp::FECollection<dim, spacedim> &fe) override;

    void
    set_active_fe_indices(
      const std::vector<unsigned int> &active_fe_indices) override;

    void
    get_active_fe_indices(
      std::vector<unsigned int> &active_fe_indices) const override;

    DEAL_II_DEPRECATED
    virtual void
    distribute_mg_dofs(const FiniteElement<dim, spacedim> &fe) override;

    virtual void
    distribute_mg_dofs() override;

    bool
    has_level_dofs() const override;

    bool
    has_active_dofs() const override;

    void
    initialize_local_block_info() override;

    virtual void
    clear() override;

    void
    renumber_dofs(
      const std::vector<types::global_dof_index> &new_numbers) override;

    void
    renumber_dofs(
      const unsigned int                          level,
      const std::vector<types::global_dof_index> &new_numbers) override;

    unsigned int
    max_couplings_between_dofs() const override;

    unsigned int
    max_couplings_between_boundary_dofs() const override;

    types::global_dof_index
    n_dofs() const override;

    types::global_dof_index
    n_dofs(const unsigned int level) const override;

    types::global_dof_index
    n_locally_owned_dofs() const override;

    const IndexSet &
    locally_owned_dofs() const override;

    std::vector<IndexSet>
    compute_locally_owned_dofs_per_processor() const override;

    std::vector<types::global_dof_index>
    compute_n_locally_owned_dofs_per_processor() const override;

    DEAL_II_DEPRECATED const std::vector<IndexSet> &
                             locally_owned_dofs_per_processor() const override;

    DEAL_II_DEPRECATED const std::vector<types::global_dof_index> &
                             n_locally_owned_dofs_per_processor() const override;

    const IndexSet &
    locally_owned_mg_dofs(const unsigned int level) const override;

    std::vector<IndexSet>
    compute_locally_owned_mg_dofs_per_processor(
      const unsigned int level) const override;

    DEAL_II_DEPRECATED const std::vector<IndexSet> &
                             locally_owned_mg_dofs_per_processor(
                               const unsigned int level) const override;

    virtual std::size_t
    memory_consumption() const override;

    void
    prepare_for_serialization_of_active_fe_indices() override;

    void
    deserialize_active_fe_indices() override;

    template <class Archive>
    void
    save(Archive &ar, const unsigned int version) const;

    template <class Archive>
    void
    load(Archive &ar, const unsigned int version);

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    DeclException0(ExcNoFESelected);
    DeclException0(ExcGridsDoNotMatch);
    DeclException0(ExcInvalidBoundaryIndicator);
    DeclException1(ExcMatrixHasWrongSize,
                   int,
                   << "The matrix has the wrong dimension " << arg1);
    DeclException0(ExcFunctionNotUseful);
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
    DeclException1(ExcInvalidLevel,
                   int,
                   << "The given level " << arg1
                   << " is not in the valid range!");
    DeclException0(ExcFacesHaveNoLevel);
    DeclException1(ExcEmptyLevel,
                   int,
                   << "You tried to do something on level " << arg1
                   << ", but this level is empty.");

  private:
    void
    setup_policy_and_listeners();

    void
    clear_space();

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

    void
    create_active_fe_table();

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

    std::vector<std::unique_ptr<dealii::internal::hp::DoFLevel>> levels;

    std::unique_ptr<dealii::internal::hp::DoFIndicesOnFaces<dim>> faces;

    dealii::internal::DoFHandlerImplementation::NumberCache number_cache;

    std::vector<dealii::internal::DoFHandlerImplementation::NumberCache>
      mg_number_cache;

    std::vector<types::global_dof_index> vertex_dofs;

    std::vector<unsigned int> vertex_dof_offsets;

    struct ActiveFEIndexTransfer
    {
      std::map<const typename Base::cell_iterator, const unsigned int>
        persisting_cells_fe_index;

      std::map<const typename Base::cell_iterator, const unsigned int>
        refined_cells_fe_index;

      std::map<const typename Base::cell_iterator, const unsigned int>
        coarsened_cells_fe_index;

      std::vector<unsigned int> active_fe_indices;

      std::unique_ptr<
        parallel::distributed::
          CellDataTransfer<dim, spacedim, std::vector<unsigned int>>>
        cell_data_transfer;
    };

    std::unique_ptr<ActiveFEIndexTransfer> active_fe_index_transfer;

    std::vector<boost::signals2::connection> tria_listeners;

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


#ifndef DOXYGEN

/* ----------------------- Inline functions ----------------------------------
 */

namespace internal
{
  /**
   * Return a string representing the dynamic type of the given argument.
   * This is basically the same what typeid(...).name() does, but it turns out
   * this is broken on Intel 13+.
   *
   * Defined in source/dofs/dof_handler.cc.
   */
  template <int dim, int spacedim>
  std::string
  policy_to_string(const dealii::internal::DoFHandlerImplementation::Policy::
                     PolicyBase<dim, spacedim> &policy);
} // namespace internal


namespace hp
{
  template <int dim, int spacedim>
  template <class Archive>
  void
  DoFHandler<dim, spacedim>::save(Archive &ar, unsigned int) const
  {
    ar &vertex_dofs;
    ar &vertex_dof_offsets;
    ar &number_cache;
    ar &mg_number_cache;

    // some versions of gcc have trouble with loading vectors of
    // std::unique_ptr objects because std::unique_ptr does not
    // have a copy constructor. do it one level at a time
    const unsigned int n_levels = levels.size();
    ar &               n_levels;
    for (unsigned int i = 0; i < n_levels; ++i)
      ar &levels[i];

    // boost dereferences a nullptr when serializing a nullptr
    // at least up to 1.65.1. This causes problems with clang-5.
    // Therefore, work around it.
    bool faces_is_nullptr = (faces.get() == nullptr);
    ar & faces_is_nullptr;
    if (!faces_is_nullptr)
      ar &faces;

    // write out the number of triangulation cells and later check during
    // loading that this number is indeed correct; same with something that
    // identifies the policy
    const unsigned int n_cells = this->tria->n_cells();
    std::string policy_name = dealii::internal::policy_to_string(*this->policy);

    ar &n_cells &policy_name;
  }



  template <int dim, int spacedim>
  template <class Archive>
  void
  DoFHandler<dim, spacedim>::load(Archive &ar, unsigned int)
  {
    ar &vertex_dofs;
    ar &vertex_dof_offsets;
    ar &number_cache;
    ar &mg_number_cache;

    // boost::serialization can restore pointers just fine, but if the
    // pointer object still points to something useful, that object is not
    // destroyed and we end up with a memory leak. consequently, first delete
    // previous content before re-loading stuff
    levels.clear();
    faces.reset();

    // some versions of gcc have trouble with loading vectors of
    // std::unique_ptr objects because std::unique_ptr does not
    // have a copy constructor. do it one level at a time
    unsigned int size;
    ar &         size;
    levels.resize(size);
    for (unsigned int i = 0; i < size; ++i)
      {
        std::unique_ptr<dealii::internal::hp::DoFLevel> level;
        ar &                                            level;
        levels[i] = std::move(level);
      }

    // Workaround for nullptr, see in save().
    bool faces_is_nullptr = true;
    ar & faces_is_nullptr;
    if (!faces_is_nullptr)
      ar &faces;

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

  template <int dim, int spacedim>
  inline types::global_dof_index
  DoFHandler<dim, spacedim>::n_dofs() const
  {
    return number_cache.n_global_dofs;
  }



  template <int dim, int spacedim>
  inline types::global_dof_index
  DoFHandler<dim, spacedim>::n_dofs(const unsigned int) const
  {
    Assert(false, ExcNotImplemented());
    return numbers::invalid_dof_index;
  }



  template <int dim, int spacedim>
  types::global_dof_index
  DoFHandler<dim, spacedim>::n_locally_owned_dofs() const
  {
    return number_cache.n_locally_owned_dofs;
  }



  template <int dim, int spacedim>
  const IndexSet &
  DoFHandler<dim, spacedim>::locally_owned_dofs() const
  {
    return number_cache.locally_owned_dofs;
  }



  template <int dim, int spacedim>
  const std::vector<types::global_dof_index> &
  DoFHandler<dim, spacedim>::n_locally_owned_dofs_per_processor() const
  {
    if (number_cache.n_locally_owned_dofs_per_processor.empty() &&
        number_cache.n_global_dofs > 0)
      {
        const_cast<dealii::internal::DoFHandlerImplementation::NumberCache &>(
          number_cache)
          .n_locally_owned_dofs_per_processor =
          compute_n_locally_owned_dofs_per_processor();
      }
    return number_cache.n_locally_owned_dofs_per_processor;
  }



  template <int dim, int spacedim>
  const std::vector<IndexSet> &
  DoFHandler<dim, spacedim>::locally_owned_dofs_per_processor() const
  {
    if (number_cache.locally_owned_dofs_per_processor.empty() &&
        number_cache.n_global_dofs > 0)
      {
        const_cast<dealii::internal::DoFHandlerImplementation::NumberCache &>(
          number_cache)
          .locally_owned_dofs_per_processor =
          compute_locally_owned_dofs_per_processor();
      }
    return number_cache.locally_owned_dofs_per_processor;
  }



  template <int dim, int spacedim>
  std::vector<types::global_dof_index>
  DoFHandler<dim, spacedim>::compute_n_locally_owned_dofs_per_processor() const
  {
    const parallel::TriangulationBase<dim, spacedim> *tr =
      (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
        &this->get_triangulation()));
    if (tr != nullptr)
      return number_cache.get_n_locally_owned_dofs_per_processor(
        tr->get_communicator());
    else
      return number_cache.get_n_locally_owned_dofs_per_processor(MPI_COMM_SELF);
  }



  template <int dim, int spacedim>
  std::vector<IndexSet>
  DoFHandler<dim, spacedim>::compute_locally_owned_dofs_per_processor() const
  {
    const parallel::TriangulationBase<dim, spacedim> *tr =
      (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
        &this->get_triangulation()));
    if (tr != nullptr)
      return number_cache.get_locally_owned_dofs_per_processor(
        tr->get_communicator());
    else
      return number_cache.get_locally_owned_dofs_per_processor(MPI_COMM_SELF);
  }



  template <int dim, int spacedim>
  const IndexSet &
  DoFHandler<dim, spacedim>::locally_owned_mg_dofs(
    const unsigned int level) const
  {
    Assert(false, ExcNotImplemented());
    (void)level;
    Assert(level < this->get_triangulation().n_global_levels(),
           ExcMessage("The given level index exceeds the number of levels "
                      "present in the triangulation"));
    return mg_number_cache[0].locally_owned_dofs;
  }



  template <int dim, int spacedim>
  const std::vector<IndexSet> &
  DoFHandler<dim, spacedim>::locally_owned_mg_dofs_per_processor(
    const unsigned int level) const
  {
    Assert(level < this->get_triangulation().n_global_levels(),
           ExcMessage("The given level index exceeds the number of levels "
                      "present in the triangulation"));
    Assert(
      mg_number_cache.size() == this->get_triangulation().n_global_levels(),
      ExcMessage(
        "The level dofs are not set up properly! Did you call distribute_mg_dofs()?"));
    if (mg_number_cache[level].locally_owned_dofs_per_processor.empty() &&
        mg_number_cache[level].n_global_dofs > 0)
      {
        const_cast<dealii::internal::DoFHandlerImplementation::NumberCache &>(
          mg_number_cache[level])
          .locally_owned_dofs_per_processor =
          compute_locally_owned_mg_dofs_per_processor(level);
      }
    return mg_number_cache[level].locally_owned_dofs_per_processor;
  }



  template <int dim, int spacedim>
  std::vector<IndexSet>
  DoFHandler<dim, spacedim>::compute_locally_owned_mg_dofs_per_processor(
    const unsigned int level) const
  {
    Assert(false, ExcNotImplemented());
    (void)level;
    Assert(level < this->get_triangulation().n_global_levels(),
           ExcMessage("The given level index exceeds the number of levels "
                      "present in the triangulation"));
    const parallel::TriangulationBase<dim, spacedim> *tr =
      (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
        &this->get_triangulation()));
    if (tr != nullptr)
      return mg_number_cache[level].get_locally_owned_dofs_per_processor(
        tr->get_communicator());
    else
      return mg_number_cache[level].get_locally_owned_dofs_per_processor(
        MPI_COMM_SELF);
  }

#endif

} // namespace hp

DEAL_II_NAMESPACE_CLOSE

#endif
