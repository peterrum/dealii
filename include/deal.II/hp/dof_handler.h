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
    initialize_impl(const Triangulation<dim, spacedim> &   tria,
                    const hp::FECollection<dim, spacedim> &fe);

    virtual void
    distribute_dofs_impl(const hp::FECollection<dim, spacedim> &fe);

    virtual void
    distribute_mg_dofs_impl();

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
    clear_space();

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
    std::string policy_name = dealii::internal::policy_to_string(*this->policy);

    ar &n_cells &policy_name;
  }



  template <int dim, int spacedim>
  template <class Archive>
  void
  DoFHandler<dim, spacedim>::load(Archive &ar, unsigned int)
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

#endif

} // namespace hp

DEAL_II_NAMESPACE_CLOSE

#endif
