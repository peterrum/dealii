// ---------------------------------------------------------------------
//
// Copyright (C) 2003 - 2019 by the deal.II authors
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
#include <deal.II/base/thread_management.h>

#include <deal.II/distributed/cell_data_transfer.templates.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler_base.templates.h>
#include <deal.II/dofs/dof_handler_policy.h>

#include <deal.II/fe/fe.h>

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_levels.h>

#include <deal.II/hp/dof_faces.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/dof_level.h>

#include <boost/serialization/array.hpp>

#include <algorithm>
#include <functional>
#include <set>
#include <unordered_set>

DEAL_II_NAMESPACE_OPEN

// The following is necessary for compilation under Visual Studio which is
// unable to correctly distinguish between dealii::DoFHandler and
// dealii::hp::DoFHandler. Plus it makes code in dof_handler.cc easier to read.
#if defined(_MSC_VER) && (_MSC_VER >= 1800)
template <int dim, int spacedim>
using HpDoFHandler = ::dealii::hp::DoFHandler<dim, spacedim>;
#else
// When using older Visual Studio or a different compiler just fall back.
#  define HpDoFHandler DoFHandler
#endif

namespace parallel
{
  namespace distributed
  {
    template <int, int>
    class Triangulation;
  }
} // namespace parallel



namespace hp
{
  template <int dim, int spacedim>
  const unsigned int DoFHandler<dim, spacedim>::dimension;

  template <int dim, int spacedim>
  const unsigned int DoFHandler<dim, spacedim>::space_dimension;

  template <int dim, int spacedim>
  const unsigned int DoFHandler<dim, spacedim>::default_fe_index;



  template <int dim, int spacedim>
  DoFHandler<dim, spacedim>::DoFHandler()
    : Base()
  {}



  template <int dim, int spacedim>
  DoFHandler<dim, spacedim>::DoFHandler(
    const Triangulation<dim, spacedim> &tria)
    : Base(tria)
  {}


  //------------------------------------------------------------------



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::distribute_mg_dofs_impl()
  {
    AssertThrow(false, ExcNotImplemented());
  }

  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::initialize_local_block_info()
  {
    AssertThrow(false, ExcNotImplemented());
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::distribute_dofs_impl(
    const hp::FECollection<dim, spacedim> &ff)
  {
    // assign the fe_collection and initialize all active_fe_indices
    this->set_fe(ff);

    // If an underlying shared::Tria allows artificial cells,
    // then save the current set of subdomain ids, and set
    // subdomain ids to the "true" owner of each cell. we later
    // restore these flags
    std::vector<types::subdomain_id>                      saved_subdomain_ids;
    const parallel::shared::Triangulation<dim, spacedim> *shared_tria =
      (dynamic_cast<const parallel::shared::Triangulation<dim, spacedim> *>(
        &this->get_triangulation()));
    if (shared_tria != nullptr && shared_tria->with_artificial_cells())
      {
        saved_subdomain_ids.resize(shared_tria->n_active_cells());

        const std::vector<types::subdomain_id> &true_subdomain_ids =
          shared_tria->get_true_subdomain_ids_of_cells();

        for (const auto &cell : shared_tria->active_cell_iterators())
          {
            const unsigned int index   = cell->active_cell_index();
            saved_subdomain_ids[index] = cell->subdomain_id();
            cell->set_subdomain_id(true_subdomain_ids[index]);
          }
      }

    // then allocate space for all the other tables
    dealii::internal::hp::DoFHandlerImplementation::Implementation::
      reserve_space(*this);

    // now undo the subdomain modification
    if (shared_tria != nullptr && shared_tria->with_artificial_cells())
      for (const auto &cell : shared_tria->active_cell_iterators())
        cell->set_subdomain_id(saved_subdomain_ids[cell->active_cell_index()]);


    // Clear user flags because we will need them. But first we save
    // them and make sure that we restore them later such that at the
    // end of this function the Triangulation will be in the same
    // state as it was at the beginning of this function.
    std::vector<bool> user_flags;
    this->tria->save_user_flags(user_flags);
    const_cast<Triangulation<dim, spacedim> &>(*this->tria).clear_user_flags();


    /////////////////////////////////

    // Now for the real work:
    this->number_cache = this->policy->distribute_dofs();

    /////////////////////////////////

    // do some housekeeping: compress indices
    {
      Threads::TaskGroup<> tg;
      for (int level = this->levels_hp.size() - 1; level >= 0; --level)
        tg += Threads::new_task(
          &dealii::internal::hp::DoFLevel::compress_data<dim, spacedim>,
          *this->levels_hp[level],
          this->fe_collection);
      tg.join_all();
    }

    // finally restore the user flags
    const_cast<Triangulation<dim, spacedim> &>(*this->tria)
      .load_user_flags(user_flags);
  }
} // namespace hp



/*-------------- Explicit Instantiations -------------------------------*/
#include "dof_handler.inst"


DEAL_II_NAMESPACE_CLOSE
