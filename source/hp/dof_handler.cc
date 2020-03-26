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
  const types::global_dof_index DoFHandler<dim, spacedim>::invalid_dof_index;

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
  {
    setup_policy_and_listeners();

    create_active_fe_table();
  }



  template <int dim, int spacedim>
  DoFHandler<dim, spacedim>::~DoFHandler()
  {
    // unsubscribe as a listener to refinement of the underlying
    // triangulation
    for (auto &connection : this->tria_listeners)
      connection.disconnect();
    this->tria_listeners.clear();

    // ...and release allocated memory
    // virtual functions called in constructors and destructors never use the
    // override in a derived class
    // for clarity be explicit on which function is called
    DoFHandler<dim, spacedim>::clear();
  }


  //------------------------------------------------------------------



  template <int dim, int spacedim>
  std::size_t
  DoFHandler<dim, spacedim>::memory_consumption() const
  {
    std::size_t mem =
      (MemoryConsumption::memory_consumption(this->tria) +
       MemoryConsumption::memory_consumption(this->fe_collection) +
       MemoryConsumption::memory_consumption(this->tria) +
       MemoryConsumption::memory_consumption(this->levels_hp) +
       MemoryConsumption::memory_consumption(*this->faces_hp) +
       MemoryConsumption::memory_consumption(this->number_cache) +
       MemoryConsumption::memory_consumption(this->vertex_dofs) +
       MemoryConsumption::memory_consumption(this->vertex_dof_offsets));
    for (unsigned int i = 0; i < this->levels_hp.size(); ++i)
      mem += MemoryConsumption::memory_consumption(*this->levels_hp[i]);
    mem += MemoryConsumption::memory_consumption(*this->faces_hp);

    return mem;
  }



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
  DoFHandler<dim, spacedim>::initialize_impl(
    const Triangulation<dim, spacedim> &   tria,
    const hp::FECollection<dim, spacedim> &fe)
  {
    clear();

    if (this->tria != &tria)
      {
        for (auto &connection : this->tria_listeners)
          connection.disconnect();
        this->tria_listeners.clear();

        this->tria = &tria;

        this->setup_policy_and_listeners();
      }

    this->create_active_fe_table();

    this->distribute_dofs(fe);
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

    // ensure that the active_fe_indices vectors are initialized correctly
    create_active_fe_table();

    // make sure every processor knows the active_fe_indices
    // on both its own cells and all ghost cells
    dealii::internal::hp::DoFHandlerImplementation::Implementation::
      communicate_active_fe_indices(*this);

    // make sure that the fe collection is large enough to
    // cover all fe indices presently in use on the mesh
    for (const auto &cell : this->active_cell_iterators())
      if (!cell->is_artificial())
        Assert(cell->active_fe_index() < this->fe_collection.size(),
               ExcInvalidFEIndex(cell->active_fe_index(),
                                 this->fe_collection.size()));
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



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::setup_policy_and_listeners()
  {
    // connect functions to signals of the underlying triangulation
    this->tria_listeners.push_back(this->tria->signals.pre_refinement.connect(
      [this]() { this->pre_refinement_action(); }));
    this->tria_listeners.push_back(this->tria->signals.post_refinement.connect(
      [this]() { this->post_refinement_action(); }));
    this->tria_listeners.push_back(this->tria->signals.create.connect(
      [this]() { this->post_refinement_action(); }));

    // decide whether we need a sequential or a parallel shared/distributed
    // policy and attach corresponding callback functions dealing with the
    // transfer of active_fe_indices
    if (dynamic_cast<
          const dealii::parallel::distributed::Triangulation<dim, spacedim> *>(
          &this->get_triangulation()))
      {
        this->policy = std_cxx14::make_unique<
          internal::DoFHandlerImplementation::Policy::ParallelDistributed<
            DoFHandler<dim, spacedim>>>(*this);

        // repartitioning signals
        this->tria_listeners.push_back(
          this->tria->signals.pre_distributed_repartition.connect([this]() {
            internal::hp::DoFHandlerImplementation::Implementation::
              ensure_absence_of_future_fe_indices<dim, spacedim>(*this);
          }));
        this->tria_listeners.push_back(
          this->tria->signals.pre_distributed_repartition.connect(
            [this]() { this->pre_distributed_active_fe_index_transfer(); }));
        this->tria_listeners.push_back(
          this->tria->signals.post_distributed_repartition.connect(
            [this] { this->post_distributed_active_fe_index_transfer(); }));

        // refinement signals
        this->tria_listeners.push_back(
          this->tria->signals.pre_distributed_refinement.connect(
            [this]() { this->pre_distributed_active_fe_index_transfer(); }));
        this->tria_listeners.push_back(
          this->tria->signals.post_distributed_refinement.connect(
            [this]() { this->post_distributed_active_fe_index_transfer(); }));

        // serialization signals
        this->tria_listeners.push_back(
          this->tria->signals.post_distributed_save.connect([this]() {
            this->post_distributed_serialization_of_active_fe_indices();
          }));
      }
    else if (dynamic_cast<
               const dealii::parallel::shared::Triangulation<dim, spacedim> *>(
               &this->get_triangulation()) != nullptr)
      {
        this->policy =
          std_cxx14::make_unique<internal::DoFHandlerImplementation::Policy::
                                   ParallelShared<DoFHandler<dim, spacedim>>>(
            *this);

        // partitioning signals
        this->tria_listeners.push_back(
          this->tria->signals.pre_partition.connect([this]() {
            internal::hp::DoFHandlerImplementation::Implementation::
              ensure_absence_of_future_fe_indices(*this);
          }));

        // refinement signals
        this->tria_listeners.push_back(
          this->tria->signals.pre_refinement.connect(
            [this] { this->pre_active_fe_index_transfer(); }));
        this->tria_listeners.push_back(
          this->tria->signals.post_refinement.connect(
            [this] { this->post_active_fe_index_transfer(); }));
      }
    else
      {
        this->policy =
          std_cxx14::make_unique<internal::DoFHandlerImplementation::Policy::
                                   Sequential<DoFHandler<dim, spacedim>>>(
            *this);

        // refinement signals
        this->tria_listeners.push_back(
          this->tria->signals.pre_refinement.connect(
            [this] { this->pre_active_fe_index_transfer(); }));
        this->tria_listeners.push_back(
          this->tria->signals.post_refinement.connect(
            [this] { this->post_active_fe_index_transfer(); }));
      }
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::clear()
  {
    // release memory
    clear_space();
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::renumber_dofs(
    const std::vector<types::global_dof_index> &new_numbers)
  {
    Assert(this->levels_hp.size() > 0,
           ExcMessage(
             "You need to distribute DoFs before you can renumber them."));

    AssertDimension(new_numbers.size(), this->n_locally_owned_dofs());

#ifdef DEBUG
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

    // uncompress the internal storage scheme of dofs on cells so that
    // we can access dofs in turns. uncompress in parallel, starting
    // with the most expensive levels (the highest ones)
    {
      Threads::TaskGroup<> tg;
      for (int level = this->levels_hp.size() - 1; level >= 0; --level)
        tg += Threads::new_task(
          &dealii::internal::hp::DoFLevel::uncompress_data<dim, spacedim>,
          *this->levels_hp[level],
          this->fe_collection);
      tg.join_all();
    }

    // do the renumbering
    this->number_cache = this->policy->renumber_dofs(new_numbers);

    // now re-compress the dof indices
    {
      Threads::TaskGroup<> tg;
      for (int level = this->levels_hp.size() - 1; level >= 0; --level)
        tg += Threads::new_task(
          &dealii::internal::hp::DoFLevel::compress_data<dim, spacedim>,
          *this->levels_hp[level],
          this->fe_collection);
      tg.join_all();
    }
  }


  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::renumber_dofs(
    const unsigned int                          level,
    const std::vector<types::global_dof_index> &new_numbers)
  {
    (void)level;
    (void)new_numbers;
  }



  template <int dim, int spacedim>
  unsigned int
  DoFHandler<dim, spacedim>::max_couplings_between_dofs() const
  {
    Assert(this->fe_collection.size() > 0, ExcNoFESelected());
    return dealii::internal::hp::DoFHandlerImplementation::Implementation::
      max_couplings_between_dofs(*this);
  }



  template <int dim, int spacedim>
  unsigned int
  DoFHandler<dim, spacedim>::max_couplings_between_boundary_dofs() const
  {
    Assert(this->fe_collection.size() > 0, ExcNoFESelected());

    switch (dim)
      {
        case 1:
          return this->fe_collection.max_dofs_per_vertex();
        case 2:
          return (3 * this->fe_collection.max_dofs_per_vertex() +
                  2 * this->fe_collection.max_dofs_per_line());
        case 3:
          // we need to take refinement of one boundary face into
          // consideration here; in fact, this function returns what
          // #max_coupling_between_dofs<2> returns
          //
          // we assume here, that only four faces meet at the boundary;
          // this assumption is not justified and needs to be fixed some
          // time. fortunately, omitting it for now does no harm since
          // the matrix will cry foul if its requirements are not
          // satisfied
          return (19 * this->fe_collection.max_dofs_per_vertex() +
                  28 * this->fe_collection.max_dofs_per_line() +
                  8 * this->fe_collection.max_dofs_per_quad());
        default:
          Assert(false, ExcNotImplemented());
          return 0;
      }
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::create_active_fe_table()
  {
    // Create sufficiently many hp::DoFLevels.
    while (this->levels_hp.size() < this->tria->n_levels())
      this->levels_hp.emplace_back(new dealii::internal::hp::DoFLevel);

    // then make sure that on each level we have the appropriate size
    // of active_fe_indices; preset them to zero, i.e. the default FE
    for (unsigned int level = 0; level < this->levels_hp.size(); ++level)
      {
        if (this->levels_hp[level]->active_fe_indices.size() == 0 &&
            this->levels_hp[level]->future_fe_indices.size() == 0)
          {
            this->levels_hp[level]->active_fe_indices.resize(
              this->tria->n_raw_cells(level), 0);
            this->levels_hp[level]->future_fe_indices.resize(
              this->tria->n_raw_cells(level),
              dealii::internal::hp::DoFLevel::invalid_active_fe_index);
          }
        else
          {
            // Either the active_fe_indices have size zero because
            // they were just created, or the correct size. Other
            // sizes indicate that something went wrong.
            Assert(this->levels_hp[level]->active_fe_indices.size() ==
                       this->tria->n_raw_cells(level) &&
                     this->levels_hp[level]->future_fe_indices.size() ==
                       this->tria->n_raw_cells(level),
                   ExcInternalError());
          }

        // it may be that the previous table was compressed; in that
        // case, restore the correct active_fe_index. the fact that
        // this no longer matches the indices in the table is of no
        // importance because the current function is called at a
        // point where we have to recreate the dof_indices tables in
        // the levels anyway
        this->levels_hp[level]->normalize_active_fe_indices();
      }
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::pre_refinement_action()
  {
    create_active_fe_table();
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::post_refinement_action()
  {
    // Normally only one level is added, but if this Triangulation
    // is created by copy_triangulation, it can be more than one level.
    while (this->levels_hp.size() < this->tria->n_levels())
      this->levels_hp.emplace_back(new dealii::internal::hp::DoFLevel);

    // Coarsening can lead to the loss of levels. Hence remove them.
    while (this->levels_hp.size() > this->tria->n_levels())
      {
        // drop the last element. that also releases the memory pointed to
        this->levels_hp.pop_back();
      }

    Assert(this->levels_hp.size() == this->tria->n_levels(),
           ExcInternalError());
    for (unsigned int i = 0; i < this->levels_hp.size(); ++i)
      {
        // Resize active_fe_indices vectors. Use zero indicator to extend.
        this->levels_hp[i]->active_fe_indices.resize(this->tria->n_raw_cells(i),
                                                     0);

        // Resize future_fe_indices vectors. Make sure that all
        // future_fe_indices have been cleared after refinement happened.
        //
        // We have used future_fe_indices to update all active_fe_indices
        // before refinement happened, thus we are safe to clear them now.
        this->levels_hp[i]->future_fe_indices.assign(
          this->tria->n_raw_cells(i),
          dealii::internal::hp::DoFLevel::invalid_active_fe_index);
      }
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::pre_active_fe_index_transfer()
  {
    // Finite elements need to be assigned to each cell by calling
    // distribute_dofs() first to make this functionality available.
    if (this->fe_collection.size() > 0)
      {
        Assert(this->active_fe_index_transfer == nullptr, ExcInternalError());

        this->active_fe_index_transfer =
          std_cxx14::make_unique<typename Base::ActiveFEIndexTransfer>();

        dealii::internal::hp::DoFHandlerImplementation::Implementation::
          collect_fe_indices_on_cells_to_be_refined(*this);
      }
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::pre_distributed_active_fe_index_transfer()
  {
#ifndef DEAL_II_WITH_P4EST
    Assert(false, ExcInternalError());
#else
    // Finite elements need to be assigned to each cell by calling
    // distribute_dofs() first to make this functionality available.
    if (this->fe_collection.size() > 0)
      {
        Assert(this->active_fe_index_transfer == nullptr, ExcInternalError());

        this->active_fe_index_transfer =
          std_cxx14::make_unique<typename Base::ActiveFEIndexTransfer>();

        // If we work on a p::d::Triangulation, we have to transfer all
        // active_fe_indices since ownership of cells may change. We will
        // use our p::d::CellDataTransfer member to achieve this. Further,
        // we prepare the values in such a way that they will correspond to
        // the active_fe_indices on the new mesh.

        // Gather all current future_fe_indices.
        this->active_fe_index_transfer->active_fe_indices.resize(
          this->get_triangulation().n_active_cells(),
          numbers::invalid_unsigned_int);

        for (const auto &cell : this->active_cell_iterators())
          if (cell->is_locally_owned())
            this->active_fe_index_transfer
              ->active_fe_indices[cell->active_cell_index()] =
              cell->future_fe_index();

        // Create transfer object and attach to it.
        const auto *distributed_tria = dynamic_cast<
          const parallel::distributed::Triangulation<dim, spacedim> *>(
          &this->get_triangulation());

        this->active_fe_index_transfer->cell_data_transfer =
          std_cxx14::make_unique<
            parallel::distributed::
              CellDataTransfer<dim, spacedim, std::vector<unsigned int>>>(
            *distributed_tria,
            /*transfer_variable_size_data=*/false,
            [this](const std::vector<unsigned int> &children_fe_indices) {
              return dealii::internal::hp::DoFHandlerImplementation::
                Implementation::determine_fe_from_children<dim, spacedim>(
                  children_fe_indices, this->fe_collection);
            });

        this->active_fe_index_transfer->cell_data_transfer
          ->prepare_for_coarsening_and_refinement(
            this->active_fe_index_transfer->active_fe_indices);
      }
#endif
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::post_active_fe_index_transfer()
  {
    // Finite elements need to be assigned to each cell by calling
    // distribute_dofs() first to make this functionality available.
    if (this->fe_collection.size() > 0)
      {
        Assert(this->active_fe_index_transfer != nullptr, ExcInternalError());

        dealii::internal::hp::DoFHandlerImplementation::Implementation::
          distribute_fe_indices_on_refined_cells(*this);

        // We have to distribute the information about active_fe_indices
        // of all cells (including the artificial ones) on all processors,
        // if a parallel::shared::Triangulation has been used.
        dealii::internal::hp::DoFHandlerImplementation::Implementation::
          communicate_active_fe_indices(*this);

        // Free memory.
        this->active_fe_index_transfer.reset();
      }
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::post_distributed_active_fe_index_transfer()
  {
#ifndef DEAL_II_WITH_P4EST
    Assert(false, ExcInternalError());
#else
    // Finite elements need to be assigned to each cell by calling
    // distribute_dofs() first to make this functionality available.
    if (this->fe_collection.size() > 0)
      {
        Assert(this->active_fe_index_transfer != nullptr, ExcInternalError());

        // Unpack active_fe_indices.
        this->active_fe_index_transfer->active_fe_indices.resize(
          this->get_triangulation().n_active_cells(),
          numbers::invalid_unsigned_int);
        this->active_fe_index_transfer->cell_data_transfer->unpack(
          this->active_fe_index_transfer->active_fe_indices);

        // Update all locally owned active_fe_indices.
        this->set_active_fe_indices(
          this->active_fe_index_transfer->active_fe_indices);

        // Update active_fe_indices on ghost cells.
        dealii::internal::hp::DoFHandlerImplementation::Implementation::
          communicate_active_fe_indices(*this);

        // Free memory.
        this->active_fe_index_transfer.reset();
      }
#endif
  }


  template <int dim, int spacedim>
  void
  DoFHandler<dim,
             spacedim>::post_distributed_serialization_of_active_fe_indices()
  {
#ifndef DEAL_II_WITH_P4EST
    Assert(false,
           ExcMessage(
             "You are attempting to use a functionality that is only available "
             "if deal.II was configured to use p4est, but cmake did not find a "
             "valid p4est library."));
#else
    if (this->fe_collection.size() > 0)
      {
        Assert(this->active_fe_index_transfer != nullptr, ExcInternalError());

        // Free memory.
        this->active_fe_index_transfer.reset();
      }
#endif
  }



  template <int dim, int spacedim>
  template <int structdim>
  types::global_dof_index
  DoFHandler<dim, spacedim>::get_dof_index(const unsigned int,
                                           const unsigned int,
                                           const unsigned int,
                                           const unsigned int) const
  {
    Assert(false, ExcNotImplemented());
    return numbers::invalid_dof_index;
  }



  template <int dim, int spacedim>
  template <int structdim>
  void
  DoFHandler<dim, spacedim>::set_dof_index(const unsigned int,
                                           const unsigned int,
                                           const unsigned int,
                                           const unsigned int,
                                           const types::global_dof_index) const
  {
    Assert(false, ExcNotImplemented());
  }



  template <int dim, int spacedim>
  void
  DoFHandler<dim, spacedim>::clear_space()
  {
    this->levels_hp.clear();
    this->faces_hp.reset();

    this->vertex_dofs        = std::vector<types::global_dof_index>();
    this->vertex_dof_offsets = std::vector<unsigned int>();
  }
} // namespace hp



/*-------------- Explicit Instantiations -------------------------------*/
#include "dof_handler.inst"


DEAL_II_NAMESPACE_CLOSE
