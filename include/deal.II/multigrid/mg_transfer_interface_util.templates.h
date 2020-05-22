// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
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


#ifndef dealii_mg_transfer_inteface_util_templates_h
#define dealii_mg_transfer_inteface_util_templates_h

#include <deal.II/base/config.h>

#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/base/mpi_consensus_algorithms.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>

#include <deal.II/hp/dof_handler.h>

#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/multigrid/mg_transfer_interface.h>
#include <deal.II/multigrid/mg_transfer_interface.templates.h>

DEAL_II_NAMESPACE_OPEN

namespace MGTransferUtil
{
  namespace
  {
    class CellProlongatorTest
    {
    public:
      template <int degree_fine, int degree_coarse>
      void
      run()
      {}
    };

    template <typename MeshType>
    MPI_Comm
    get_mpi_comm(const MeshType &mesh)
    {
      const auto *tria_parallel = dynamic_cast<
        const parallel::TriangulationBase<MeshType::dimension,
                                          MeshType::space_dimension> *>(
        &(mesh.get_triangulation()));

      return tria_parallel != nullptr ? tria_parallel->get_communicator() :
                                        MPI_COMM_SELF;
    }

    template <int dim>
    unsigned int
    compute_shift_within_children(const unsigned int child,
                                  const unsigned int fe_shift_1d,
                                  const unsigned int fe_degree)
    {
      // we put the degrees of freedom of all child cells in lexicographic
      // ordering
      unsigned int c_tensor_index[dim];
      unsigned int tmp = child;
      for (unsigned int d = 0; d < dim; ++d)
        {
          c_tensor_index[d] = tmp % 2;
          tmp /= 2;
        }
      const unsigned int n_child_dofs_1d = fe_degree + 1 + fe_shift_1d;
      unsigned int       factor          = 1;
      unsigned int       shift           = fe_shift_1d * c_tensor_index[0];
      for (unsigned int d = 1; d < dim; ++d)
        {
          factor *= n_child_dofs_1d;
          shift = shift + factor * fe_shift_1d * c_tensor_index[d];
        }
      return shift;
    }

    template <int dim>
    void
    get_child_offset(const unsigned int         child,
                     const unsigned int         fe_shift_1d,
                     const unsigned int         fe_degree,
                     std::vector<unsigned int> &local_dof_indices)
    {
      const unsigned int n_child_dofs_1d = fe_degree + 1 + fe_shift_1d;
      const unsigned int shift =
        compute_shift_within_children<dim>(child, fe_shift_1d, fe_degree);
      const unsigned int n_components =
        local_dof_indices.size() / Utilities::fixed_power<dim>(fe_degree + 1);
      const unsigned int n_scalar_cell_dofs =
        Utilities::fixed_power<dim>(n_child_dofs_1d);
      for (unsigned int c = 0, m = 0; c < n_components; ++c)
        for (unsigned int k = 0; k < (dim > 2 ? (fe_degree + 1) : 1); ++k)
          for (unsigned int j = 0; j < (dim > 1 ? (fe_degree + 1) : 1); ++j)
            for (unsigned int i = 0; i < (fe_degree + 1); ++i, ++m)
              local_dof_indices[m] = c * n_scalar_cell_dofs +
                                     k * n_child_dofs_1d * n_child_dofs_1d +
                                     j * n_child_dofs_1d + i + shift;
    }

    template <int dim>
    std::vector<std::vector<unsigned int>>
    get_child_offsets(const unsigned int n_cell_dofs_coarse,
                      const unsigned int fe_shift_1d,
                      const unsigned int fe_degree)
    {
      std::vector<std::vector<unsigned int>> cell_local_chilren_indices(
        GeometryInfo<dim>::max_children_per_cell,
        std::vector<unsigned int>(n_cell_dofs_coarse));
      {
        for (unsigned int c = 0; c < GeometryInfo<dim>::max_children_per_cell;
             c++)
          get_child_offset<dim>(c,
                                fe_shift_1d,
                                fe_degree,
                                cell_local_chilren_indices[c]);
      }
      return cell_local_chilren_indices;
    }

    template <int dim>
    types::coarse_cell_id
    convert_cell_id_binary_type_to_level_coarse_cell_id(
      const typename CellId::binary_type &binary_representation)
    {
      // exploiting the structure of CellId::binary_type
      // see also the documentation of CellId

      // actual coarse-grid id
      const unsigned int coarse_cell_id  = binary_representation[0];
      const unsigned int n_child_indices = binary_representation[1] >> 2;

      const unsigned int children_per_value =
        sizeof(CellId::binary_type::value_type) * 8 / dim;
      unsigned int child_level  = 0;
      unsigned int binary_entry = 2;

      // path to the get to the cell
      std::vector<unsigned int> cell_indices;
      while (child_level < n_child_indices)
        {
          Assert(binary_entry < binary_representation.size(),
                 ExcInternalError());

          for (unsigned int j = 0; j < children_per_value; ++j)
            {
              unsigned int cell_index =
                (((binary_representation[binary_entry] >> (j * dim))) &
                 (GeometryInfo<dim>::max_children_per_cell - 1));
              cell_indices.push_back(cell_index);
              ++child_level;
              if (child_level == n_child_indices)
                break;
            }
          ++binary_entry;
        }

      // compute new coarse-grid id: c_{i+1} = c_{i}*2^dim + q;
      types::coarse_cell_id level_coarse_cell_id = coarse_cell_id;
      for (auto i : cell_indices)
        level_coarse_cell_id =
          level_coarse_cell_id * GeometryInfo<dim>::max_children_per_cell + i;

      return level_coarse_cell_id;
    }

  } // namespace

  class FineDoFHandlerViewCell
  {
  public:
    FineDoFHandlerViewCell(
      const std::function<bool()> &has_children_,
      const std::function<void(std::vector<types::global_dof_index> &)>
        &get_dof_indices_)
      : has_children_(has_children_)
      , get_dof_indices_(get_dof_indices_)
    {}

    bool
    has_children() const
    {
      return has_children_();
    }

    void
    get_dof_indices(std::vector<types::global_dof_index> &dof_indices) const
    {
      get_dof_indices_(dof_indices);
    }


  private:
    const std::function<bool()> has_children_;
    const std::function<void(std::vector<types::global_dof_index> &)>
      get_dof_indices_;
  };

  template <int dim>
  class CellIDTranslator
  {
  public:
    CellIDTranslator(const unsigned int n_coarse_cells,
                     const unsigned int n_global_levels)
      : n_coarse_cells(n_coarse_cells)
      , n_global_levels(n_global_levels)
    {
      tree_sizes.push_back(0);
      for (unsigned int i = 0; i < n_global_levels; ++i)
        tree_sizes.push_back(
          tree_sizes.back() +
          Utilities::pow(GeometryInfo<dim>::max_children_per_cell, i) *
            n_coarse_cells);
    }

    unsigned int
    size() const
    {
      return n_coarse_cells *
             (Utilities::pow(GeometryInfo<dim>::max_children_per_cell,
                             n_global_levels) -
              1);
    }

    template <typename T>
    unsigned int
    translate(const T &cell) const
    {
      unsigned int id = 0;

      id += convert_cell_id_binary_type_to_level_coarse_cell_id<dim>(
        cell->id().template to_binary<dim>());

      id += tree_sizes[cell->level()];

      return id;
    }

    template <typename T>
    unsigned int
    translate(const T &cell, const unsigned int i) const
    {
      return translate(cell) * GeometryInfo<dim>::max_children_per_cell + i +
             tree_sizes[cell->level()];
    }

    CellId
    to_cell_id(const unsigned int id) const
    {
      std::vector<std::uint8_t> child_indices;

      unsigned int id_temp = id;

      unsigned int level = 0;

      for (; level < n_global_levels; ++level)
        if (id < tree_sizes[level])
          break;
      level -= 1;

      id_temp -= tree_sizes[level];

      for (unsigned int l = 0; l < level; ++l)
        {
          child_indices.push_back(id_temp %
                                  GeometryInfo<dim>::max_children_per_cell);
          id_temp /= GeometryInfo<dim>::max_children_per_cell;
        }

      std::reverse(child_indices.begin(), child_indices.end());

      return CellId(id_temp, child_indices); // TODO
    }

  private:
    const unsigned int        n_coarse_cells;
    const unsigned int        n_global_levels;
    std::vector<unsigned int> tree_sizes;
  };

  template <typename MeshType>
  class FineDoFHandlerView
  {
  public:
    FineDoFHandlerView(const MeshType &mesh_fine, const MeshType &mesh_coarse)
      : mesh_fine(mesh_fine)
      , mesh_coarse(mesh_coarse)
      , communicator(get_mpi_comm(mesh_fine) /*TODO: fix for different comms*/)
      , cell_id_translator(n_coarse_cells(mesh_fine),
                           n_global_levels(mesh_fine))
    {
      AssertDimension(n_coarse_cells(mesh_fine), n_coarse_cells(mesh_coarse));
      AssertIndexRange(n_global_levels(mesh_coarse),
                       n_global_levels(mesh_fine) + 1);
    }

    void
    reinit(const IndexSet &is_dst_locally_owned,
           const IndexSet &is_dst_remote_input,
           const IndexSet &is_src_locally_owned,
           const bool      check_if_elements_in_is_dst_remote_exist = false)
    {
      IndexSet is_dst_remote = is_dst_remote_input;

      if (check_if_elements_in_is_dst_remote_exist)
        {
          IndexSet is_dst_remote_potentially_relevant = is_dst_remote;
          is_dst_remote.clear();

          is_dst_remote_potentially_relevant.subtract_set(is_dst_locally_owned);

          std::vector<unsigned int> owning_ranks_of_ghosts(
            is_dst_remote_potentially_relevant.n_elements());

          {
            Utilities::MPI::internal::ComputeIndexOwner::
              ConsensusAlgorithmsPayload process(
                is_dst_locally_owned,
                is_dst_remote_potentially_relevant,
                communicator,
                owning_ranks_of_ghosts,
                false);

            Utilities::MPI::ConsensusAlgorithms::Selector<
              std::pair<types::global_dof_index, types::global_dof_index>,
              unsigned int>
              consensus_algorithm(process, communicator);
            consensus_algorithm.run();
          }

          for (unsigned i = 0;
               i < is_dst_remote_potentially_relevant.n_elements();
               ++i)
            if (owning_ranks_of_ghosts[i] != numbers::invalid_unsigned_int)
              is_dst_remote.add_index(
                is_dst_remote_potentially_relevant.nth_index_in_set(i));
        }

      // determine owner of remote cells
      std::vector<unsigned int> is_dst_remote_owners(
        is_dst_remote.n_elements());

      Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload
        process(is_dst_locally_owned,
                is_dst_remote,
                communicator,
                is_dst_remote_owners,
                true);

      Utilities::MPI::ConsensusAlgorithms::Selector<
        std::pair<types::global_dof_index, types::global_dof_index>,
        unsigned int>
        consensus_algorithm(process, communicator);
      consensus_algorithm.run();

      this->is_dst_locally_owned = is_dst_locally_owned;
      this->is_dst_remote        = is_dst_remote;
      this->is_src_locally_owned = is_src_locally_owned;

#if false
        std::cout << "IS_SRC_LOCALLY_OWNED" << std::endl;
        for (auto i : is_src_locally_owned)
          std::cout << cell_id_translator.to_cell_id(i) << std::endl;
        std::cout << std::endl << std::endl << std::endl;


        std::cout << "IS_DST_LOCALLY_OWNED" << std::endl;
        for (auto i : is_dst_locally_owned)
          std::cout << cell_id_translator.to_cell_id(i) << std::endl;
        std::cout << std::endl << std::endl << std::endl;
#endif

      const auto &dof_handler_dst = mesh_fine; // TODO: remove
      const auto &tria_dst        = mesh_fine.get_triangulation(); // TODO

      const unsigned int dofs_per_cell =
        mesh_coarse.get_fe_collection()[0].dofs_per_cell;

      const auto targets_with_indexset = process.get_requesters();

      std::map<unsigned int, std::vector<unsigned int>> indices_to_be_sent;
      std::vector<MPI_Request>                          requests;
      requests.reserve(targets_with_indexset.size());

      {
        std::vector<types::global_dof_index> indices(dofs_per_cell);

        for (auto i : targets_with_indexset)
          {
            indices_to_be_sent[i.first] = {};
            auto &buffer                = indices_to_be_sent[i.first];

            for (auto cell_id : i.second)
              {
                typename MeshType::cell_iterator cell(
                  *cell_id_translator.to_cell_id(cell_id).to_cell(tria_dst),
                  &dof_handler_dst);

                cell->get_dof_indices(indices);
                buffer.insert(buffer.end(), indices.begin(), indices.end());
              }

            requests.resize(requests.size() + 1);

            MPI_Isend(buffer.data(),
                      buffer.size(),
                      MPI_UNSIGNED,
                      i.first,
                      11,
                      communicator,
                      &requests.back());
          }
      }


      std::vector<unsigned int> ghost_indices;

      {
        std::map<unsigned int, std::vector<unsigned int>> rank_to_ids;

        std::set<unsigned int> ranks;

        for (auto i : is_dst_remote_owners)
          ranks.insert(i);

        for (auto i : ranks)
          rank_to_ids[i] = {};

        for (unsigned int i = 0; i < is_dst_remote_owners.size(); ++i)
          rank_to_ids[is_dst_remote_owners[i]].push_back(
            is_dst_remote.nth_index_in_set(i));


        for (unsigned int i = 0; i < ranks.size(); ++i)
          {
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, 11, communicator, &status);

            int message_length;
            MPI_Get_count(&status, MPI_UNSIGNED, &message_length);

            std::vector<unsigned int> buffer(message_length);

            MPI_Recv(buffer.data(),
                     buffer.size(),
                     MPI_UNSIGNED,
                     status.MPI_SOURCE,
                     11,
                     communicator,
                     MPI_STATUS_IGNORE);

            ghost_indices.insert(ghost_indices.end(),
                                 buffer.begin(),
                                 buffer.end());

            const unsigned int rank = status.MPI_SOURCE;

            const auto ids = rank_to_ids[rank];

            std::vector<types::global_dof_index> indices(dofs_per_cell);

            for (unsigned int i = 0, k = 0; i < ids.size(); ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j, ++k)
                  indices[j] = buffer[k];
                map[ids[i]] = indices;
              }
          }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
      }

      std::sort(ghost_indices.begin(), ghost_indices.end());
      ghost_indices.erase(std::unique(ghost_indices.begin(),
                                      ghost_indices.end()),
                          ghost_indices.end());

      this->is_extended_locally_owned = dof_handler_dst.locally_owned_dofs();

      this->is_extendende_ghosts = IndexSet(dof_handler_dst.n_dofs());
      this->is_extendende_ghosts.add_indices(ghost_indices.begin(),
                                             ghost_indices.end());
      this->is_extendende_ghosts.subtract_set(this->is_extended_locally_owned);

      is_extended_locally_owned.print(std::cout);
      is_extendende_ghosts.print(std::cout);
    }

    FineDoFHandlerViewCell
    get_cell(const typename MeshType::cell_iterator &cell) const
    {
      const auto id = this->cell_id_translator.translate(cell);

      const bool is_cell_locally_owned =
        this->is_dst_locally_owned.is_element(id);
      const bool is_cell_remotly_owned = this->is_dst_remote.is_element(id);

      const bool has_cell_any_children = [&]() {
        for (unsigned int i = 0;
             i < GeometryInfo<MeshType::dimension>::max_children_per_cell;
             ++i)
          {
            const auto j = this->cell_id_translator.translate(cell, i);

            if (this->is_dst_locally_owned.is_element(j))
              return true;

            if (this->is_dst_remote.is_element(j))
              return true;
          }

        return false;
      }();

      return FineDoFHandlerViewCell(
        [has_cell_any_children]() { return has_cell_any_children; },
        [cell, is_cell_locally_owned, is_cell_remotly_owned, id, this](
          auto &dof_indices) {
          if (is_cell_locally_owned)
            {
              (typename MeshType::cell_iterator(
                 *cell->id().to_cell(mesh_fine.get_triangulation()),
                 &mesh_fine))
                ->get_dof_indices(dof_indices);
            }
          else if (is_cell_remotly_owned)
            {
              dof_indices = map.at(id);
            }
          else
            {
              AssertThrow(false, ExcNotImplemented()); // should not happen!
            }
        });
    }

    FineDoFHandlerViewCell
    get_cell(const typename MeshType::cell_iterator &cell,
             const unsigned int                      c) const
    {
      const auto id = this->cell_id_translator.translate(cell, c);

      const bool is_cell_locally_owned =
        this->is_dst_locally_owned.is_element(id);
      const bool is_cell_remotly_owned = this->is_dst_remote.is_element(id);

      return FineDoFHandlerViewCell(
        [cell, c, this]() {
          AssertThrow(false, ExcNotImplemented()); // currently we do not need
                                                   // children of children

          return false;
        },
        [cell, is_cell_locally_owned, is_cell_remotly_owned, c, id, this](
          auto &dof_indices) {
          if (is_cell_locally_owned)
            {
              (typename MeshType::cell_iterator(
                 *cell->id().to_cell(mesh_fine.get_triangulation()),
                 &mesh_fine))
                ->child(c)
                ->get_dof_indices(dof_indices);
            }
          else if (is_cell_remotly_owned)
            {
              dof_indices = map.at(id);
            }
          else
            {
              AssertThrow(false, ExcNotImplemented()); // should not happen!
            }
        });
    }

    const IndexSet &
    locally_owned_dofs() const
    {
      return is_extended_locally_owned;
    }

    const IndexSet &
    locally_relevant_dofs() const
    {
      return is_extendende_ghosts;
    }

  private:
    const MeshType &mesh_fine;
    const MeshType &mesh_coarse;
    const MPI_Comm  communicator;

  protected:
    const CellIDTranslator<MeshType::dimension> cell_id_translator;

  private:
    IndexSet is_dst_locally_owned;
    IndexSet is_dst_remote;
    IndexSet is_src_locally_owned;


    IndexSet is_extended_locally_owned;
    IndexSet is_extendende_ghosts;

    std::map<unsigned int, std::vector<types::global_dof_index>> map;

    static unsigned int
    n_coarse_cells(const MeshType &mesh)
    {
      types::coarse_cell_id n_coarse_cells = 0;

      for (auto cell : mesh.get_triangulation().active_cell_iterators())
        if (!cell->is_artificial())
          n_coarse_cells =
            std::max(n_coarse_cells, cell->id().get_coarse_cell_id());

      return Utilities::MPI::max(n_coarse_cells, get_mpi_comm(mesh)) + 1;
    }

    static unsigned int
    n_global_levels(const MeshType &mesh)
    {
      return mesh.get_triangulation().n_global_levels();
    }
  };

  template <typename MeshType>
  class GlobalCoarseningFineDoFHandlerView : public FineDoFHandlerView<MeshType>
  {
  public:
    GlobalCoarseningFineDoFHandlerView(const MeshType &dof_handler_dst,
                                       const MeshType &dof_handler_src)
      : FineDoFHandlerView<MeshType>(dof_handler_dst, dof_handler_src)
    {
      // get reference to triangulations
      const auto &tria_dst = dof_handler_dst.get_triangulation();
      const auto &tria_src = dof_handler_src.get_triangulation();

      // create index sets
      IndexSet is_dst_locally_owned(this->cell_id_translator.size());
      IndexSet is_dst_remote(this->cell_id_translator.size());
      IndexSet is_src_locally_owned(this->cell_id_translator.size());

      for (auto cell : tria_dst.active_cell_iterators())
        if (!cell->is_artificial() && cell->is_locally_owned())
          is_dst_locally_owned.add_index(
            this->cell_id_translator.translate(cell));


      for (auto cell : tria_src.active_cell_iterators())
        if (!cell->is_artificial() && cell->is_locally_owned())
          {
            is_src_locally_owned.add_index(
              this->cell_id_translator.translate(cell));
            is_dst_remote.add_index(this->cell_id_translator.translate(cell));

            if (cell->level() + 1u == tria_dst.n_global_levels())
              continue;

            for (unsigned int i = 0;
                 i < GeometryInfo<MeshType::dimension>::max_children_per_cell;
                 ++i)
              is_dst_remote.add_index(
                this->cell_id_translator.translate(cell, i));
          }

      this->reinit(is_dst_locally_owned,
                   is_dst_remote,
                   is_src_locally_owned,
                   true);
    }
  };

  template <typename MeshType>
  class PermutationFineDoFHandlerView : public FineDoFHandlerView<MeshType>
  {
  public:
    PermutationFineDoFHandlerView(const MeshType &dof_handler_dst,
                                  const MeshType &dof_handler_src)
      : FineDoFHandlerView<MeshType>(dof_handler_dst, dof_handler_src)
    {
      // get reference to triangulations
      const auto &tria_dst = dof_handler_dst.get_triangulation();
      const auto &tria_src = dof_handler_src.get_triangulation();

      // create index sets
      IndexSet is_dst_locally_owned(this->cell_id_translator.size());
      IndexSet is_dst_remote(this->cell_id_translator.size());
      IndexSet is_src_locally_owned(this->cell_id_translator.size());

      for (auto cell : tria_dst.active_cell_iterators())
        if (!cell->is_artificial() && cell->is_locally_owned())
          is_dst_locally_owned.add_index(
            this->cell_id_translator.translate(cell));


      for (auto cell : tria_src.active_cell_iterators())
        if (!cell->is_artificial() && cell->is_locally_owned())
          {
            is_src_locally_owned.add_index(
              this->cell_id_translator.translate(cell));
            is_dst_remote.add_index(this->cell_id_translator.translate(cell));
          }

      this->reinit(is_dst_locally_owned,
                   is_dst_remote,
                   is_src_locally_owned,
                   false);
    }
  };

  bool
  polynomial_transfer_supported(const unsigned int fe_degree_fine,
                                const unsigned int fe_degree_coarse)
  {
    CellTransfer        cell_transfer(fe_degree_fine, fe_degree_coarse);
    CellProlongatorTest cell_transfer_test;

    return cell_transfer.run(cell_transfer_test);
  }



  template <int dim, typename Number, typename MeshType>
  void
  setup_vector_repartitioner(const MeshType &dof_handler_fine,
                             const MeshType &dof_handler_coarse,
                             VectorRepartitioner<dim, Number> &transfer)
  {
    const PermutationFineDoFHandlerView<MeshType> view(dof_handler_fine,
                                                       dof_handler_coarse);

    const auto &is_extended_locally_owned = view.locally_owned_dofs();
    const auto &is_extendende_ghosts      = view.locally_relevant_dofs();

    transfer.extended_partitioner.reset(new Utilities::MPI::Partitioner(
      is_extended_locally_owned,
      is_extendende_ghosts,
      get_mpi_comm(dof_handler_fine) /*TODO: generalize*/));

    transfer.indices.resize(
      dof_handler_coarse.locally_owned_dofs().n_elements());

    const unsigned int dofs_per_cell =
      dof_handler_fine.get_fe_collection()[0].dofs_per_cell;

    std::vector<types::global_dof_index> indices_coarse(dofs_per_cell);
    std::vector<types::global_dof_index> indices_fine(dofs_per_cell);

    // loop over all cells
    for (auto cell_coarse : dof_handler_coarse.active_cell_iterators())
      {
        if (cell_coarse->is_artificial() ||
            cell_coarse->is_locally_owned() == false)
          continue;

        const auto cell_fine = view.get_cell(cell_coarse);

        cell_coarse->get_dof_indices(indices_coarse);
        cell_fine.get_dof_indices(indices_fine);

        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            if (dof_handler_coarse.locally_owned_dofs().is_element(
                  indices_coarse[j]))
              transfer.indices[dof_handler_coarse.locally_owned_dofs()
                                 .index_within_set(indices_coarse[j])] =
                indices_fine[j];
          }
      }

    for (auto &i : transfer.indices)
      if (is_extended_locally_owned.is_element(i))
        i = is_extended_locally_owned.index_within_set(i);
      else if (is_extendende_ghosts.is_element(i))
        i = is_extended_locally_owned.n_elements() +
            is_extendende_ghosts.index_within_set(i);
      else
        Assert(false, ExcNotImplemented());
  }



  template <int dim, typename Number, typename MeshType>
  void
  setup_global_coarsening_transfer(
    const MeshType &                 dof_handler_fine,
    const MeshType &                 dof_handler_coarse,
    const AffineConstraints<Number> &constraint_fine,
    const AffineConstraints<Number> &constraint_coarse,
    Transfer<dim, Number> &          transfer)
  {
    const GlobalCoarseningFineDoFHandlerView<MeshType> view(dof_handler_fine,
                                                            dof_handler_coarse);

    // copy constrain matrix; TODO: why only for the coarse level?
    transfer.constraint_coarse.copy_from(constraint_coarse);

    // create partitioners and vectors for internal purposes
    {
      // ... for fine mesh
      {
        transfer.partitioner_fine.reset(
          new Utilities::MPI::Partitioner(view.locally_owned_dofs(),
                                          view.locally_relevant_dofs(),
                                          get_mpi_comm(dof_handler_fine)));
        transfer.vec_fine.reinit(transfer.partitioner_fine);
      }

      // ... coarse mesh (needed since user vector might be const)
      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler_coarse,
                                                locally_relevant_dofs);

        transfer.partitioner_coarse.reset(new Utilities::MPI::Partitioner(
          dof_handler_coarse.locally_owned_dofs(),
          locally_relevant_dofs,
          get_mpi_comm(dof_handler_coarse)));
        transfer.vec_coarse.reinit(transfer.partitioner_coarse);
      }
    }

    // helper function: to process the fine level cells; function @p fu_1 is
    // performed on cells that are not refined and @fu_2 is performed on
    // children of cells that are refined
    const auto process_cells = [&view,
                                &dof_handler_coarse](const auto &fu_non_refined,
                                                     const auto &fu_refined) {
      // loop over all active locally-owned cells
      for (const auto cell_coarse : dof_handler_coarse.active_cell_iterators())
        {
          if (cell_coarse->is_locally_owned() == false)
            continue;

          // get a reference to the equivalent cell on the fine triangulation
          const auto cell_coarse_on_fine_mesh = view.get_cell(cell_coarse);

          // check if cell has children
          if (cell_coarse_on_fine_mesh.has_children())
            // ... cell has children -> process children
            for (unsigned int c = 0;
                 c < GeometryInfo<dim>::max_children_per_cell;
                 c++)
              fu_refined(cell_coarse, view.get_cell(cell_coarse, c), c);
          else // ... cell has no children -> process cell
            fu_non_refined(cell_coarse, cell_coarse_on_fine_mesh);
        }
    };

    // set up two mg-schesmes
    //   (0) no refinement -> identity
    //   (1) h-refinement
    transfer.schemes.resize(2);

    // check if FE is the same; TODO: better way?
    AssertDimension(dof_handler_coarse.get_fe(0).dofs_per_cell,
                    dof_handler_fine.get_fe(0).dofs_per_cell);

    // number of dofs on coarse and fine cells
    transfer.schemes[0].n_cell_dofs_coarse =
      transfer.schemes[0].n_cell_dofs_fine =
        transfer.schemes[1].n_cell_dofs_coarse =
          dof_handler_coarse.get_fe(0).dofs_per_cell;
    transfer.schemes[1].n_cell_dofs_fine =
      dof_handler_coarse.get_fe(0).dofs_per_cell *
      GeometryInfo<dim>::max_children_per_cell;

    // degree of fe on coarse and fine cell
    transfer.schemes[0].degree_coarse   = transfer.schemes[0].degree_fine =
      transfer.schemes[1].degree_coarse = dof_handler_coarse.get_fe(0).degree;
    transfer.schemes[1].degree_fine =
      dof_handler_coarse.get_fe(0).degree * 2 + 1;

    // continuous or discontinuous
    transfer.schemes[0].fine_element_is_continuous =
      transfer.schemes[1].fine_element_is_continuous =
        dof_handler_fine.get_fe(0).dofs_per_vertex > 0;

    // count coarse cells for each scheme (0, 1)
    {
      transfer.schemes[0].n_cells_coarse = 0; // reset
      transfer.schemes[1].n_cells_coarse = 0;

      // count by looping over all coarse cells
      process_cells([&](const auto &,
                        const auto &) { transfer.schemes[0].n_cells_coarse++; },
                    [&](const auto &, const auto &, const auto c) {
                      if (c == 0)
                        transfer.schemes[1].n_cells_coarse++;
                    });
    }


    const auto cell_local_chilren_indices =
      get_child_offsets<dim>(transfer.schemes[0].n_cell_dofs_coarse,
                             dof_handler_fine.get_fe(0).degree + 1,
                             dof_handler_fine.get_fe(0).degree);


    // indices
    {
      transfer.schemes[0].level_dof_indices_fine.resize(
        transfer.schemes[0].n_cell_dofs_fine *
        transfer.schemes[0].n_cells_coarse);
      transfer.schemes[0].level_dof_indices_coarse.resize(
        transfer.schemes[0].n_cell_dofs_coarse *
        transfer.schemes[0].n_cells_coarse);

      transfer.schemes[1].level_dof_indices_fine.resize(
        transfer.schemes[1].n_cell_dofs_fine *
        transfer.schemes[1].n_cells_coarse);
      transfer.schemes[1].level_dof_indices_coarse.resize(
        transfer.schemes[1].n_cell_dofs_coarse *
        transfer.schemes[1].n_cells_coarse);

      std::vector<types::global_dof_index> local_dof_indices(
        transfer.schemes[0].n_cell_dofs_coarse);

      // ----------------------- lexicographic_numbering -----------------------
      std::vector<unsigned int> lexicographic_numbering;
      {
        const Quadrature<1> dummy_quadrature(
          std::vector<Point<1>>(1, Point<1>()));
        internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
        shape_info.reinit(dummy_quadrature, dof_handler_fine.get_fe(0), 0);
        lexicographic_numbering = shape_info.lexicographic_numbering;
      }

      // ------------------------------- indices -------------------------------
      unsigned int *level_dof_indices_coarse_0 =
        &transfer.schemes[0].level_dof_indices_coarse[0];
      unsigned int *level_dof_indices_fine_0 =
        &transfer.schemes[0].level_dof_indices_fine[0];

      unsigned int *level_dof_indices_coarse_1 =
        &transfer.schemes[1].level_dof_indices_coarse[0];
      unsigned int *level_dof_indices_fine_1 =
        &transfer.schemes[1].level_dof_indices_fine[0];

      process_cells(
        [&](const auto &cell_coarse, const auto &cell_fine) {
          // parent
          {
            cell_coarse->get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < transfer.schemes[0].n_cell_dofs_coarse;
                 i++)
              level_dof_indices_coarse_0[i] =
                transfer.partitioner_coarse->global_to_local(
                  local_dof_indices[lexicographic_numbering[i]]);
          }

          // child
          {
            cell_fine.get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < transfer.schemes[0].n_cell_dofs_coarse;
                 i++)
              level_dof_indices_fine_0[i] =
                transfer.partitioner_fine->global_to_local(
                  local_dof_indices[lexicographic_numbering[i]]);
          }

          // move pointers
          {
            level_dof_indices_coarse_0 +=
              transfer.schemes[0].n_cell_dofs_coarse;
            level_dof_indices_fine_0 += transfer.schemes[0].n_cell_dofs_fine;
          }
        },
        [&](const auto &cell_coarse, const auto &cell_fine, const auto c) {
          // parent (only once at the beginning)
          if (c == 0)
            {
              cell_coarse->get_dof_indices(local_dof_indices);
              for (unsigned int i = 0;
                   i < transfer.schemes[1].n_cell_dofs_coarse;
                   i++)
                level_dof_indices_coarse_1[i] =
                  transfer.partitioner_coarse->global_to_local(
                    local_dof_indices[lexicographic_numbering[i]]);
            }

          // child
          {
            cell_fine.get_dof_indices(local_dof_indices);
            for (unsigned int i = 0; i < transfer.schemes[1].n_cell_dofs_coarse;
                 i++)
              level_dof_indices_fine_1[cell_local_chilren_indices[c][i]] =
                transfer.partitioner_fine->global_to_local(
                  local_dof_indices[lexicographic_numbering[i]]);
          }

          // move pointers (only once at the end)
          if (c + 1 == GeometryInfo<dim>::max_children_per_cell)
            {
              level_dof_indices_coarse_1 +=
                transfer.schemes[1].n_cell_dofs_coarse;
              level_dof_indices_fine_1 += transfer.schemes[1].n_cell_dofs_fine;
            }
        });
    }

    // -------------- prolongation matrix (0) -> identity matrix ---------------
    {
      AssertDimension(dof_handler_fine.get_fe(0).n_base_elements(), 1);
      std::string fe_name =
        dof_handler_fine.get_fe(0).base_element(0).get_name();
      {
        const std::size_t template_starts = fe_name.find_first_of('<');
        Assert(fe_name[template_starts + 1] ==
                 (dim == 1 ? '1' : (dim == 2 ? '2' : '3')),
               ExcInternalError());
        fe_name[template_starts + 1] = '1';
      }
      const std::unique_ptr<FiniteElement<1>> fe(
        FETools::get_fe_by_name<1, 1>(fe_name));

      transfer.schemes[0].prolongation_matrix_1d.resize(fe->dofs_per_cell *
                                                        fe->dofs_per_cell);

      for (unsigned int i = 0; i < fe->dofs_per_cell; i++)
        transfer.schemes[0].prolongation_matrix_1d[i + i * fe->dofs_per_cell] =
          Number(1.0);
    }

    // ------------------------ prolongation matrix (1) ------------------------
    {
      AssertDimension(dof_handler_fine.get_fe(0).n_base_elements(), 1);
      std::string fe_name =
        dof_handler_fine.get_fe(0).base_element(0).get_name();
      {
        const std::size_t template_starts = fe_name.find_first_of('<');
        Assert(fe_name[template_starts + 1] ==
                 (dim == 1 ? '1' : (dim == 2 ? '2' : '3')),
               ExcInternalError());
        fe_name[template_starts + 1] = '1';
      }
      const std::unique_ptr<FiniteElement<1>> fe(
        FETools::get_fe_by_name<1, 1>(fe_name));

      std::vector<unsigned int> renumbering(fe->dofs_per_cell);
      {
        AssertIndexRange(fe->dofs_per_vertex, 2);
        renumbering[0] = 0;
        for (unsigned int i = 0; i < fe->dofs_per_line; ++i)
          renumbering[i + fe->dofs_per_vertex] =
            GeometryInfo<1>::vertices_per_cell * fe->dofs_per_vertex + i;
        if (fe->dofs_per_vertex > 0)
          renumbering[fe->dofs_per_cell - fe->dofs_per_vertex] =
            fe->dofs_per_vertex;
      }

      // TODO: data structures are saved in form of DG data structures here
      const unsigned int shift           = fe->dofs_per_cell;
      const unsigned int n_child_dofs_1d = fe->dofs_per_cell * 2;

      transfer.schemes[1].prolongation_matrix_1d.resize(fe->dofs_per_cell *
                                                        n_child_dofs_1d);

      for (unsigned int c = 0; c < GeometryInfo<1>::max_children_per_cell; ++c)
        for (unsigned int i = 0; i < fe->dofs_per_cell; ++i)
          for (unsigned int j = 0; j < fe->dofs_per_cell; ++j)
            transfer.schemes[1]
              .prolongation_matrix_1d[i * n_child_dofs_1d + j + c * shift] =
              fe->get_prolongation_matrix(c)(renumbering[j], renumbering[i]);
    }


    // -------------------------------- weights --------------------------------
    if (transfer.schemes[0].fine_element_is_continuous)
      {
        LinearAlgebra::distributed::Vector<Number> touch_count, touch_count_;

        touch_count.reinit(transfer.partitioner_fine);

        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler_fine,
                                                locally_relevant_dofs);

        const auto partitioner_fine_ =
          std::make_shared<Utilities::MPI::Partitioner>(
            dof_handler_fine.locally_owned_dofs(),
            locally_relevant_dofs,
            get_mpi_comm(dof_handler_fine));
        transfer.vec_fine.reinit(transfer.partitioner_fine);

        touch_count_.reinit(partitioner_fine_);

        std::vector<types::global_dof_index> local_dof_indices(
          transfer.schemes[0].n_cell_dofs_coarse);

        for (auto cell : dof_handler_fine.active_cell_iterators())
          {
            if (cell->is_locally_owned() == false)
              continue;

            cell->get_dof_indices(local_dof_indices);

            for (auto i : local_dof_indices)
              if (constraint_fine.is_constrained(i) == false)
                touch_count_[i] += 1;
          }

        touch_count_.compress(VectorOperation::add);

        for (unsigned int i = 0; i < touch_count_.local_size(); ++i)
          touch_count_.local_element(i) =
            constraint_fine.is_constrained(
              touch_count_.get_partitioner()->local_to_global(i)) ?
              Number(0.) :
              Number(1.) / touch_count_.local_element(i);

        // TODO: needed?
        touch_count_.update_ghost_values();


        // copy weights to other indexset
        touch_count.copy_locally_owned_data_from(touch_count_);
        touch_count.update_ghost_values();

        transfer.schemes[0].weights.resize(
          transfer.schemes[0].n_cells_coarse *
          transfer.schemes[0].n_cell_dofs_fine);
        transfer.schemes[1].weights.resize(
          transfer.schemes[1].n_cells_coarse *
          transfer.schemes[1].n_cell_dofs_fine);

        Number *      weights_0 = &transfer.schemes[0].weights[0];
        Number *      weights_1 = &transfer.schemes[1].weights[0];
        unsigned int *dof_indices_fine_0 =
          &transfer.schemes[0].level_dof_indices_fine[0];
        unsigned int *dof_indices_fine_1 =
          &transfer.schemes[1].level_dof_indices_fine[0];

        process_cells(
          [&](const auto &, const auto &) {
            for (unsigned int i = 0; i < transfer.schemes[0].n_cell_dofs_fine;
                 i++)
              weights_0[i] = touch_count.local_element(dof_indices_fine_0[i]);

            dof_indices_fine_0 += transfer.schemes[0].n_cell_dofs_fine;
            weights_0 += transfer.schemes[0].n_cell_dofs_fine;
          },
          [&](const auto &, const auto &, const auto c) {
            for (unsigned int i = 0; i < transfer.schemes[1].n_cell_dofs_coarse;
                 i++)
              weights_1[cell_local_chilren_indices[c][i]] =
                touch_count.local_element(
                  dof_indices_fine_1[cell_local_chilren_indices[c][i]]);

            // move pointers (only once at the end)
            if (c + 1 == GeometryInfo<dim>::max_children_per_cell)
              {
                dof_indices_fine_1 += transfer.schemes[1].n_cell_dofs_fine;
                weights_1 += transfer.schemes[1].n_cell_dofs_fine;
              }
          });
      }
  }


  template <int dim, typename Number, typename MeshType>
  void
  setup_polynomial_transfer(const MeshType &                 dof_handler_fine,
                            const MeshType &                 dof_handler_coarse,
                            const AffineConstraints<Number> &constraint_fine,
                            const AffineConstraints<Number> &constraint_coarse,
                            Transfer<dim, Number> &          transfer)
  {
    AssertDimension(dof_handler_fine.get_triangulation().n_active_cells(),
                    dof_handler_coarse.get_triangulation().n_active_cells());

    auto process_cells = [&](const auto &fu) {
      typename MeshType::active_cell_iterator            //
        cell_coarse = dof_handler_coarse.begin_active(), //
        end_coarse  = dof_handler_coarse.end(),          //
        cell_fine   = dof_handler_fine.begin_active();

      for (; cell_coarse != end_coarse; cell_coarse++, cell_fine++)
        {
          if (!cell_coarse->is_locally_owned())
            continue;

          fu(cell_coarse, cell_fine);
        }
    };

    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      fe_index_pairs;

    process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
      fe_index_pairs.emplace(
        std::pair<unsigned int, unsigned int>(cell_coarse->active_fe_index(),
                                              cell_fine->active_fe_index()),
        0);
    });

    unsigned int counter = 0;
    for (auto &f : fe_index_pairs)
      f.second = counter++;

    transfer.schemes.resize(fe_index_pairs.size());

    // extract number of coarse cells
    {
      for (auto &scheme : transfer.schemes)
        scheme.n_cells_coarse = 0;
      process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
        transfer
          .schemes[fe_index_pairs[std::pair<unsigned int, unsigned int>(
            cell_coarse->active_fe_index(), cell_fine->active_fe_index())]]
          .n_cells_coarse++;
      });
    }

    for (const auto fe_index_pair : fe_index_pairs)
      {
        transfer.schemes[fe_index_pair.second].n_cell_dofs_coarse =
          dof_handler_coarse.get_fe(fe_index_pair.first.first).dofs_per_cell;
        transfer.schemes[fe_index_pair.second].n_cell_dofs_fine =
          dof_handler_fine.get_fe(fe_index_pair.first.second).dofs_per_cell;

        transfer.schemes[fe_index_pair.second].degree_coarse =
          dof_handler_coarse.get_fe(fe_index_pair.first.first).degree;
        transfer.schemes[fe_index_pair.second].degree_fine =
          dof_handler_fine.get_fe(fe_index_pair.first.second).degree;

        transfer.schemes[fe_index_pair.second].fine_element_is_continuous =
          dof_handler_fine.get_fe(fe_index_pair.first.second).dofs_per_vertex >
          0;
      }

    transfer.constraint_coarse.copy_from(constraint_coarse);

    MPI_Comm comm;
    {
      const parallel::TriangulationBase<dim> *dist_tria =
        dynamic_cast<const parallel::TriangulationBase<dim> *>(
          &(dof_handler_coarse.get_triangulation()));

      comm =
        dist_tria != nullptr ? dist_tria->get_communicator() : MPI_COMM_SELF;

      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler_fine,
                                                locally_relevant_dofs);

        transfer.partitioner_fine.reset(new Utilities::MPI::Partitioner(
          dof_handler_fine.locally_owned_dofs(), locally_relevant_dofs, comm));
        transfer.vec_fine.reinit(transfer.partitioner_fine);
      }
      {
        IndexSet locally_relevant_dofs;
        DoFTools::extract_locally_relevant_dofs(dof_handler_coarse,
                                                locally_relevant_dofs);

        transfer.partitioner_coarse.reset(new Utilities::MPI::Partitioner(
          dof_handler_coarse.locally_owned_dofs(),
          locally_relevant_dofs,
          comm));
        transfer.vec_coarse.reinit(transfer.partitioner_coarse);
      }
    }

    {
      std::vector<std::vector<unsigned int>> lexicographic_numbering_fine(
        fe_index_pairs.size());
      std::vector<std::vector<unsigned int>> lexicographic_numbering_coarse(
        fe_index_pairs.size());
      std::vector<std::vector<types::global_dof_index>>
                                                        local_dof_indices_coarse(fe_index_pairs.size());
      std::vector<std::vector<types::global_dof_index>> local_dof_indices_fine(
        fe_index_pairs.size());

      for (const auto fe_index_pair : fe_index_pairs)
        {
          local_dof_indices_coarse[fe_index_pair.second].resize(
            transfer.schemes[fe_index_pair.second].n_cell_dofs_coarse);
          local_dof_indices_fine[fe_index_pair.second].resize(
            transfer.schemes[fe_index_pair.second].n_cell_dofs_fine);

          transfer.schemes[fe_index_pair.second].level_dof_indices_fine.resize(
            transfer.schemes[fe_index_pair.second].n_cell_dofs_fine *
            transfer.schemes[fe_index_pair.second].n_cells_coarse);
          transfer.schemes[fe_index_pair.second]
            .level_dof_indices_coarse.resize(
              transfer.schemes[fe_index_pair.second].n_cell_dofs_coarse *
              transfer.schemes[fe_index_pair.second].n_cells_coarse);


          // ----------------------- lexicographic_numbering
          // -----------------------
          {
            const Quadrature<1> dummy_quadrature(
              std::vector<Point<1>>(1, Point<1>()));
            internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
            shape_info.reinit(dummy_quadrature,
                              dof_handler_fine.get_fe(
                                fe_index_pair.first.second),
                              0);
            lexicographic_numbering_fine[fe_index_pair.second] =
              shape_info.lexicographic_numbering;

            shape_info.reinit(dummy_quadrature,
                              dof_handler_coarse.get_fe(
                                fe_index_pair.first.first),
                              0);
            lexicographic_numbering_coarse[fe_index_pair.second] =
              shape_info.lexicographic_numbering;
          }
        }

      // ------------------------------- indices -------------------------------
      std::vector<unsigned int *> level_dof_indices_coarse_(
        fe_index_pairs.size());
      std::vector<unsigned int *> level_dof_indices_fine_(
        fe_index_pairs.size());

      for (unsigned int i = 0; i < fe_index_pairs.size(); i++)
        {
          level_dof_indices_coarse_[i] =
            &transfer.schemes[i].level_dof_indices_coarse[0];
          level_dof_indices_fine_[i] =
            &transfer.schemes[i].level_dof_indices_fine[0];
        }

      process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
        const auto fe_pair_no =
          fe_index_pairs[std::pair<unsigned int, unsigned int>(
            cell_coarse->active_fe_index(), cell_fine->active_fe_index())];

        cell_coarse->get_dof_indices(local_dof_indices_coarse[fe_pair_no]);
        for (unsigned int i = 0;
             i < transfer.schemes[fe_pair_no].n_cell_dofs_coarse;
             i++)
          level_dof_indices_coarse_[fe_pair_no][i] =
            transfer.partitioner_coarse->global_to_local(
              local_dof_indices_coarse
                [fe_pair_no][lexicographic_numbering_coarse[fe_pair_no][i]]);


        cell_fine->get_dof_indices(local_dof_indices_fine[fe_pair_no]);
        for (unsigned int i = 0;
             i < transfer.schemes[fe_pair_no].n_cell_dofs_fine;
             i++)
          level_dof_indices_fine_[fe_pair_no][i] =
            transfer.partitioner_fine->global_to_local(
              local_dof_indices_fine
                [fe_pair_no][lexicographic_numbering_fine[fe_pair_no][i]]);


        level_dof_indices_coarse_[fe_pair_no] +=
          transfer.schemes[fe_pair_no].n_cell_dofs_coarse;
        level_dof_indices_fine_[fe_pair_no] +=
          transfer.schemes[fe_pair_no].n_cell_dofs_fine;
      });
    }

    // -------------------------- prolongation matrix --------------------------
    for (auto const &[fe_index_pair, fe_index_no] : fe_index_pairs)
      {
        AssertDimension(
          dof_handler_fine.get_fe(fe_index_pair.second).n_base_elements(), 1);
        std::string fe_name_fine = dof_handler_fine.get_fe(fe_index_pair.second)
                                     .base_element(0)
                                     .get_name();
        {
          const std::size_t template_starts = fe_name_fine.find_first_of('<');
          Assert(fe_name_fine[template_starts + 1] ==
                   (dim == 1 ? '1' : (dim == 2 ? '2' : '3')),
                 ExcInternalError());
          fe_name_fine[template_starts + 1] = '1';
        }
        const std::unique_ptr<FiniteElement<1>> fe_fine(
          FETools::get_fe_by_name<1, 1>(fe_name_fine));

        std::vector<unsigned int> renumbering_fine(fe_fine->dofs_per_cell);
        {
          AssertIndexRange(fe_fine->dofs_per_vertex, 2);
          renumbering_fine[0] = 0;
          for (unsigned int i = 0; i < fe_fine->dofs_per_line; ++i)
            renumbering_fine[i + fe_fine->dofs_per_vertex] =
              GeometryInfo<1>::vertices_per_cell * fe_fine->dofs_per_vertex + i;
          if (fe_fine->dofs_per_vertex > 0)
            renumbering_fine[fe_fine->dofs_per_cell -
                             fe_fine->dofs_per_vertex] =
              fe_fine->dofs_per_vertex;
        }



        AssertDimension(
          dof_handler_coarse.get_fe(fe_index_pair.first).n_base_elements(), 1);
        std::string fe_name_coarse =
          dof_handler_coarse.get_fe(fe_index_pair.first)
            .base_element(0)
            .get_name();
        {
          const std::size_t template_starts = fe_name_coarse.find_first_of('<');
          Assert(fe_name_coarse[template_starts + 1] ==
                   (dim == 1 ? '1' : (dim == 2 ? '2' : '3')),
                 ExcInternalError());
          fe_name_coarse[template_starts + 1] = '1';
        }
        const std::unique_ptr<FiniteElement<1>> fe_coarse(
          FETools::get_fe_by_name<1, 1>(fe_name_coarse));

        std::vector<unsigned int> renumbering_coarse(fe_coarse->dofs_per_cell);
        {
          AssertIndexRange(fe_coarse->dofs_per_vertex, 2);
          renumbering_coarse[0] = 0;
          for (unsigned int i = 0; i < fe_coarse->dofs_per_line; ++i)
            renumbering_coarse[i + fe_coarse->dofs_per_vertex] =
              GeometryInfo<1>::vertices_per_cell * fe_coarse->dofs_per_vertex +
              i;
          if (fe_coarse->dofs_per_vertex > 0)
            renumbering_coarse[fe_coarse->dofs_per_cell -
                               fe_coarse->dofs_per_vertex] =
              fe_coarse->dofs_per_vertex;
        }



        FullMatrix<double> matrix(fe_fine->dofs_per_cell,
                                  fe_coarse->dofs_per_cell);
        FETools::get_projection_matrix(*fe_coarse, *fe_fine, matrix);
        transfer.schemes[fe_index_no].prolongation_matrix_1d.resize(
          fe_fine->dofs_per_cell * fe_coarse->dofs_per_cell);

        for (unsigned int i = 0, k = 0; i < fe_coarse->dofs_per_cell; ++i)
          for (unsigned int j = 0; j < fe_fine->dofs_per_cell; ++j, ++k)
            transfer.schemes[fe_index_no].prolongation_matrix_1d[k] =
              matrix(renumbering_fine[j], renumbering_coarse[i]);
      }

    // -------------------------------- weights --------------------------------
    const bool fine_element_is_continuous = Utilities::MPI::max(
      static_cast<unsigned int>(
        transfer.schemes.size() > 0 ?
          transfer.schemes.front().fine_element_is_continuous :
          false),
      comm);
    if (fine_element_is_continuous)
      {
        for (auto &scheme : transfer.schemes)
          scheme.weights.resize(scheme.n_cells_coarse *
                                scheme.n_cell_dofs_fine);

        LinearAlgebra::distributed::Vector<Number> touch_count;
        touch_count.reinit(transfer.partitioner_fine);

        std::vector<unsigned int *> level_dof_indices_fine_(
          fe_index_pairs.size());
        std::vector<Number *> weights_(fe_index_pairs.size());

        for (unsigned int i = 0; i < fe_index_pairs.size(); i++)
          level_dof_indices_fine_[i] =
            &transfer.schemes[i].level_dof_indices_fine[0];

        process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
          const auto fe_pair_no =
            fe_index_pairs[std::pair<unsigned int, unsigned int>(
              cell_coarse->active_fe_index(), cell_fine->active_fe_index())];

          for (unsigned int i = 0;
               i < transfer.schemes[fe_pair_no].n_cell_dofs_fine;
               i++)
            if (constraint_fine.is_constrained(
                  transfer.partitioner_fine->local_to_global(
                    level_dof_indices_fine_[fe_pair_no][i])) == false)
              touch_count.local_element(
                level_dof_indices_fine_[fe_pair_no][i]) += 1;

          level_dof_indices_fine_[fe_pair_no] +=
            transfer.schemes[fe_pair_no].n_cell_dofs_fine;
        });

        touch_count.compress(VectorOperation::add);
        touch_count.update_ghost_values();

        for (unsigned int i = 0; i < fe_index_pairs.size(); i++)
          {
            level_dof_indices_fine_[i] =
              &transfer.schemes[i].level_dof_indices_fine[0];
            weights_[i] = &transfer.schemes[i].weights[0];
          }

        process_cells([&](const auto &cell_coarse, const auto &cell_fine) {
          const auto fe_pair_no =
            fe_index_pairs[std::pair<unsigned int, unsigned int>(
              cell_coarse->active_fe_index(), cell_fine->active_fe_index())];

          for (unsigned int i = 0;
               i < transfer.schemes[fe_pair_no].n_cell_dofs_fine;
               i++)
            if (constraint_fine.is_constrained(
                  transfer.partitioner_fine->local_to_global(
                    level_dof_indices_fine_[fe_pair_no][i])) == true)
              weights_[fe_pair_no][i] = Number(0.);
            else
              weights_[fe_pair_no][i] =
                Number(1.) / touch_count.local_element(
                               level_dof_indices_fine_[fe_pair_no][i]);

          level_dof_indices_fine_[fe_pair_no] +=
            transfer.schemes[fe_pair_no].n_cell_dofs_fine;
          weights_[fe_pair_no] += transfer.schemes[fe_pair_no].n_cell_dofs_fine;
        });
      }
  }

} // namespace MGTransferUtil

DEAL_II_NAMESPACE_CLOSE

#endif
