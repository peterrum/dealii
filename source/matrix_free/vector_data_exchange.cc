// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/base/mpi_consensus_algorithms.h>
#include <deal.II/base/timer.h>

#include <deal.II/matrix_free/vector_data_exchange.h>

#ifdef DEAL_II_WITH_64BIT_INDICES
#  include <deal.II/base/mpi_consensus_algorithms.templates.h>
#endif

#include <map>
#include <vector>


DEAL_II_NAMESPACE_OPEN

namespace internal
{
  namespace MatrixFreeFunctions
  {
    namespace VectorDataExchange
    {
      PartitionerWrapper::PartitionerWrapper(
        const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner)
        : partitioner(partitioner)
      {}



      unsigned int
      PartitionerWrapper::local_size() const
      {
        return partitioner->local_size();
      }



      unsigned int
      PartitionerWrapper::n_ghost_indices() const
      {
        return partitioner->n_ghost_indices();
      }



      unsigned int
      PartitionerWrapper::n_import_indices() const
      {
        return partitioner->n_import_indices();
      }



      unsigned int
      PartitionerWrapper::n_import_sm_procs() const
      {
        return 0;
      }



      types::global_dof_index
      PartitionerWrapper::size() const
      {
        return partitioner->size();
      }



      void
      PartitionerWrapper::export_to_ghosted_array_start(
        const unsigned int                          communication_channel,
        const ArrayView<const double> &             locally_owned_array,
        const std::vector<ArrayView<const double>> &shared_arrays,
        const ArrayView<double> &                   ghost_array,
        const ArrayView<double> &                   temporary_storage,
        std::vector<MPI_Request> &                  requests) const
      {
        (void)shared_arrays;
#ifndef DEAL_II_WITH_MPI
        (void)communication_channel;
        (void)locally_owned_array;
        (void)ghost_array;
        (void)temporary_storage;
        (void)requests;
#else
        partitioner->export_to_ghosted_array_start(communication_channel,
                                                   locally_owned_array,
                                                   temporary_storage,
                                                   ghost_array,
                                                   requests);
#endif
      }



      void
      PartitionerWrapper::export_to_ghosted_array_finish(
        const ArrayView<const double> &             locally_owned_array,
        const std::vector<ArrayView<const double>> &shared_arrays,
        const ArrayView<double> &                   ghost_array,
        std::vector<MPI_Request> &                  requests) const
      {
        (void)locally_owned_array;
        (void)shared_arrays;
#ifndef DEAL_II_WITH_MPI
        (void)ghost_array;
        (void)requests;
#else
        partitioner->export_to_ghosted_array_finish(ghost_array, requests);
#endif
      }



      void
      PartitionerWrapper::import_from_ghosted_array_start(
        const VectorOperation::values               vector_operation,
        const unsigned int                          communication_channel,
        const ArrayView<const double> &             locally_owned_array,
        const std::vector<ArrayView<const double>> &shared_arrays,
        const ArrayView<double> &                   ghost_array,
        const ArrayView<double> &                   temporary_storage,
        std::vector<MPI_Request> &                  requests) const
      {
        (void)locally_owned_array;
        (void)shared_arrays;
#ifndef DEAL_II_WITH_MPI
        (void)vector_operation;
        (void)communication_channel;
        (void)ghost_array;
        (void)temporary_storage;
        (void)requests;
#else
        partitioner->import_from_ghosted_array_start(vector_operation,
                                                     communication_channel,
                                                     ghost_array,
                                                     temporary_storage,
                                                     requests);
#endif
      }



      void
      PartitionerWrapper::import_from_ghosted_array_finish(
        const VectorOperation::values               vector_operation,
        const ArrayView<double> &                   locally_owned_storage,
        const std::vector<ArrayView<const double>> &shared_arrays,
        const ArrayView<double> &                   ghost_array,
        const ArrayView<const double> &             temporary_storage,
        std::vector<MPI_Request> &                  requests) const
      {
        (void)shared_arrays;
#ifndef DEAL_II_WITH_MPI
        (void)vector_operation;
        (void)locally_owned_storage;
        (void)ghost_array;
        (void)temporary_storage;
        (void)requests;
#else
        partitioner->import_from_ghosted_array_finish(vector_operation,
                                                      temporary_storage,
                                                      locally_owned_storage,
                                                      ghost_array,
                                                      requests);
#endif
      }



      void
      PartitionerWrapper::reset_ghost_values(
        const ArrayView<double> &ghost_array) const
      {
        reset_ghost_values_impl(ghost_array);
      }



      void
      PartitionerWrapper::export_to_ghosted_array_start(
        const unsigned int                         communication_channel,
        const ArrayView<const float> &             locally_owned_array,
        const std::vector<ArrayView<const float>> &shared_arrays,
        const ArrayView<float> &                   ghost_array,
        const ArrayView<float> &                   temporary_storage,
        std::vector<MPI_Request> &                 requests) const
      {
        (void)shared_arrays;
#ifndef DEAL_II_WITH_MPI
        (void)communication_channel;
        (void)locally_owned_array;
        (void)ghost_array;
        (void)temporary_storage;
        (void)requests;
#else
        partitioner->export_to_ghosted_array_start(communication_channel,
                                                   locally_owned_array,
                                                   temporary_storage,
                                                   ghost_array,
                                                   requests);
#endif
      }



      void
      PartitionerWrapper::export_to_ghosted_array_finish(
        const ArrayView<const float> &             locally_owned_array,
        const std::vector<ArrayView<const float>> &shared_arrays,
        const ArrayView<float> &                   ghost_array,
        std::vector<MPI_Request> &                 requests) const
      {
        (void)locally_owned_array;
        (void)shared_arrays;
#ifndef DEAL_II_WITH_MPI
        (void)ghost_array;
        (void)requests;
#else
        partitioner->export_to_ghosted_array_finish(ghost_array, requests);
#endif
      }



      void
      PartitionerWrapper::import_from_ghosted_array_start(
        const VectorOperation::values              vector_operation,
        const unsigned int                         communication_channel,
        const ArrayView<const float> &             locally_owned_array,
        const std::vector<ArrayView<const float>> &shared_arrays,
        const ArrayView<float> &                   ghost_array,
        const ArrayView<float> &                   temporary_storage,
        std::vector<MPI_Request> &                 requests) const
      {
        (void)locally_owned_array;
        (void)shared_arrays;
#ifndef DEAL_II_WITH_MPI
        (void)vector_operation;
        (void)communication_channel;
        (void)ghost_array;
        (void)temporary_storage;
        (void)requests;
#else
        partitioner->import_from_ghosted_array_start(vector_operation,
                                                     communication_channel,
                                                     ghost_array,
                                                     temporary_storage,
                                                     requests);
#endif
      }



      void
      PartitionerWrapper::import_from_ghosted_array_finish(
        const VectorOperation::values              vector_operation,
        const ArrayView<float> &                   locally_owned_storage,
        const std::vector<ArrayView<const float>> &shared_arrays,
        const ArrayView<float> &                   ghost_array,
        const ArrayView<const float> &             temporary_storage,
        std::vector<MPI_Request> &                 requests) const
      {
        (void)shared_arrays;
#ifndef DEAL_II_WITH_MPI
        (void)vector_operation;
        (void)locally_owned_storage;
        (void)ghost_array;
        (void)temporary_storage;
        (void)requests;
#else
        partitioner->import_from_ghosted_array_finish(vector_operation,
                                                      temporary_storage,
                                                      locally_owned_storage,
                                                      ghost_array,
                                                      requests);
#endif
      }



      void
      PartitionerWrapper::reset_ghost_values(
        const ArrayView<float> &ghost_array) const
      {
        reset_ghost_values_impl(ghost_array);
      }



      template <typename Number>
      void
      PartitionerWrapper::reset_ghost_values_impl(
        const ArrayView<Number> &ghost_array) const
      {
        for (const auto &my_ghosts :
             partitioner->ghost_indices_within_larger_ghost_set())
          for (unsigned int j = my_ghosts.first; j < my_ghosts.second; ++j)
            ghost_array[j] = 0.;
      }



      namespace internal
      {
        void
        compress(std::vector<unsigned int> &sm_export_ptr,
                 std::vector<unsigned int> &sm_export_indices,
                 std::vector<unsigned int> &sm_export_len)
        {
          std::vector<unsigned int> recv_ptr = {0};
          std::vector<unsigned int> recv_indices;
          std::vector<unsigned int> recv_len;

          for (unsigned int i = 0; i + 1 < sm_export_ptr.size(); i++)
            {
              if (sm_export_ptr[i] != sm_export_ptr[i + 1])
                {
                  recv_indices.push_back(sm_export_indices[sm_export_ptr[i]]);
                  recv_len.push_back(1);

                  for (unsigned int j = sm_export_ptr[i] + 1;
                       j < sm_export_ptr[i + 1];
                       j++)
                    if (recv_indices.back() + recv_len.back() !=
                        sm_export_indices[j])
                      {
                        recv_indices.push_back(sm_export_indices[j]);
                        recv_len.push_back(1);
                      }
                    else
                      recv_len.back()++;
                }
              recv_ptr.push_back(recv_indices.size());
            }

          sm_export_ptr = recv_ptr;
          sm_export_ptr.shrink_to_fit();
          sm_export_indices = recv_indices;
          sm_export_indices.shrink_to_fit();
          sm_export_len = recv_len;
          sm_export_len.shrink_to_fit();
        }

        std::vector<unsigned int>
        procs_of_sm(const MPI_Comm &comm, const MPI_Comm &comm_shared)
        {
          std::vector<unsigned int> ranks_shared;

#ifndef DEAL_II_WITH_MPI
          Assert(false, ExcNeedsMPI());
          (void)comm;
          (void)comm_shared;
#else
          // extract information from comm
          int rank_;
          MPI_Comm_rank(comm, &rank_);

          const unsigned int rank = rank_;

          // extract information from sm-comm
          int size_shared;
          MPI_Comm_size(comm_shared, &size_shared);

          // gather ranks
          ranks_shared.resize(size_shared);
          MPI_Allgather(&rank,
                        1,
                        MPI_UNSIGNED,
                        ranks_shared.data(),
                        1,
                        MPI_INT,
                        comm_shared);
#endif

          return ranks_shared;
        }

      } // namespace internal



      Full::Full(
        const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
        const MPI_Comm &communicator_sm)
      {
        const IndexSet &is_locally_owned = partitioner->locally_owned_range();
        const IndexSet &is_locally_ghost = partitioner->ghost_indices();
        const MPI_Comm &communicator     = partitioner->get_mpi_communicator();
        const std::vector<std::pair<unsigned int, unsigned int>>
          ghost_indices_within_larger_ghost_set =
            partitioner->ghost_indices_within_larger_ghost_set();

        this->comm    = communicator;
        this->comm_sm = communicator_sm;

#ifndef DEAL_II_WITH_MPI
        Assert(false, ExcNeedsMPI());

        (void)is_locally_owned;
        (void)is_locally_ghost;
        (void)ghost_indices_within_larger_ghost_set;
#else
        this->n_local_elements  = is_locally_owned.n_elements();
        this->n_ghost_elements  = is_locally_ghost.n_elements();
        this->n_global_elements = is_locally_owned.size();

        sm_export_data.first = {0};
        sm_import_data.first = {0};


        std::vector<unsigned int> sm_export_indices_;
        std::vector<unsigned int> sm_export_len_;

        std::vector<unsigned int> sm_import_indices;
        std::vector<unsigned int> sm_import_len;

        if (Utilities::MPI::job_supports_mpi() == false)
          return; // nothing to do in serial case

        std::vector<unsigned int> sm_ranks(
          Utilities::MPI::n_mpi_processes(comm_sm));

        const unsigned int rank = Utilities::MPI::this_mpi_process(comm);

        MPI_Allgather(
          &rank, 1, MPI_UNSIGNED, sm_ranks.data(), 1, MPI_UNSIGNED, comm_sm);

        std::vector<unsigned int> owning_ranks_of_ghosts(
          is_locally_ghost.n_elements());

        Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload
          process(is_locally_owned,
                  is_locally_ghost,
                  comm,
                  owning_ranks_of_ghosts,
                  true);

        Utilities::MPI::ConsensusAlgorithms::Selector<
          std::pair<types::global_dof_index, types::global_dof_index>,
          unsigned int>
          consensus_algorithm(process, comm);
        consensus_algorithm.run();

        std::vector<unsigned int> send_remote_indices; // import_indices_data
        std::vector<unsigned int> send_remote_len;     // import_indices_data

        std::vector<MPI_Request> sm_export_req;
        std::vector<MPI_Request> sm_import_req;

        std::vector<unsigned int> sm_export_indices;
        std::vector<unsigned int> sm_export_len;

        std::vector<unsigned int> sm_import_indices_;
        std::vector<unsigned int> sm_import_len_;

        std::vector<unsigned int> shifts_indices; //
        std::vector<unsigned int> shifts_len;     //

        for (const auto &pair : ghost_indices_within_larger_ghost_set)
          for (unsigned int k = pair.first; k < pair.second; ++k)
            shifts_indices.push_back(k);

        std::vector<unsigned int> shifts_ptr_ = {0}; //
        std::vector<unsigned int> shifts_indices_;   //
        std::vector<unsigned int> shifts_len_;       //

        {
          std::map<unsigned int, std::vector<types::global_dof_index>>
            rank_to_local_indices;

          for (unsigned int i = 0; i < owning_ranks_of_ghosts.size(); i++)
            rank_to_local_indices[owning_ranks_of_ghosts[i]].push_back(i);

          unsigned int offset = 0;

          sm_export_data_this.first = {0};

          for (const auto &rank_and_local_indices : rank_to_local_indices)
            {
              const auto ptr = std::find(sm_ranks.begin(),
                                         sm_ranks.end(),
                                         rank_and_local_indices.first);

              if (ptr == sm_ranks.end())
                {
                  // remote process
                  ghost_targets_data.emplace_back(
                    rank_and_local_indices.first,
                    std::pair<unsigned int, unsigned int>{
                      shifts_indices[offset],
                      rank_and_local_indices.second.size()});

                  for (unsigned int i = 0;
                       i < rank_and_local_indices.second.size();
                       ++i)
                    shifts_indices_.push_back(shifts_indices[i + offset]);
                  shifts_ptr_.push_back(shifts_indices_.size());

                  ghost_indices_subset_data.first.push_back(offset);

                  unsigned int i =
                    n_ghost_indices_in_larger_set_by_remote_rank.size();

                  n_ghost_indices_in_larger_set_by_remote_rank.push_back(
                    (shifts_indices[ghost_indices_subset_data.first[i] +
                                    (ghost_targets_data[i].second.second - 1)] -
                     shifts_indices[ghost_indices_subset_data.first[i]]) +
                    1);
                }
              else
                {
                  // shared process
                  sm_ghost_ranks.push_back(
                    std::distance(sm_ranks.begin(), ptr));
                  sm_export_data.first.push_back(
                    sm_export_data.first.back() +
                    rank_and_local_indices.second.size());

                  for (unsigned int i = offset, c = 0;
                       c < rank_and_local_indices.second.size();
                       ++c, ++i)
                    sm_export_indices_.push_back(shifts_indices[i] +
                                                 is_locally_owned.n_elements());
                  sm_export_data_this.first.push_back(
                    sm_export_indices_.size());
                }
              offset += rank_and_local_indices.second.size();
            }
          sm_export_req.resize(sm_ghost_ranks.size());

          sm_export_indices.resize(sm_export_data.first.back());
        }

        {
          const auto rank_to_global_indices = process.get_requesters();

          import_indices_data.first = {0};

          for (const auto &rank_and_global_indices : rank_to_global_indices)
            {
              const auto ptr = std::find(sm_ranks.begin(),
                                         sm_ranks.end(),
                                         rank_and_global_indices.first);

              if (ptr == sm_ranks.end())
                {
                  // remote process
                  const unsigned int prev = send_remote_indices.size();

                  for (const auto &i : rank_and_global_indices.second)
                    send_remote_indices.push_back(
                      is_locally_owned.index_within_set(i));

                  import_targets_data.emplace_back(
                    rank_and_global_indices.first,
                    std::pair<unsigned int, unsigned int>{
                      prev, send_remote_indices.size() - prev});

                  import_indices_data.first.push_back(
                    send_remote_indices.size());
                }
              else
                {
                  // shared process
                  sm_import_ranks.push_back(
                    std::distance(sm_ranks.begin(), ptr));

                  for (const auto &i : rank_and_global_indices.second)
                    sm_import_indices.push_back(
                      is_locally_owned.index_within_set(i));

                  sm_import_data.first.push_back(sm_import_indices.size());
                }
            }
          sm_import_req.resize(sm_import_ranks.size());

          sm_import_data_this.first = sm_import_data.first;
          sm_import_indices_.resize(sm_import_data.first.back());
        }


        {
          for (unsigned int i = 0; i < sm_ghost_ranks.size(); i++)
            MPI_Isend(sm_export_indices_.data() + sm_export_data_this.first[i],
                      sm_export_data_this.first[i + 1] -
                        sm_export_data_this.first[i],
                      MPI_UNSIGNED,
                      sm_ghost_ranks[i],
                      4,
                      comm_sm,
                      sm_export_req.data() + i);

          for (unsigned int i = 0; i < sm_import_ranks.size(); i++)
            MPI_Irecv(sm_import_indices_.data() + sm_import_data_this.first[i],
                      sm_import_data_this.first[i + 1] -
                        sm_import_data_this.first[i],
                      MPI_UNSIGNED,
                      sm_import_ranks[i],
                      4,
                      comm_sm,
                      sm_import_req.data() + i);

          MPI_Waitall(sm_export_req.size(),
                      sm_export_req.data(),
                      MPI_STATUSES_IGNORE);
          MPI_Waitall(sm_import_req.size(),
                      sm_import_req.data(),
                      MPI_STATUSES_IGNORE);
        }

        {
          for (unsigned int i = 0; i < sm_import_ranks.size(); i++)
            MPI_Isend(sm_import_indices.data() + sm_import_data.first[i],
                      sm_import_data.first[i + 1] - sm_import_data.first[i],
                      MPI_UNSIGNED,
                      sm_import_ranks[i],
                      2,
                      comm_sm,
                      sm_import_req.data() + i);

          for (unsigned int i = 0; i < sm_ghost_ranks.size(); i++)
            MPI_Irecv(sm_export_indices.data() + sm_export_data.first[i],
                      sm_export_data.first[i + 1] - sm_export_data.first[i],
                      MPI_UNSIGNED,
                      sm_ghost_ranks[i],
                      2,
                      comm_sm,
                      sm_export_req.data() + i);

          MPI_Waitall(sm_export_req.size(),
                      sm_export_req.data(),
                      MPI_STATUSES_IGNORE);
          MPI_Waitall(sm_import_req.size(),
                      sm_import_req.data(),
                      MPI_STATUSES_IGNORE);
        }

        internal::compress(sm_export_data.first,
                           sm_export_indices,
                           sm_export_len);

        internal::compress(import_indices_data.first,
                           send_remote_indices,
                           send_remote_len);

        internal::compress(sm_import_data.first,
                           sm_import_indices,
                           sm_import_len);
        internal::compress(sm_export_data_this.first,
                           sm_export_indices_,
                           sm_export_len_);
        internal::compress(sm_import_data_this.first,
                           sm_import_indices_,
                           sm_import_len_);

        for (unsigned int i = 0; i < send_remote_len.size(); ++i)
          import_indices_data.second.emplace_back(send_remote_indices[i],
                                                  send_remote_len[i]);

        for (unsigned int i = 0; i < sm_export_indices.size(); ++i)
          sm_export_data.second.emplace_back(sm_export_indices[i],
                                             sm_export_len[i]);

        for (unsigned int i = 0; i < sm_export_indices_.size(); ++i)
          sm_export_data_this.second.emplace_back(sm_export_indices_[i],
                                                  sm_export_len_[i]);

        for (unsigned int i = 0; i < sm_import_indices.size(); ++i)
          sm_import_data.second.emplace_back(sm_import_indices[i],
                                             sm_import_len[i]);

        for (unsigned int i = 0; i < sm_import_indices_.size(); ++i)
          sm_import_data_this.second.emplace_back(sm_import_indices_[i],
                                                  sm_import_len_[i]);

        ghost_indices_subset_data.first = shifts_ptr_;

        internal::compress(ghost_indices_subset_data.first,
                           shifts_indices_,
                           shifts_len_);

        for (unsigned int i = 0; i < shifts_indices_.size(); ++i)
          ghost_indices_subset_data.second.emplace_back(shifts_indices_[i],
                                                        shifts_len_[i]);

#endif
      }



      void
      Full::export_to_ghosted_array_start(
        const unsigned int                          communication_channel,
        const ArrayView<const double> &             locally_owned_array,
        const std::vector<ArrayView<const double>> &shared_arrays,
        const ArrayView<double> &                   ghost_array,
        const ArrayView<double> &                   temporary_storage,
        std::vector<MPI_Request> &                  requests) const
      {
        export_to_ghosted_array_start_impl(communication_channel,
                                           locally_owned_array,
                                           shared_arrays,
                                           ghost_array,
                                           temporary_storage,
                                           requests);
      }



      void
      Full::export_to_ghosted_array_finish(
        const ArrayView<const double> &             locally_owned_array,
        const std::vector<ArrayView<const double>> &shared_arrays,
        const ArrayView<double> &                   ghost_array,
        std::vector<MPI_Request> &                  requests) const
      {
        export_to_ghosted_array_finish_impl(locally_owned_array,
                                            shared_arrays,
                                            ghost_array,
                                            requests);
      }



      void
      Full::import_from_ghosted_array_start(
        const VectorOperation::values               vector_operation,
        const unsigned int                          communication_channel,
        const ArrayView<const double> &             locally_owned_array,
        const std::vector<ArrayView<const double>> &shared_arrays,
        const ArrayView<double> &                   ghost_array,
        const ArrayView<double> &                   temporary_storage,
        std::vector<MPI_Request> &                  requests) const
      {
        import_from_ghosted_array_start_impl(vector_operation,
                                             communication_channel,
                                             locally_owned_array,
                                             shared_arrays,
                                             ghost_array,
                                             temporary_storage,
                                             requests);
      }



      void
      Full::import_from_ghosted_array_finish(
        const VectorOperation::values               vector_operation,
        const ArrayView<double> &                   locally_owned_storage,
        const std::vector<ArrayView<const double>> &shared_arrays,
        const ArrayView<double> &                   ghost_array,
        const ArrayView<const double> &             temporary_storage,
        std::vector<MPI_Request> &                  requests) const
      {
        import_from_ghosted_array_finish_impl(vector_operation,
                                              locally_owned_storage,
                                              shared_arrays,
                                              ghost_array,
                                              temporary_storage,
                                              requests);
      }



      void
      Full::export_to_ghosted_array_start(
        const unsigned int                         communication_channel,
        const ArrayView<const float> &             locally_owned_array,
        const std::vector<ArrayView<const float>> &shared_arrays,
        const ArrayView<float> &                   ghost_array,
        const ArrayView<float> &                   temporary_storage,
        std::vector<MPI_Request> &                 requests) const
      {
        export_to_ghosted_array_start_impl(communication_channel,
                                           locally_owned_array,
                                           shared_arrays,
                                           ghost_array,
                                           temporary_storage,
                                           requests);
      }



      void
      Full::export_to_ghosted_array_finish(
        const ArrayView<const float> &             locally_owned_array,
        const std::vector<ArrayView<const float>> &shared_arrays,
        const ArrayView<float> &                   ghost_array,
        std::vector<MPI_Request> &                 requests) const
      {
        export_to_ghosted_array_finish_impl(locally_owned_array,
                                            shared_arrays,
                                            ghost_array,
                                            requests);
      }



      void
      Full::import_from_ghosted_array_start(
        const VectorOperation::values              vector_operation,
        const unsigned int                         communication_channel,
        const ArrayView<const float> &             locally_owned_array,
        const std::vector<ArrayView<const float>> &shared_arrays,
        const ArrayView<float> &                   ghost_array,
        const ArrayView<float> &                   temporary_storage,
        std::vector<MPI_Request> &                 requests) const
      {
        import_from_ghosted_array_start_impl(vector_operation,
                                             communication_channel,
                                             locally_owned_array,
                                             shared_arrays,
                                             ghost_array,
                                             temporary_storage,
                                             requests);
      }



      void
      Full::import_from_ghosted_array_finish(
        const VectorOperation::values              vector_operation,
        const ArrayView<float> &                   locally_owned_storage,
        const std::vector<ArrayView<const float>> &shared_arrays,
        const ArrayView<float> &                   ghost_array,
        const ArrayView<const float> &             temporary_storage,
        std::vector<MPI_Request> &                 requests) const
      {
        import_from_ghosted_array_finish_impl(vector_operation,
                                              locally_owned_storage,
                                              shared_arrays,
                                              ghost_array,
                                              temporary_storage,
                                              requests);
      }



      template <typename Number>
      void
      Full::export_to_ghosted_array_start_impl(
        const unsigned int                          communication_channel,
        const ArrayView<const Number> &             data_this,
        const std::vector<ArrayView<const Number>> &data_others,
        const ArrayView<Number> &                   buffer,
        const ArrayView<Number> &                   temporary_storage,
        std::vector<MPI_Request> &                  requests) const
      {
#ifndef DEAL_II_WITH_MPI
        Assert(false, ExcNeedsMPI());

        (void)communication_channel;
        (void)data_this;
        (void)data_others;
        (void)buffer;
        (void)temporary_storage;
        (void)requests;
#else
        (void)data_others;

        requests.resize(sm_import_ranks.size() + sm_ghost_ranks.size() +
                        ghost_targets_data.size() + import_targets_data.size());

        int dummy;
        // receive a signal that relevant sm neighbors are ready
        for (unsigned int i = 0; i < sm_ghost_ranks.size(); i++)
          MPI_Irecv(&dummy,
                    0,
                    MPI_INT,
                    sm_ghost_ranks[i],
                    communication_channel + 2,
                    comm_sm,
                    requests.data() + sm_import_ranks.size() + i);

        // signal to all relevant sm neighbors that this process is ready
        for (unsigned int i = 0; i < sm_import_ranks.size(); i++)
          MPI_Isend(&dummy,
                    0,
                    MPI_INT,
                    sm_import_ranks[i],
                    communication_channel + 2,
                    comm_sm,
                    requests.data() + i);

        // receive data from remote processes
        for (unsigned int i = 0; i < ghost_targets_data.size(); i++)
          {
            const unsigned int offset =
              n_ghost_indices_in_larger_set_by_remote_rank[i] -
              ghost_targets_data[i].second.second;

            MPI_Irecv(buffer.data() + ghost_targets_data[i].second.first +
                        offset,
                      ghost_targets_data[i].second.second,
                      Utilities::MPI::internal::mpi_type_id(buffer.data()),
                      ghost_targets_data[i].first,
                      communication_channel + 3,
                      comm,
                      requests.data() + sm_import_ranks.size() +
                        sm_ghost_ranks.size() + i);
          }

        // send data to remote processes
        for (unsigned int i = 0, k = 0; i < import_targets_data.size(); i++)
          {
            for (unsigned int j = import_indices_data.first[i];
                 j < import_indices_data.first[i + 1];
                 j++)
              for (unsigned int l = 0; l < import_indices_data.second[j].second;
                   l++, k++)
                temporary_storage[k] =
                  data_this[import_indices_data.second[j].first + l];

            // send data away
            MPI_Isend(temporary_storage.data() +
                        import_targets_data[i].second.first,
                      import_targets_data[i].second.second,
                      Utilities::MPI::internal::mpi_type_id(data_this.data()),
                      import_targets_data[i].first,
                      communication_channel + 3,
                      comm,
                      requests.data() + sm_import_ranks.size() +
                        sm_ghost_ranks.size() + ghost_targets_data.size() + i);
          }
#endif
      }



      template <typename Number>
      void
      Full::export_to_ghosted_array_finish_impl(
        const ArrayView<const Number> &             data_this,
        const std::vector<ArrayView<const Number>> &data_others,
        const ArrayView<Number> &                   ghost_array,
        std::vector<MPI_Request> &                  requests) const
      {
        (void)data_this;

#ifndef DEAL_II_WITH_MPI
        Assert(false, ExcNeedsMPI());

        (void)data_others;
        (void)ghost_array;
        (void)requests;
#else

        AssertDimension(requests.size(),
                        sm_import_ranks.size() + sm_ghost_ranks.size() +
                          ghost_targets_data.size() +
                          import_targets_data.size());

        const auto split =
          [&](const unsigned int i) -> std::pair<unsigned int, unsigned int> {
          AssertIndexRange(i,
                           (sm_ghost_ranks.size() + ghost_targets_data.size()));

          if (i < sm_ghost_ranks.size())
            return {0, i};
          else
            return {1, i - sm_ghost_ranks.size()};
        };

        for (unsigned int c = 0;
             c < sm_ghost_ranks.size() + ghost_targets_data.size();
             c++)
          {
            int i;
            MPI_Waitany(sm_ghost_ranks.size() + ghost_targets_data.size(),
                        requests.data() + sm_import_ranks.size(),
                        &i,
                        MPI_STATUS_IGNORE);

            const auto s = split(i);
            i            = s.second;

            if (s.first == 0)
              {
                const Number *__restrict__ data_others_ptr =
                  data_others[sm_ghost_ranks[i]].data();
                Number *__restrict__ data_this_ptr = ghost_array.data();

                for (unsigned int lo = sm_export_data.first[i],
                                  ko = sm_export_data_this.first[i],
                                  li = 0,
                                  ki = 0;
                     (lo < sm_export_data.first[i + 1]) &&
                     (ko < sm_export_data_this.first[i + 1]);)
                  {
                    for (; (li < sm_export_data.second[lo].second) &&
                           (ki < sm_export_data_this.second[ko].second);
                         ++li, ++ki)
                      data_this_ptr[sm_export_data_this.second[ko].first + ki -
                                    n_local_elements] =
                        data_others_ptr[sm_export_data.second[lo].first + li];

                    if (li == sm_export_data.second[lo].second)
                      {
                        lo++;   // increment outer counter
                        li = 0; // reset inner counter
                      }

                    if (ki == sm_export_data_this.second[ko].second)
                      {
                        ko++;   // increment outer counter
                        ki = 0; // reset inner counter
                      }
                  }
              }
            else /*if(s.second == 1)*/
              {
                const unsigned int offset =
                  n_ghost_indices_in_larger_set_by_remote_rank[i] -
                  ghost_targets_data[i].second.second;

                for (unsigned int c  = 0,
                                  ko = ghost_indices_subset_data.first[i],
                                  ki = 0;
                     c < ghost_targets_data[i].second.second;
                     ++c)
                  {
                    AssertIndexRange(ko,
                                     ghost_indices_subset_data.second.size());

                    const unsigned int idx_1 =
                      ghost_indices_subset_data.second[ko].first + ki;
                    const unsigned int idx_2 =
                      ghost_targets_data[i].second.first + c + offset;

                    AssertIndexRange(idx_1, ghost_array.size());
                    AssertIndexRange(idx_2, ghost_array.size());

                    if (idx_1 == idx_2)
                      {
                        // noting to do
                      }
                    else if (idx_1 < idx_2)
                      {
                        ghost_array[idx_1] = ghost_array[idx_2];
                        ghost_array[idx_2] = 0.0;
                      }
                    else
                      {
                        Assert(false, ExcNotImplemented());
                      }

                    ++ki;

                    if (ki == ghost_indices_subset_data.second[ko].second)
                      {
                        ko++;   // increment outer counter
                        ki = 0; // reset inner counter
                      }
                  }
              }
          }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
#endif
      }



      template <typename Number>
      void
      Full::import_from_ghosted_array_start_impl(
        const VectorOperation::values               operation,
        const unsigned int                          communication_channel,
        const ArrayView<const Number> &             data_this,
        const std::vector<ArrayView<const Number>> &data_others,
        const ArrayView<Number> &                   buffer,
        const ArrayView<Number> &                   temporary_storage,
        std::vector<MPI_Request> &                  requests) const
      {
        (void)data_this;

#ifndef DEAL_II_WITH_MPI
        Assert(false, ExcNeedsMPI());

        (void)operation;
        (void)communication_channel;
        (void)data_others;
        (void)buffer;
        (void)temporary_storage;
        (void)requests;
#else
        // return;

        (void)data_others;
        (void)operation;

        Assert(operation == dealii::VectorOperation::add, ExcNotImplemented());

        requests.resize(sm_ghost_ranks.size() + sm_import_ranks.size() +
                        ghost_targets_data.size() + import_targets_data.size());

        int dummy;
        for (unsigned int i = 0; i < sm_ghost_ranks.size(); i++)
          MPI_Isend(&dummy,
                    0,
                    MPI_INT,
                    sm_ghost_ranks[i],
                    communication_channel + 1,
                    comm_sm,
                    requests.data() + i);

        for (unsigned int i = 0; i < sm_import_ranks.size(); i++)
          MPI_Irecv(&dummy,
                    0,
                    MPI_INT,
                    sm_import_ranks[i],
                    communication_channel + 1,
                    comm_sm,
                    requests.data() + sm_ghost_ranks.size() + i);

        for (unsigned int i = 0; i < ghost_targets_data.size(); i++)
          {
            for (unsigned int c  = 0,
                              ko = ghost_indices_subset_data.first[i],
                              ki = 0;
                 c < ghost_targets_data[i].second.second;
                 ++c)
              {
                AssertIndexRange(ko, ghost_indices_subset_data.second.size());

                const unsigned int idx_1 =
                  ghost_indices_subset_data.second[ko].first + ki;
                const unsigned int idx_2 =
                  ghost_targets_data[i].second.first + c;

                AssertIndexRange(idx_1, buffer.size());
                AssertIndexRange(idx_2, buffer.size());

                if (idx_1 == idx_2)
                  {
                    // nothing to do
                  }
                else if (idx_2 < idx_1)
                  {
                    buffer[idx_2] = buffer[idx_1];
                    buffer[idx_1] = 0.0;
                  }
                else
                  {
                    Assert(false, ExcNotImplemented());
                  }

                if (++ki == ghost_indices_subset_data.second[ko].second)
                  {
                    ko++;   // increment outer counter
                    ki = 0; // reset inner counter
                  }
              }

            MPI_Isend(buffer.data() + ghost_targets_data[i].second.first,
                      ghost_targets_data[i].second.second,
                      Utilities::MPI::internal::mpi_type_id(buffer.data()),
                      ghost_targets_data[i].first,
                      communication_channel + 0,
                      comm,
                      requests.data() + sm_ghost_ranks.size() +
                        sm_import_ranks.size() + i);
          }

        for (unsigned int i = 0; i < import_targets_data.size(); i++)
          MPI_Irecv(
            temporary_storage.data() + import_targets_data[i].second.first,
            import_targets_data[i].second.second,
            Utilities::MPI::internal::mpi_type_id(temporary_storage.data()),
            import_targets_data[i].first,
            communication_channel + 0,
            comm,
            requests.data() + sm_ghost_ranks.size() + sm_import_ranks.size() +
              ghost_targets_data.size() + i);
#endif
      }



      template <typename Number>
      void
      Full::import_from_ghosted_array_finish_impl(
        const VectorOperation::values               operation,
        const ArrayView<Number> &                   data_this,
        const std::vector<ArrayView<const Number>> &data_others,
        const ArrayView<Number> &                   buffer,
        const ArrayView<const Number> &             temporary_storage,
        std::vector<MPI_Request> &                  requests) const
      {
#ifndef DEAL_II_WITH_MPI
        Assert(false, ExcNeedsMPI());

        (void)operation;
        (void)data_this;
        (void)data_others;
        (void)buffer;
        (void)temporary_storage;
        (void)requests;
#else

        (void)operation;

        Assert(operation == dealii::VectorOperation::add, ExcNotImplemented());

        AssertDimension(requests.size(),
                        sm_ghost_ranks.size() + sm_import_ranks.size() +
                          ghost_targets_data.size() +
                          import_targets_data.size());

        const auto split =
          [&](const unsigned int i) -> std::pair<unsigned int, unsigned int> {
          AssertIndexRange(i,
                           (sm_import_ranks.size() + ghost_targets_data.size() +
                            import_targets_data.size()));

          if (i < sm_import_ranks.size())
            return {0, i};
          else if (i < (sm_import_ranks.size() + ghost_targets_data.size()))
            return {2, i - sm_import_ranks.size()};
          else
            return {1, i - sm_import_ranks.size() - ghost_targets_data.size()};
        };

        for (unsigned int c = 0;
             c < sm_import_ranks.size() + import_targets_data.size() +
                   ghost_targets_data.size();
             c++)
          {
            int i;
            MPI_Waitany(sm_import_ranks.size() + import_targets_data.size() +
                          ghost_targets_data.size(),
                        requests.data() + sm_ghost_ranks.size(),
                        &i,
                        MPI_STATUS_IGNORE);

            const auto &s = split(i);
            i             = s.second;

            if (s.first == 0)
              {
                Number *__restrict__ data_others_ptr =
                  const_cast<Number *>(data_others[sm_import_ranks[i]].data());
                Number *__restrict__ data_this_ptr = data_this.data();

                for (unsigned int lo = sm_import_data.first[i],
                                  ko = sm_import_data_this.first[i],
                                  li = 0,
                                  ki = 0;
                     (lo < sm_import_data.first[i + 1]) &&
                     (ko < sm_import_data_this.first[i + 1]);)
                  {
                    for (; (li < sm_import_data.second[lo].second) &&
                           (ki < sm_import_data_this.second[ko].second);
                         ++li, ++ki)
                      {
                        data_this_ptr[sm_import_data.second[lo].first + li] +=
                          data_others_ptr[sm_import_data_this.second[ko].first +
                                          ki];
                        data_others_ptr[sm_import_data_this.second[ko].first +
                                        ki] = 0.0;
                      }

                    if (li == sm_import_data.second[lo].second)
                      {
                        lo++;   // increment outer counter
                        li = 0; // reset inner counter
                      }
                    if (ki == sm_import_data_this.second[ko].second)
                      {
                        ko++;   // increment outer counter
                        ki = 0; // reset inner counter
                      }
                  }
              }
            else if (s.first == 1)
              {
                for (unsigned int j = import_indices_data.first[i],
                                  k = import_targets_data[i].second.first;
                     j < import_indices_data.first[i + 1];
                     j++)
                  for (unsigned int l = 0;
                       l < import_indices_data.second[j].second;
                       l++)
                    data_this[import_indices_data.second[j].first + l] +=
                      temporary_storage[k++];
              }
            else /*if (s.first == 2)*/
              {
                std::memset(buffer.data() + ghost_targets_data[i].second.first,
                            0.0,
                            (ghost_targets_data[i].second.second) *
                              sizeof(Number));
              }
          }

        MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
#endif
      }



      unsigned int
      Full::local_size() const
      {
        return n_local_elements;
      }



      unsigned int
      Full::n_ghost_indices() const
      {
        return n_ghost_elements;
      }



      unsigned int
      Full::n_import_indices() const
      {
        if (import_targets_data.size() == 0)
          return 0;
        return import_targets_data.back().second.first +
               import_targets_data.back().second.second;
      }



      unsigned int
      Full::n_import_sm_procs() const
      {
        return sm_import_ranks.size() + sm_ghost_ranks.size(); // TODO
      }



      types::global_dof_index
      Full::size() const
      {
        return n_global_elements;
      }



      void
      Full::reset_ghost_values(const ArrayView<double> &ghost_array) const
      {
        reset_ghost_values_impl(ghost_array);
      }



      void
      Full::reset_ghost_values(const ArrayView<float> &ghost_array) const
      {
        reset_ghost_values_impl(ghost_array);
      }



      template <typename Number>
      void
      Full::reset_ghost_values_impl(const ArrayView<Number> &ghost_array) const
      {
        // TODO
        std::memset(ghost_array.data(),
                    0.0,
                    ghost_array.size() * sizeof(Number));
      }



    } // namespace VectorDataExchange
  }   // namespace MatrixFreeFunctions
} // namespace internal


DEAL_II_NAMESPACE_CLOSE
