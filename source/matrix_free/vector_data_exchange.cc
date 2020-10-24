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
        compress(std::vector<unsigned int> &recv_sm_ptr,
                 std::vector<unsigned int> &recv_sm_indices,
                 std::vector<unsigned int> &recv_sm_len)
        {
          std::vector<unsigned int> recv_ptr = {0};
          std::vector<unsigned int> recv_indices;
          std::vector<unsigned int> recv_len;

          for (unsigned int i = 0; i + 1 < recv_sm_ptr.size(); i++)
            {
              if (recv_sm_ptr[i] != recv_sm_ptr[i + 1])
                {
                  recv_indices.push_back(recv_sm_indices[recv_sm_ptr[i]]);
                  recv_len.push_back(1);

                  for (unsigned int j = recv_sm_ptr[i] + 1;
                       j < recv_sm_ptr[i + 1];
                       j++)
                    if (recv_indices.back() + recv_len.back() !=
                        recv_sm_indices[j])
                      {
                        recv_indices.push_back(recv_sm_indices[j]);
                        recv_len.push_back(1);
                      }
                    else
                      recv_len.back()++;
                }
              recv_ptr.push_back(recv_indices.size());
            }

          recv_sm_ptr = recv_ptr;
          recv_sm_ptr.shrink_to_fit();
          recv_sm_indices = recv_indices;
          recv_sm_indices.shrink_to_fit();
          recv_sm_len = recv_len;
          recv_sm_len.shrink_to_fit();
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
        : recv_sm_ptr{0}
        , send_remote_ptr{0}
        , send_remote_offset{0}
        , send_sm_ptr{0}
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

        std::vector<MPI_Request> recv_sm_req;
        std::vector<MPI_Request> send_sm_req;

        for (const auto &pair : ghost_indices_within_larger_ghost_set)
          for (unsigned int c = 0, k = pair.first; k < pair.second; ++c, ++k)
            shifts.push_back(k);

        {
          std::map<unsigned int, std::vector<types::global_dof_index>>
            rank_to_local_indices;

          for (unsigned int i = 0; i < owning_ranks_of_ghosts.size(); i++)
            rank_to_local_indices[owning_ranks_of_ghosts[i]].push_back(i);

          unsigned int offset = 0;

          recv_sm_ptr_ = {0};

          for (const auto &rank_and_local_indices : rank_to_local_indices)
            {
              const auto ptr = std::find(sm_ranks.begin(),
                                         sm_ranks.end(),
                                         rank_and_local_indices.first);

              if (ptr == sm_ranks.end())
                {
                  // remote process
                  recv_remote_ranks.push_back(rank_and_local_indices.first);
                  recv_remote_ptr.emplace_back(
                    shifts[offset], rank_and_local_indices.second.size());

                  shifts_ptr.push_back(offset);
                }
              else
                {
                  // shared process
                  recv_sm_ranks.push_back(std::distance(sm_ranks.begin(), ptr));
                  recv_sm_ptr.push_back(recv_sm_ptr.back() +
                                        rank_and_local_indices.second.size());
                  recv_sm_offset.push_back(is_locally_owned.n_elements() +
                                           offset);

                  for (unsigned int i = offset, c = 0;
                       c < rank_and_local_indices.second.size();
                       ++c, ++i)
                    recv_sm_indices_.push_back(shifts[i] +
                                               is_locally_owned.n_elements());
                  recv_sm_ptr_.push_back(recv_sm_indices_.size());
                }
              offset += rank_and_local_indices.second.size();
            }
          recv_sm_req.resize(recv_sm_ranks.size());

          recv_sm_indices.resize(recv_sm_ptr.back());
        }

        {
          const auto rank_to_global_indices = process.get_requesters();

          for (const auto &rank_and_global_indices : rank_to_global_indices)
            {
              const auto ptr = std::find(sm_ranks.begin(),
                                         sm_ranks.end(),
                                         rank_and_global_indices.first);

              if (ptr == sm_ranks.end())
                {
                  // remote process
                  send_remote_ranks.push_back(rank_and_global_indices.first);

                  for (const auto &i : rank_and_global_indices.second)
                    send_remote_indices.push_back(
                      is_locally_owned.index_within_set(i));

                  send_remote_ptr.push_back(send_remote_indices.size());
                }
              else
                {
                  // shared process
                  send_sm_ranks.push_back(std::distance(sm_ranks.begin(), ptr));

                  for (const auto &i : rank_and_global_indices.second)
                    send_sm_indices.push_back(
                      is_locally_owned.index_within_set(i));

                  send_sm_ptr.push_back(send_sm_indices.size());
                }
            }
          send_sm_req.resize(send_sm_ranks.size());

          send_sm_ptr_ = send_sm_ptr;
          send_sm_indices_.resize(send_sm_ptr.back());
        }


        {
          for (unsigned int i = 0; i < recv_sm_ranks.size(); i++)
            MPI_Isend(recv_sm_indices_.data() + recv_sm_ptr_[i],
                      recv_sm_ptr_[i + 1] - recv_sm_ptr_[i],
                      MPI_UNSIGNED,
                      recv_sm_ranks[i],
                      4,
                      comm_sm,
                      recv_sm_req.data() + i);

          for (unsigned int i = 0; i < send_sm_ranks.size(); i++)
            MPI_Irecv(send_sm_indices_.data() + send_sm_ptr_[i],
                      send_sm_ptr_[i + 1] - send_sm_ptr_[i],
                      MPI_UNSIGNED,
                      send_sm_ranks[i],
                      4,
                      comm_sm,
                      send_sm_req.data() + i);

          MPI_Waitall(recv_sm_req.size(),
                      recv_sm_req.data(),
                      MPI_STATUSES_IGNORE);
          MPI_Waitall(send_sm_req.size(),
                      send_sm_req.data(),
                      MPI_STATUSES_IGNORE);
        }

        {
          for (unsigned int i = 0; i < send_sm_ranks.size(); i++)
            MPI_Isend(send_sm_indices.data() + send_sm_ptr[i],
                      send_sm_ptr[i + 1] - send_sm_ptr[i],
                      MPI_UNSIGNED,
                      send_sm_ranks[i],
                      2,
                      comm_sm,
                      send_sm_req.data() + i);

          for (unsigned int i = 0; i < recv_sm_ranks.size(); i++)
            MPI_Irecv(recv_sm_indices.data() + recv_sm_ptr[i],
                      recv_sm_ptr[i + 1] - recv_sm_ptr[i],
                      MPI_UNSIGNED,
                      recv_sm_ranks[i],
                      2,
                      comm_sm,
                      recv_sm_req.data() + i);

          MPI_Waitall(recv_sm_req.size(),
                      recv_sm_req.data(),
                      MPI_STATUSES_IGNORE);
          MPI_Waitall(send_sm_req.size(),
                      send_sm_req.data(),
                      MPI_STATUSES_IGNORE);
        }

        {
          send_sm_offset.resize(send_sm_ranks.size());

          for (unsigned int i = 0; i < send_sm_ranks.size(); i++)
            MPI_Irecv(send_sm_offset.data() + i,
                      1,
                      MPI_UNSIGNED,
                      send_sm_ranks[i],
                      3,
                      comm_sm,
                      send_sm_req.data() + i);

          for (unsigned int i = 0; i < recv_sm_ranks.size(); i++)
            MPI_Isend(recv_sm_offset.data() + i,
                      1,
                      MPI_UNSIGNED,
                      recv_sm_ranks[i],
                      3,
                      comm_sm,
                      recv_sm_req.data() + i);

          MPI_Waitall(recv_sm_req.size(),
                      recv_sm_req.data(),
                      MPI_STATUSES_IGNORE);
          MPI_Waitall(send_sm_req.size(),
                      send_sm_req.data(),
                      MPI_STATUSES_IGNORE);
        }

        internal::compress(recv_sm_ptr, recv_sm_indices, recv_sm_len);

        internal::compress(send_remote_ptr,
                           send_remote_indices,
                           send_remote_len);
        send_remote_offset.clear();
        send_remote_offset.push_back(0);

        for (unsigned int r = 0, c = 0; r < send_remote_ranks.size(); r++)
          {
            for (unsigned int i = send_remote_ptr[r];
                 i < send_remote_ptr[r + 1];
                 i++)
              c += send_remote_len[i];
            send_remote_offset.push_back(c);
          }

        internal::compress(send_sm_ptr, send_sm_indices, send_sm_len);
        internal::compress(recv_sm_ptr_, recv_sm_indices_, recv_sm_len_);
        internal::compress(send_sm_ptr_, send_sm_indices_, send_sm_len_);

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

        requests.resize(send_sm_ranks.size() + recv_sm_ranks.size() +
                        recv_remote_ranks.size() + send_remote_ranks.size());

        int dummy;
        // receive a signal that relevant sm neighbors are ready
        for (unsigned int i = 0; i < recv_sm_ranks.size(); i++)
          MPI_Irecv(&dummy,
                    0,
                    MPI_INT,
                    recv_sm_ranks[i],
                    communication_channel + 2,
                    comm_sm,
                    requests.data() + send_sm_ranks.size() + i);

        // signal to all relevant sm neighbors that this process is ready
        for (unsigned int i = 0; i < send_sm_ranks.size(); i++)
          MPI_Isend(&dummy,
                    0,
                    MPI_INT,
                    send_sm_ranks[i],
                    communication_channel + 2,
                    comm_sm,
                    requests.data() + i);

        // receive data from remote processes
        for (unsigned int i = 0; i < recv_remote_ranks.size(); i++)
          {
            const unsigned int offset =
              (shifts[shifts_ptr[i] + (recv_remote_ptr[i].second - 1)] -
               shifts[shifts_ptr[i]]) +
              1 - recv_remote_ptr[i].second;

            MPI_Irecv(buffer.data() + recv_remote_ptr[i].first + offset,
                      recv_remote_ptr[i].second,
                      Utilities::MPI::internal::mpi_type_id(buffer.data()),
                      recv_remote_ranks[i],
                      communication_channel + 3,
                      comm,
                      requests.data() + send_sm_ranks.size() +
                        recv_sm_ranks.size() + i);
          }

        // send data to remote processes
        for (unsigned int i = 0, k = 0; i < send_remote_ranks.size(); i++)
          {
            for (unsigned int j = send_remote_ptr[i];
                 j < send_remote_ptr[i + 1];
                 j++)
              for (unsigned int l = 0; l < send_remote_len[j]; l++, k++)
                temporary_storage[k] = data_this[send_remote_indices[j] + l];

            // send data away
            MPI_Isend(temporary_storage.data() + send_remote_offset[i],
                      send_remote_offset[i + 1] - send_remote_offset[i],
                      Utilities::MPI::internal::mpi_type_id(data_this.data()),
                      send_remote_ranks[i],
                      communication_channel + 3,
                      comm,
                      requests.data() + send_sm_ranks.size() +
                        recv_sm_ranks.size() + recv_remote_ranks.size() + i);
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
                        send_sm_ranks.size() + recv_sm_ranks.size() +
                          recv_remote_ranks.size() + send_remote_ranks.size());

        const auto split =
          [&](const unsigned int i) -> std::pair<unsigned int, unsigned int> {
          AssertIndexRange(i,
                           (recv_sm_ranks.size() + recv_remote_ranks.size()));

          if (i < recv_sm_ranks.size())
            return {0, i};
          else
            return {1, i - recv_sm_ranks.size()};
        };

        for (unsigned int c = 0;
             c < recv_sm_ranks.size() + recv_remote_ranks.size();
             c++)
          {
            int i;
            MPI_Waitany(recv_sm_ranks.size() + recv_remote_ranks.size(),
                        requests.data() + send_sm_ranks.size(),
                        &i,
                        MPI_STATUS_IGNORE);

            const auto s = split(i);
            i            = s.second;

            if (s.first == 0)
              {
                const Number *__restrict__ data_others_ptr =
                  data_others[recv_sm_ranks[i]].data();
                Number *__restrict__ data_this_ptr = ghost_array.data();

                for (unsigned int lo = recv_sm_ptr[i],
                                  ko = recv_sm_ptr_[i],
                                  li = 0,
                                  ki = 0;
                     (lo < recv_sm_ptr[i + 1]) && (ko < recv_sm_ptr_[i + 1]);)
                  {
                    for (; (li < recv_sm_len[lo]) && (ki < recv_sm_len_[ko]);
                         ++li, ++ki)
                      data_this_ptr[recv_sm_indices_[ko] + ki -
                                    n_local_elements] =
                        data_others_ptr[recv_sm_indices[lo] + li];

                    if (li == recv_sm_len[lo])
                      {
                        lo++;   // increment outer counter
                        li = 0; // reset inner counter
                      }

                    if (ki == recv_sm_len_[ko])
                      {
                        ko++;   // increment outer counter
                        ki = 0; // reset inner counter
                      }
                  }
              }
            else /*if(s.second == 1)*/
              {
                const unsigned int offset =
                  (shifts[shifts_ptr[i] + (recv_remote_ptr[i].second - 1)] -
                   shifts[shifts_ptr[i]]) +
                  1 - recv_remote_ptr[i].second;

                for (unsigned int c = 0; c < recv_remote_ptr[i].second; ++c)
                  {
                    const unsigned int idx_1 = shifts[shifts_ptr[i] + c];
                    const unsigned int idx_2 =
                      recv_remote_ptr[i].first + c + offset;

                    if (idx_1 == idx_2)
                      continue;

                    AssertIndexRange(idx_1, idx_2);

                    ghost_array[idx_1] = ghost_array[idx_2];
                    ghost_array[idx_2] = 0.0;
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

        requests.resize(recv_sm_ranks.size() + send_sm_ranks.size() +
                        recv_remote_ranks.size() + send_remote_ranks.size());

        int dummy;
        for (unsigned int i = 0; i < recv_sm_ranks.size(); i++)
          MPI_Isend(&dummy,
                    0,
                    MPI_INT,
                    recv_sm_ranks[i],
                    communication_channel + 1,
                    comm_sm,
                    requests.data() + i);

        for (unsigned int i = 0; i < send_sm_ranks.size(); i++)
          MPI_Irecv(&dummy,
                    0,
                    MPI_INT,
                    send_sm_ranks[i],
                    communication_channel + 1,
                    comm_sm,
                    requests.data() + recv_sm_ranks.size() + i);

        for (unsigned int i = 0; i < recv_remote_ranks.size(); i++)
          {
            for (unsigned int c = 0; c < recv_remote_ptr[i].second; ++c)
              {
                const unsigned int idx_1 = shifts[shifts_ptr[i] + c];
                const unsigned int idx_2 = recv_remote_ptr[i].first + c;

                if (idx_1 == idx_2)
                  continue;

                Assert(idx_2 < idx_1, ExcNotImplemented());

                buffer[idx_2] = buffer[idx_1];
                buffer[idx_1] = 0.0;
              }

            MPI_Isend(buffer.data() + recv_remote_ptr[i].first,
                      recv_remote_ptr[i].second,
                      Utilities::MPI::internal::mpi_type_id(buffer.data()),
                      recv_remote_ranks[i],
                      communication_channel + 0,
                      comm,
                      requests.data() + recv_sm_ranks.size() +
                        send_sm_ranks.size() + i);
          }

        for (unsigned int i = 0; i < send_remote_ranks.size(); i++)
          MPI_Irecv(temporary_storage.data() + send_remote_offset[i],
                    send_remote_offset[i + 1] - send_remote_offset[i],
                    Utilities::MPI::internal::mpi_type_id(
                      temporary_storage.data()),
                    send_remote_ranks[i],
                    communication_channel + 0,
                    comm,
                    requests.data() + recv_sm_ranks.size() +
                      send_sm_ranks.size() + recv_remote_ranks.size() + i);
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
                        recv_sm_ranks.size() + send_sm_ranks.size() +
                          recv_remote_ranks.size() + send_remote_ranks.size());

        const auto split =
          [&](const unsigned int i) -> std::pair<unsigned int, unsigned int> {
          AssertIndexRange(i,
                           (send_sm_ranks.size() + recv_remote_ranks.size() +
                            send_remote_ranks.size()));

          if (i < send_sm_ranks.size())
            return {0, i};
          else if (i < (send_sm_ranks.size() + recv_remote_ranks.size()))
            return {2, i - send_sm_ranks.size()};
          else
            return {1, i - send_sm_ranks.size() - recv_remote_ranks.size()};
        };

        for (unsigned int c = 0;
             c < send_sm_ranks.size() + send_remote_ranks.size() +
                   recv_remote_ranks.size();
             c++)
          {
            int i;
            MPI_Waitany(send_sm_ranks.size() + send_remote_ranks.size() +
                          recv_remote_ranks.size(),
                        requests.data() + recv_sm_ranks.size(),
                        &i,
                        MPI_STATUS_IGNORE);

            const auto &s = split(i);
            i             = s.second;

            if (s.first == 0)
              {
                Number *__restrict__ data_others_ptr =
                  const_cast<Number *>(data_others[send_sm_ranks[i]].data());
                Number *__restrict__ data_this_ptr = data_this.data();

                for (unsigned int lo = send_sm_ptr[i],
                                  ko = send_sm_ptr_[i],
                                  li = 0,
                                  ki = 0;
                     (lo < send_sm_ptr[i + 1]) && (ko < send_sm_ptr_[i + 1]);)
                  {
                    for (; (li < send_sm_len[lo]) && (ki < send_sm_len_[ko]);
                         ++li, ++ki)
                      {
                        data_this_ptr[send_sm_indices[lo] + li] +=
                          data_others_ptr[send_sm_indices_[ko] + ki];
                        data_others_ptr[send_sm_indices_[ko] + ki] = 0.0;
                      }

                    if (li == send_sm_len[lo])
                      {
                        lo++;   // increment outer counter
                        li = 0; // reset inner counter
                      }
                    if (ki == send_sm_len_[ko])
                      {
                        ko++;   // increment outer counter
                        ki = 0; // reset inner counter
                      }
                  }
              }
            else if (s.first == 1)
              {
                for (unsigned int j = send_remote_ptr[i],
                                  k = send_remote_offset[i];
                     j < send_remote_ptr[i + 1];
                     j++)
                  for (unsigned int l = 0; l < send_remote_len[j]; l++)
                    data_this[send_remote_indices[j] + l] +=
                      temporary_storage[k++];
              }
            else /*if (s.first == 2)*/
              {
                std::memset(buffer.data() + recv_remote_ptr[i].first,
                            0.0,
                            (recv_remote_ptr[i].second) * sizeof(Number));
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
        return send_remote_offset.back();
      }



      unsigned int
      Full::n_import_sm_procs() const
      {
        return send_sm_ranks.size() + recv_sm_ranks.size(); // TODO
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
