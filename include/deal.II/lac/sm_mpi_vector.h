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

#ifndef dealii_sm_mpi_vector_h
#define dealii_sm_mpi_vector_h

#include <deal.II/base/config.h>

#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>

#include <map>
#include <vector>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace SharedMPI
  {
    namespace
    {
      MPI_Comm
      create_sm(const MPI_Comm &comm)
      {
        int rank;
        MPI_Comm_rank(comm, &rank);

        MPI_Comm comm_shared;
#ifdef MPI_SM_SIZE
        // fix group size: only for testing
        int color = rank / MPI_SM_SIZE;
        MPI_Comm_split(comm, color, rank, &comm_shared);
#else
        MPI_Comm_split_type(
          comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &comm_shared);
#endif

        return comm_shared;
      }

      unsigned int
      n_procs_of_sm(const MPI_Comm &comm)
      {
        // create shared memory communicator
        MPI_Comm comm_shared = create_sm(comm);

        int size_shared_max;
        int size_shared;
        MPI_Comm_size(comm_shared, &size_shared);

        // get biggest size_shared
        MPI_Allreduce(
          &size_shared, &size_shared_max, 1, MPI_INT, MPI_MAX, comm);

        MPI_Comm_free(&comm_shared);

        return size_shared_max;
      }



      std::vector<int>
      procs_of_sm(const MPI_Comm &comm)
      {
        // extract information from comm
        int rank;
        MPI_Comm_rank(comm, &rank);

        // create shared memory communicator
        MPI_Comm comm_shared = create_sm(comm);

        // extract information from sm-comm
        int size_shared;
        MPI_Comm_size(comm_shared, &size_shared);

        // gather ranks
        std::vector<int> ranks_shared(size_shared);
        MPI_Allgather(
          &rank, 1, MPI_INT, ranks_shared.data(), 1, MPI_INT, comm_shared);

        MPI_Comm_free(&comm_shared);

        return ranks_shared;
      }

      template <typename T, typename U>
      std::vector<std::pair<T, U>>
      MPI_Allgather_Pairs(const std::vector<std::pair<T, U>> &src,
                          const MPI_Comm &                    comm)
      {
        int size;
        MPI_Comm_size(comm, &size);

        std::vector<T> src_1;
        std::vector<U> src_2;

        for (auto i : src)
          {
            src_1.push_back(i.first);
            src_2.push_back(i.second);
          }


        unsigned int     len_local = src_1.size();
        std::vector<int> len_global(
          size); // actually unsigned int but MPI wants int
        MPI_Allgather(&len_local,
                      1,
                      MPI_INT,
                      &len_global[0],
                      1,
                      Utilities::MPI::internal::mpi_type_id(&len_local),
                      comm);


        std::vector<int> displs; // actually unsigned int but MPI wants int
        displs.push_back(0);

        int total_size = 0;

        for (auto i : len_global)
          {
            displs.push_back(i + displs.back());
            total_size += i;
          }

        std::vector<T> dst_1(total_size);
        std::vector<U> dst_2(total_size);
        MPI_Allgatherv(&src_1[0],
                       len_local,
                       Utilities::MPI::internal::mpi_type_id(&src_1[0]),
                       &dst_1[0],
                       &len_global[0],
                       &displs[0],
                       Utilities::MPI::internal::mpi_type_id(&dst_1[0]),
                       comm);
        MPI_Allgatherv(&src_2[0],
                       len_local,
                       Utilities::MPI::internal::mpi_type_id(&src_2[0]),
                       &dst_2[0],
                       &len_global[0],
                       &displs[0],
                       Utilities::MPI::internal::mpi_type_id(&dst_2[0]),
                       comm);

        std::vector<std::pair<T, U>> dst(total_size);

        for (unsigned int i = 0; i < dst_1.size(); i++)
          dst[i] = {dst_1[i], dst_2[i]};

        return dst;
      }

    } // namespace

    class Partitioner
    {
    public:
      void
      configure(const bool         do_buffering,
                const unsigned int degree,
                const unsigned int dim)
      {
        this->do_buffering = do_buffering;
        this->degree       = degree;
        this->dim          = dim;
      }

      void
      reinit(const std::vector<types::global_dof_index> local_cells,
             const std::vector<
               std::pair<types::global_dof_index, std::vector<unsigned int>>>
                            local_ghost_faces,
             const MPI_Comm comm)
      {
        AssertThrow(local_cells.size() > 0, ExcMessage("No local cells!"));

        // unknowns per ghost cell and ...
        const types::global_dof_index dofs_per_cell =
          Utilities::pow(degree + 1, dim);

        // .. ghost face
        const types::global_dof_index dofs_per_face =
          Utilities::pow(degree + 1, dim - 1);

        // 1) determine if ghost faces or ghost cells are needed
        const types::global_dof_index dofs_per_ghost = [&]() {
          unsigned int result = dofs_per_face;

          for (const auto &ghost_faces : local_ghost_faces)
            for (const auto ghost_face : ghost_faces.second)
              if (ghost_face == numbers::invalid_unsigned_int)
                result = dofs_per_cell;
          return Utilities::MPI::max(result, comm);
        }();

        const auto sm_comm  = create_sm(comm);
        const auto sm_procs = procs_of_sm(comm);
        const auto sm_rank =
          std::distance(sm_procs.begin(),
                        std::find(sm_procs.begin(),
                                  sm_procs.end(),
                                  Utilities::MPI::this_mpi_process(comm)));

        std::map<types::global_dof_index, std::pair<unsigned int, unsigned int>>
          maps;

        for (unsigned int i = 0; i < local_cells.size(); i++)
          maps[local_cells[i]] = {sm_rank, i * dofs_per_cell};

        // 2) determine which ghost face is shared or remote
        std::vector<
          std::pair<types::global_dof_index, std::vector<unsigned int>>>
          local_ghost_faces_remote, local_ghost_faces_shared;
        {
          const auto n_total_cells = Utilities::MPI::sum(
            static_cast<types::global_dof_index>(local_cells.size()), comm);

          IndexSet is_local_cells(n_total_cells);
          is_local_cells.add_indices(local_cells.begin(), local_cells.end());

          IndexSet is_ghost_cells(n_total_cells);
          for (const auto &ghost_faces : local_ghost_faces)
            is_ghost_cells.add_index(ghost_faces.first);

          AssertDimension(local_ghost_faces.size(),
                          is_ghost_cells.n_elements());

          std::vector<unsigned int> owning_ranks_of_ghosts(
            is_ghost_cells.n_elements());

          // set up dictionary
          Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmPayload
            process(is_local_cells,
                    is_ghost_cells,
                    comm,
                    owning_ranks_of_ghosts,
                    false);

          Utilities::MPI::ConsensusAlgorithmSelector<
            std::pair<types::global_dof_index, types::global_dof_index>,
            unsigned int>
            consensus_algorithm(process, comm);
          consensus_algorithm.run();

          std::map<unsigned int, std::vector<types::global_dof_index>>
            shared_procs_to_cells;

          for (unsigned int i = 0; i < owning_ranks_of_ghosts.size(); i++)
            if (std::find(sm_procs.begin(),
                          sm_procs.end(),
                          owning_ranks_of_ghosts[i]) == sm_procs.end())
              local_ghost_faces_remote.push_back(local_ghost_faces[i]);
            else
              {
                local_ghost_faces_shared.push_back(local_ghost_faces[i]);
                shared_procs_to_cells[std::distance(
                                        sm_procs.begin(),
                                        std::find(sm_procs.begin(),
                                                  sm_procs.end(),
                                                  owning_ranks_of_ghosts[i]))]
                  .emplace_back(local_ghost_faces[i].first);
              }

          // determine offsets -> maps
          {
            auto shared_procs_to_cells_ptr = shared_procs_to_cells.begin();
            std::vector<MPI_Request> querry_request(
              shared_procs_to_cells.size());
            std::vector<MPI_Request> answer_request(
              shared_procs_to_cells.size());

            std::map<unsigned int, std::vector<types::global_dof_index>>
              answer_buffer;
            for (const auto &i : shared_procs_to_cells)
              answer_buffer[i.first].resize(i.second.size());

            std::map<unsigned int, std::vector<types::global_dof_index>>
              shared_procs_to_offset;
            for (const auto &i : shared_procs_to_cells)
              shared_procs_to_offset[i.first].resize(i.second.size());

            for (unsigned int i = 0; i < shared_procs_to_cells.size();
                 i++, shared_procs_to_cells_ptr++)
              {
                types::global_dof_index dummy;
                MPI_Isend(shared_procs_to_cells_ptr->second.data(),
                          shared_procs_to_cells_ptr->second.size(),
                          Utilities::MPI::internal::mpi_type_id(&dummy),
                          shared_procs_to_cells_ptr->first,
                          102,
                          sm_comm,
                          &querry_request[i]);
              }

            for (unsigned int i = 0; i < shared_procs_to_cells.size(); i++)
              {
                MPI_Status status;
                auto ierr = MPI_Probe(MPI_ANY_SOURCE, 102, sm_comm, &status);
                AssertThrowMPI(ierr);

                types::global_dof_index dummy;
                ierr = MPI_Recv(answer_buffer[status.MPI_SOURCE].data(),
                                answer_buffer[status.MPI_SOURCE].size(),
                                Utilities::MPI::internal::mpi_type_id(&dummy),
                                status.MPI_SOURCE,
                                102,
                                sm_comm,
                                &status);
                AssertThrowMPI(ierr);

                for (auto &i : answer_buffer[status.MPI_SOURCE])
                  i = std::distance(local_cells.begin(),
                                    std::find(local_cells.begin(),
                                              local_cells.end(),
                                              i));

                MPI_Isend(answer_buffer[status.MPI_SOURCE].data(),
                          answer_buffer[status.MPI_SOURCE].size(),
                          Utilities::MPI::internal::mpi_type_id(&dummy),
                          status.MPI_SOURCE,
                          103,
                          sm_comm,
                          &answer_request[i]);
              }

            for (unsigned int i = 0; i < shared_procs_to_cells.size(); i++)
              {
                MPI_Status status;
                auto ierr = MPI_Probe(MPI_ANY_SOURCE, 103, sm_comm, &status);
                AssertThrowMPI(ierr);

                types::global_dof_index dummy;
                ierr =
                  MPI_Recv(shared_procs_to_offset[status.MPI_SOURCE].data(),
                           shared_procs_to_offset[status.MPI_SOURCE].size(),
                           Utilities::MPI::internal::mpi_type_id(&dummy),
                           status.MPI_SOURCE,
                           103,
                           sm_comm,
                           &status);
                AssertThrowMPI(ierr);

                for (unsigned int i = 0;
                     i < shared_procs_to_offset[status.MPI_SOURCE].size();
                     i++)
                  {
                    const unsigned int cell =
                      shared_procs_to_cells[status.MPI_SOURCE][i];
                    const unsigned int offset =
                      shared_procs_to_offset[status.MPI_SOURCE][i];

                    maps[cell] = {status.MPI_SOURCE, offset * dofs_per_cell};
                  }
              }

            MPI_Waitall(querry_request.size(),
                        querry_request.data(),
                        MPI_STATUSES_IGNORE);
            MPI_Waitall(answer_request.size(),
                        answer_request.data(),
                        MPI_STATUSES_IGNORE);
          }
        }

        // 3) merge local_ghost_faces_remote and sort -> ghost_faces_remote
        std::vector<std::pair<types::global_dof_index, unsigned int>>
          local_ghost_faces_remote_pairs_local;

        // convert vector<pair<U, std::vector<V>>> ->.vector<std::pair<U, V>>>
        for (const auto &ghost_faces : local_ghost_faces_remote)
          for (const auto ghost_face : ghost_faces.second)
            local_ghost_faces_remote_pairs_local.emplace_back(ghost_faces.first,
                                                              ghost_face);

        deallog << local_ghost_faces_remote_pairs_local.size() << std::endl;

        // collect all on which are shared
        std::vector<std::pair<types::global_dof_index, unsigned int>>
          local_ghost_faces_remote_pairs_global =
            MPI_Allgather_Pairs(local_ghost_faces_remote_pairs_local,
                                create_sm(comm));

        deallog << local_ghost_faces_remote_pairs_global.size() << std::endl;

        // sort
        std::sort(local_ghost_faces_remote_pairs_global.begin(),
                  local_ghost_faces_remote_pairs_global.end());


        // 4) distributed ghost_faces_remote
        const auto distributed_local_ghost_faces_remote_pairs_global = [&]() {
          std::vector<
            std::vector<std::pair<types::global_dof_index, unsigned int>>>
            result(sm_procs.size());

          unsigned int       counter = 0;
          const unsigned int faces_per_process =
            (local_ghost_faces_remote_pairs_global.size() + sm_procs.size() -
             1) /
            sm_procs.size();
          for (auto p : local_ghost_faces_remote_pairs_global)
            result[(counter++) / faces_per_process].push_back(p);

          return result;
        }();

        deallog << "A"
                << distributed_local_ghost_faces_remote_pairs_global.size()
                << std::endl;

        for (auto &i : distributed_local_ghost_faces_remote_pairs_global)
          {
            for (auto &j : i)
              deallog << "(" << j.first << "," << j.second << ") ";
            deallog << std::endl;
          }

        // 5) re-determine owners remote ghost faces
        {
          const auto n_total_cells = Utilities::MPI::sum(
            static_cast<types::global_dof_index>(local_cells.size()), comm);

          IndexSet is_local_cells(n_total_cells);
          is_local_cells.add_indices(local_cells.begin(), local_cells.end());

          IndexSet is_ghost_cells(n_total_cells);
          for (const auto &ghost_faces :
               distributed_local_ghost_faces_remote_pairs_global[sm_rank])
            is_ghost_cells.add_index(ghost_faces.first);

          std::vector<unsigned int> owning_ranks_of_ghosts(
            is_ghost_cells.n_elements());

          Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmPayload
            process(is_local_cells,
                    is_ghost_cells,
                    comm,
                    owning_ranks_of_ghosts,
                    false);

          Utilities::MPI::ConsensusAlgorithmSelector<
            std::pair<types::global_dof_index, types::global_dof_index>,
            unsigned int>
            consensus_algorithm(process, comm);
          consensus_algorithm.run();

          std::set<unsigned int> send_ranks_set;

          for (const auto &i : owning_ranks_of_ghosts)
            send_ranks_set.insert(i);

          const std::vector<unsigned int> send_ranks(send_ranks_set.begin(),
                                                     send_ranks_set.end());
          const std::vector<unsigned int> recv_ranks =
            Utilities::MPI::compute_point_to_point_communication_pattern(
              comm, send_ranks);

          std::vector<std::vector<
            std::pair<types::global_dof_index, types::global_dof_index>>>
            send_data(send_ranks.size());

          std::vector<std::vector<
            std::pair<types::global_dof_index, types::global_dof_index>>>
            recv_data(recv_ranks.size());

          unsigned int index      = 0;
          unsigned int index_rank = numbers::invalid_unsigned_int;

          for (const auto &ghost_faces :
               distributed_local_ghost_faces_remote_pairs_global[sm_rank])
            {
              if (index_rank != ghost_faces.first)
                {
                  index_rank = ghost_faces.first;
                  index      = std::distance(send_ranks.begin(),
                                        std::find(send_ranks.begin(),
                                                  send_ranks.end(),
                                                  index_rank));
                }
              send_data[index].emplace_back(ghost_faces.first,
                                            ghost_faces.second);
            }

          std::vector<MPI_Request> send_requests(send_ranks.size());

          for (unsigned int i = 0; i < send_ranks.size(); i++)
            {
              types::global_dof_index dummy;
              MPI_Isend(send_data[i].data(),
                        2 * send_data[i].size(),
                        Utilities::MPI::internal::mpi_type_id(&dummy),
                        send_ranks[i],
                        101,
                        comm,
                        &send_requests[i]);
            }

          for (unsigned int i = 0; i < recv_ranks.size(); i++)
            {
              MPI_Status status;
              auto       ierr = MPI_Probe(MPI_ANY_SOURCE, 101, comm, &status);
              AssertThrowMPI(ierr);

              int len;

              types::global_dof_index dummy;
              MPI_Get_count(&status,
                            Utilities::MPI::internal::mpi_type_id(&dummy),
                            &len);

              const unsigned int index =
                std::distance(recv_ranks.begin(),
                              std::find(recv_ranks.begin(),
                                        recv_ranks.end(),
                                        status.MPI_SOURCE));

              recv_data[index].resize(len / 2);

              ierr = MPI_Recv(recv_data[index].data(),
                              len,
                              Utilities::MPI::internal::mpi_type_id(&dummy),
                              status.MPI_SOURCE,
                              status.MPI_TAG,
                              comm,
                              &status);
              AssertThrowMPI(ierr);
            }

          MPI_Waitall(send_requests.size(),
                      send_requests.data(),
                      MPI_STATUSES_IGNORE);
        }

        // 6)
      }

    private:
      bool         do_buffering;
      unsigned int degree;
      unsigned int dim;
    };

  } // namespace SharedMPI
} // namespace LinearAlgebra


DEAL_II_NAMESPACE_CLOSE

#endif
