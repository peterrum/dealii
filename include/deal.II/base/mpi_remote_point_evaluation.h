// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
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

#ifndef dealii_mpi_mpi_remote_point_evaluation_h
#define dealii_mpi_mpi_remote_point_evaluation_h

#include <deal.II/base/config.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_tools_cache.h>

DEAL_II_NAMESPACE_OPEN


namespace Utilities
{
  namespace MPI
  {
    /**
     * TODO
     */
    template <int dim, int spacedim = dim>
    class RemotePointEvaluation
    {
    public:
      RemotePointEvaluation(const double tolerance = 1e-6);

      void
      reinit(const std::vector<Point<spacedim>>  quadrature_points,
             const Triangulation<dim, spacedim> &tria,
             const Mapping<dim, spacedim> &      mapping);

      /**
       * Evaluate function @p fu in the requested quadrature points. The result
       * is sorted according to rank.
       */
      template <typename T>
      void
      evaluate_and_process(
        std::vector<T> &output,
        std::vector<T> &buffer,
        const std::function<
          void(const ArrayView<T> &,
               const std::tuple<
                 std::vector<std::pair<std::pair<int, int>, unsigned int>>,
                 std::vector<Point<dim>>,
                 std::vector<unsigned int>> &)> &fu) const;

      template <typename T>
      void
      process_and_evaluate(
        const std::vector<T> &input,
        std::vector<T> &      buffer,
        const std::function<
          void(const ArrayView<const T> &,
               const std::tuple<
                 std::vector<std::pair<std::pair<int, int>, unsigned int>>,
                 std::vector<Point<dim>>,
                 std::vector<unsigned int>> &)> &fu) const;


      const std::vector<unsigned int> &
      get_quadrature_points_ptr() const;

      bool
      is_unique_mapping() const;

      const Triangulation<dim, spacedim> &
      get_triangulation() const;

      const Mapping<dim, spacedim> &
      get_mapping() const;

      bool
      is_ready() const;

    private:
      const double tolerance;

      SmartPointer<const Triangulation<dim, spacedim>> tria;
      SmartPointer<const Mapping<dim, spacedim>>       mapping;

      MPI_Comm comm;

      bool unique_mapping;

      // receiver side
      std::vector<unsigned int> quadrature_points_ptr;
      std::vector<unsigned int> indices;
      std::vector<unsigned int> indices_ptr;

      std::vector<unsigned int> recv_ranks;

      // sender side
      std::tuple<std::vector<std::pair<std::pair<int, int>, unsigned int>>,
                 std::vector<Point<dim>>,
                 std::vector<unsigned int>>
        relevant_remote_points_per_process;

      std::vector<unsigned int> send_ranks;
      std::vector<unsigned int> send_ptr;
    };


    template <int dim, int spacedim>
    template <typename T>
    void
    RemotePointEvaluation<dim, spacedim>::evaluate_and_process(
      std::vector<T> &output,
      std::vector<T> &buffer,
      const std::function<
        void(const ArrayView<T> &,
             const std::tuple<
               std::vector<std::pair<std::pair<int, int>, unsigned int>>,
               std::vector<Point<dim>>,
               std::vector<unsigned int>> &)> &fu) const
    {
      output.resize(quadrature_points_ptr.back());
      buffer.resize(
        (std::get<1>(this->relevant_remote_points_per_process).size()) * 2);
      ArrayView<T> buffer_1(buffer.data(), buffer.size() / 2);
      ArrayView<T> buffer_2(buffer.data() + buffer.size() / 2,
                            buffer.size() / 2);

      fu(buffer_1, relevant_remote_points_per_process);

      for (unsigned int i = 0;
           i < std::get<2>(relevant_remote_points_per_process).size();
           ++i)
        buffer_2[std::get<2>(relevant_remote_points_per_process)[i]] =
          buffer_1[i];

      // process remote quadrature points and send them away
      std::map<unsigned int, std::vector<char>> temp_map;

      std::vector<MPI_Request> requests;
      requests.reserve(send_ranks.size());

      const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

      // receive result

      std::map<unsigned int, std::vector<T>> temp_recv_map;

      for (unsigned int i = 0; i < send_ranks.size(); ++i)
        {
          if (send_ranks[i] == my_rank)
            {
              // process locally-owned values
              temp_recv_map[my_rank] =
                std::vector<T>(buffer_2.begin() + send_ptr[i],
                               buffer_2.begin() + send_ptr[i + 1]);
              continue;
            }

          temp_map[send_ranks[i]] =
            Utilities::pack(std::vector<T>(buffer_2.begin() + send_ptr[i],
                                           buffer_2.begin() + send_ptr[i + 1]));

          auto &buffer = temp_map[send_ranks[i]];

          requests.resize(requests.size() + 1);

          MPI_Isend(buffer.data(),
                    buffer.size(),
                    MPI_CHAR,
                    send_ranks[i],
                    11,
                    comm,
                    &requests.back());
        }

      for (unsigned int i = 0; i < recv_ranks.size(); ++i)
        {
          MPI_Status status;
          MPI_Probe(MPI_ANY_SOURCE, 11, comm, &status);

          int message_length;
          MPI_Get_count(&status, MPI_CHAR, &message_length);

          std::vector<char> buffer(message_length);

          MPI_Recv(buffer.data(),
                   buffer.size(),
                   MPI_CHAR,
                   status.MPI_SOURCE,
                   11,
                   comm,
                   MPI_STATUS_IGNORE);

          temp_recv_map[status.MPI_SOURCE] =
            Utilities::unpack<std::vector<T>>(buffer);
        }

      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

      auto it = indices.begin();
      for (const auto &j : temp_recv_map)
        for (const auto &i : j.second)
          output[*(it++)] = i;
    }


    template <int dim, int spacedim>
    template <typename T>
    void
    RemotePointEvaluation<dim, spacedim>::process_and_evaluate(
      const std::vector<T> &input,
      std::vector<T> &      buffer,
      const std::function<
        void(const ArrayView<const T> &,
             const std::tuple<
               std::vector<std::pair<std::pair<int, int>, unsigned int>>,
               std::vector<Point<dim>>,
               std::vector<unsigned int>> &)> &fu) const
    {
      // expand
      const auto &   ptr = this->get_quadrature_points_ptr();
      std::vector<T> buffer_(ptr.back());

      for (unsigned int i = 0, c = 0; i < ptr.size() - 1; ++i)
        {
          const auto n_entries = ptr[i + 1] - ptr[i];

          for (unsigned int j = 0; j < n_entries; ++j, ++c)
            buffer_[c] = input[i];
        }

      std::map<unsigned int, std::vector<T>> temp_recv_map;

      for (unsigned int i = 0; i < recv_ranks.size(); ++i)
        {
          temp_recv_map[recv_ranks[i]] =
            std::vector<T>(indices_ptr[i + 1] - indices_ptr[i]);
        }

      const unsigned int my_rank = Utilities::MPI::this_mpi_process(comm);

      if (std::find(send_ranks.begin(), send_ranks.end(), my_rank) !=
          send_ranks.end())
        {
          const unsigned int i = std::distance(std::find(send_ranks.begin(),
                                                         send_ranks.end(),
                                                         my_rank),
                                               send_ranks.begin());
          temp_recv_map[my_rank] =
            std::vector<T>(indices_ptr[i + 1] - indices_ptr[i]);
        }

      auto it = indices.begin();
      for (auto &j : temp_recv_map)
        for (auto &i : j.second)
          i = buffer_[*(it++)];

      // buffer.resize(quadrature_points_ptr.back());
      buffer.resize(
        (std::get<1>(this->relevant_remote_points_per_process).size()) * 2);
      ArrayView<T> buffer_1(buffer.data(), buffer.size() / 2);
      ArrayView<T> buffer_2(buffer.data() + buffer.size() / 2,
                            buffer.size() / 2);

      // process remote quadrature points and send them away
      std::map<unsigned int, std::vector<char>> temp_map;

      std::vector<MPI_Request> requests;
      requests.reserve(recv_ranks.size());

      for (unsigned int i = 0; i < recv_ranks.size(); ++i)
        {
          // continue;

          temp_map[recv_ranks[i]] =
            Utilities::pack(temp_recv_map[recv_ranks[i]]);

          auto &buffer_send = temp_map[recv_ranks[i]];

          requests.resize(requests.size() + 1);

          MPI_Isend(buffer_send.data(),
                    buffer_send.size(),
                    MPI_CHAR,
                    recv_ranks[i],
                    11,
                    comm,
                    &requests.back());
        }

      for (unsigned int i = 0; i < send_ranks.size(); ++i)
        {
          // continue;

          if (send_ranks[i] == my_rank)
            {
              const auto &buffer_send = temp_recv_map[send_ranks[i]];
              // process locally-owned values
              const unsigned int j = std::distance(send_ranks.begin(),
                                                   std::find(send_ranks.begin(),
                                                             send_ranks.end(),
                                                             my_rank));

              AssertDimension(buffer_send.size(),
                              send_ptr[j + 1] - send_ptr[j]);

              for (unsigned int i = send_ptr[j], c = 0; i < send_ptr[j + 1];
                   ++i, ++c)
                buffer_1[i] = buffer_send[c];

              continue;
            }

          MPI_Status status;
          MPI_Probe(MPI_ANY_SOURCE, 11, comm, &status);

          int message_length;
          MPI_Get_count(&status, MPI_CHAR, &message_length);

          std::vector<char> recv_buffer(message_length);

          MPI_Recv(recv_buffer.data(),
                   recv_buffer.size(),
                   MPI_CHAR,
                   status.MPI_SOURCE,
                   11,
                   comm,
                   MPI_STATUS_IGNORE);


          const auto recv_buffer_unpacked =
            Utilities::unpack<std::vector<T>>(recv_buffer);

          auto ptr =
            std::find(send_ranks.begin(), send_ranks.end(), status.MPI_SOURCE);

          Assert(ptr != send_ranks.end(), ExcNotImplemented());

          const unsigned int j = std::distance(send_ranks.begin(), ptr);

          AssertDimension(recv_buffer_unpacked.size(),
                          send_ptr[j + 1] - send_ptr[j]);

          for (unsigned int i = send_ptr[j], c = 0; i < send_ptr[j + 1];
               ++i, ++c)
            {
              AssertIndexRange(i, buffer_1.size());
              AssertIndexRange(c, recv_buffer_unpacked.size());
              buffer_1[i] = recv_buffer_unpacked[c];
            }
        }

      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

      for (unsigned int i = 0;
           i < std::get<2>(relevant_remote_points_per_process).size();
           ++i)
        buffer_2[i] =
          buffer_1[std::get<2>(relevant_remote_points_per_process)[i]];

      fu(buffer_2, relevant_remote_points_per_process);
    }

  } // end of namespace MPI
} // end of namespace Utilities


DEAL_II_NAMESPACE_CLOSE

#endif
