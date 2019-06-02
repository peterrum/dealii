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


#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/vector_memory.h>

#include <iostream>
#include <numeric>
#include <set>

#ifdef DEAL_II_WITH_TRILINOS
#  ifdef DEAL_II_WITH_MPI
#    include <deal.II/lac/trilinos_parallel_block_vector.h>
#    include <deal.II/lac/trilinos_vector.h>
#    include <deal.II/lac/vector_memory.h>

#    include <Epetra_MpiComm.h>
#  endif
#endif

#ifdef DEAL_II_WITH_PETSC
#  include <deal.II/lac/petsc_block_vector.h>
#  include <deal.II/lac/petsc_vector.h>

#  include <petscsys.h>
#endif

#ifdef DEAL_II_WITH_SLEPC
#  include <deal.II/lac/slepc_solver.h>

#  include <slepcsys.h>
#endif

#ifdef DEAL_II_WITH_P4EST
#  include <p4est_bits.h>
#endif

#ifdef DEAL_II_TRILINOS_WITH_ZOLTAN
#  include <zoltan_cpp.h>
#endif

DEAL_II_NAMESPACE_OPEN


namespace Utilities
{
  namespace MPI
  {
#ifdef DEAL_II_WITH_MPI
    unsigned int
    n_mpi_processes(const MPI_Comm &mpi_communicator)
    {
      int       n_jobs = 1;
      const int ierr   = MPI_Comm_size(mpi_communicator, &n_jobs);
      AssertThrowMPI(ierr);

      return n_jobs;
    }


    unsigned int
    this_mpi_process(const MPI_Comm &mpi_communicator)
    {
      int       rank = 0;
      const int ierr = MPI_Comm_rank(mpi_communicator, &rank);
      AssertThrowMPI(ierr);

      return rank;
    }


    MPI_Comm
    duplicate_communicator(const MPI_Comm &mpi_communicator)
    {
      MPI_Comm  new_communicator;
      const int ierr = MPI_Comm_dup(mpi_communicator, &new_communicator);
      AssertThrowMPI(ierr);
      return new_communicator;
    }



    int
    create_group(const MPI_Comm & comm,
                 const MPI_Group &group,
                 const int        tag,
                 MPI_Comm *       new_comm)
    {
#  if DEAL_II_MPI_VERSION_GTE(3, 0)
      return MPI_Comm_create_group(comm, group, tag, new_comm);
#  else
      int rank;
      int ierr = MPI_Comm_rank(comm, &rank);
      AssertThrowMPI(ierr);

      int grp_rank;
      ierr = MPI_Group_rank(group, &grp_rank);
      AssertThrowMPI(ierr);
      if (grp_rank == MPI_UNDEFINED)
        {
          *new_comm = MPI_COMM_NULL;
          return MPI_SUCCESS;
        }

      int grp_size;
      ierr = MPI_Group_size(group, &grp_size);
      AssertThrowMPI(ierr);

      ierr = MPI_Comm_dup(MPI_COMM_SELF, new_comm);
      AssertThrowMPI(ierr);

      MPI_Group parent_grp;
      ierr = MPI_Comm_group(comm, &parent_grp);
      AssertThrowMPI(ierr);

      std::vector<int> pids(grp_size);
      std::vector<int> grp_pids(grp_size);
      std::iota(grp_pids.begin(), grp_pids.end(), 0);
      ierr = MPI_Group_translate_ranks(
        group, grp_size, grp_pids.data(), parent_grp, pids.data());
      AssertThrowMPI(ierr);
      ierr = MPI_Group_free(&parent_grp);
      AssertThrowMPI(ierr);

      MPI_Comm comm_old = *new_comm;
      MPI_Comm ic;
      for (int merge_sz = 1; merge_sz < grp_size; merge_sz *= 2)
        {
          const int gid = grp_rank / merge_sz;
          comm_old      = *new_comm;
          if (gid % 2 == 0)
            {
              if ((gid + 1) * merge_sz < grp_size)
                {
                  ierr = (MPI_Intercomm_create(
                    *new_comm, 0, comm, pids[(gid + 1) * merge_sz], tag, &ic));
                  AssertThrowMPI(ierr);
                  ierr = MPI_Intercomm_merge(ic, 0 /* LOW */, new_comm);
                  AssertThrowMPI(ierr);
                }
            }
          else
            {
              ierr = MPI_Intercomm_create(
                *new_comm, 0, comm, pids[(gid - 1) * merge_sz], tag, &ic);
              AssertThrowMPI(ierr);
              ierr = MPI_Intercomm_merge(ic, 1 /* HIGH */, new_comm);
              AssertThrowMPI(ierr);
            }
          if (*new_comm != comm_old)
            {
              ierr = MPI_Comm_free(&ic);
              AssertThrowMPI(ierr);
              ierr = MPI_Comm_free(&comm_old);
              AssertThrowMPI(ierr);
            }
        }

      return MPI_SUCCESS;
#  endif
    }



    std::vector<IndexSet>
    create_ascending_partitioning(const MPI_Comm &           comm,
                                  const IndexSet::size_type &local_size)
    {
      const unsigned int                     n_proc = n_mpi_processes(comm);
      const std::vector<IndexSet::size_type> sizes =
        all_gather(comm, local_size);
      const auto total_size =
        std::accumulate(sizes.begin(), sizes.end(), IndexSet::size_type(0));

      std::vector<IndexSet> res(n_proc, IndexSet(total_size));

      IndexSet::size_type begin = 0;
      for (unsigned int i = 0; i < n_proc; ++i)
        {
          res[i].add_range(begin, begin + sizes[i]);
          begin = begin + sizes[i];
        }

      return res;
    }



    std::vector<unsigned int>
    compute_point_to_point_communication_pattern(
      const MPI_Comm &                 mpi_comm,
      const std::vector<unsigned int> &destinations)
    {
      const unsigned int myid    = Utilities::MPI::this_mpi_process(mpi_comm);
      const unsigned int n_procs = Utilities::MPI::n_mpi_processes(mpi_comm);

      for (const unsigned int destination : destinations)
        {
          (void)destination;
          Assert(destination < n_procs, ExcIndexRange(destination, 0, n_procs));
          Assert(destination != myid,
                 ExcMessage(
                   "There is no point in communicating with ourselves."));
        }

#  if DEAL_II_MPI_VERSION_GTE(2, 2)
      // Calculate the number of messages to send to each process
      std::vector<unsigned int> dest_vector(n_procs);
      for (const auto &el : destinations)
        ++dest_vector[el];

      // Find how many processes will send to this one
      // by reducing with sum and then scattering the
      // results over all processes
      unsigned int n_recv_from;
      const int    ierr = MPI_Reduce_scatter_block(
        dest_vector.data(), &n_recv_from, 1, MPI_UNSIGNED, MPI_SUM, mpi_comm);

      AssertThrowMPI(ierr);

      // Send myid to every process in `destinations` vector...
      std::vector<MPI_Request> send_requests(destinations.size());
      for (const auto &el : destinations)
        MPI_Isend(&myid,
                  1,
                  MPI_UNSIGNED,
                  el,
                  32766,
                  mpi_comm,
                  send_requests.data() + (&el - destinations.data()));

      // if no one to receive from, return an empty vector
      if (n_recv_from == 0)
        return std::vector<unsigned int>();

      // ...otherwise receive `n_recv_from` times from the processes
      // who communicate with this one. Store the obtained id's
      // in the resulting vector
      std::vector<unsigned int> origins(n_recv_from);
      for (auto &el : origins)
        MPI_Recv(&el,
                 1,
                 MPI_UNSIGNED,
                 MPI_ANY_SOURCE,
                 32766,
                 mpi_comm,
                 MPI_STATUS_IGNORE);

      MPI_Waitall(destinations.size(),
                  send_requests.data(),
                  MPI_STATUSES_IGNORE);
      return origins;
#  else
      // let all processors communicate the maximal number of destinations
      // they have
      const unsigned int max_n_destinations =
        Utilities::MPI::max(destinations.size(), mpi_comm);

      if (max_n_destinations == 0)
        // all processes have nothing to send/receive:
        return std::vector<unsigned int>();

      // now that we know the number of data packets every processor wants to
      // send, set up a buffer with the maximal size and copy our destinations
      // in there, padded with -1's
      std::vector<unsigned int> my_destinations(max_n_destinations,
                                                numbers::invalid_unsigned_int);
      std::copy(destinations.begin(),
                destinations.end(),
                my_destinations.begin());

      // now exchange these (we could communicate less data if we used
      // MPI_Allgatherv, but we'd have to communicate my_n_destinations to all
      // processors in this case, which is more expensive than the reduction
      // operation above in MPI_Allreduce)
      std::vector<unsigned int> all_destinations(max_n_destinations * n_procs);
      const int                 ierr = MPI_Allgather(my_destinations.data(),
                                     max_n_destinations,
                                     MPI_UNSIGNED,
                                     all_destinations.data(),
                                     max_n_destinations,
                                     MPI_UNSIGNED,
                                     mpi_comm);
      AssertThrowMPI(ierr);

      // now we know who is going to communicate with whom. collect who is
      // going to communicate with us!
      std::vector<unsigned int> origins;
      for (unsigned int i = 0; i < n_procs; ++i)
        for (unsigned int j = 0; j < max_n_destinations; ++j)
          if (all_destinations[i * max_n_destinations + j] == myid)
            origins.push_back(i);
          else if (all_destinations[i * max_n_destinations + j] ==
                   numbers::invalid_unsigned_int)
            break;

      return origins;
#  endif
    }



    unsigned int
    compute_n_point_to_point_communications(
      const MPI_Comm &                 mpi_comm,
      const std::vector<unsigned int> &destinations)
    {
      const unsigned int n_procs = Utilities::MPI::n_mpi_processes(mpi_comm);

      for (const unsigned int destination : destinations)
        {
          (void)destination;
          Assert(destination < n_procs, ExcIndexRange(destination, 0, n_procs));
          Assert(destination != Utilities::MPI::this_mpi_process(mpi_comm),
                 ExcMessage(
                   "There is no point in communicating with ourselves."));
        }

      // Calculate the number of messages to send to each process
      std::vector<unsigned int> dest_vector(n_procs);
      for (const auto &el : destinations)
        ++dest_vector[el];

#  if DEAL_II_MPI_VERSION_GTE(2, 2)
      // Find out how many processes will send to this one
      // MPI_Reduce_scatter(_block) does exactly this
      unsigned int n_recv_from = 0;

      const int ierr = MPI_Reduce_scatter_block(
        dest_vector.data(), &n_recv_from, 1, MPI_UNSIGNED, MPI_SUM, mpi_comm);

      AssertThrowMPI(ierr);

      return n_recv_from;
#  else
      // Find out how many processes will send to this one
      // by reducing with sum and then scattering the
      // results over all processes
      std::vector<unsigned int> buffer(dest_vector.size());
      unsigned int              n_recv_from = 0;

      MPI_Reduce(dest_vector.data(),
                 buffer.data(),
                 dest_vector.size(),
                 MPI_UNSIGNED,
                 MPI_SUM,
                 0,
                 mpi_comm);
      MPI_Scatter(buffer.data(),
                  1,
                  MPI_UNSIGNED,
                  &n_recv_from,
                  1,
                  MPI_UNSIGNED,
                  0,
                  mpi_comm);

      return n_recv_from;
#  endif
    }



    namespace
    {
      // custom MIP_Op for calculate_collective_mpi_min_max_avg
      void
      max_reduce(const void *in_lhs_,
                 void *      inout_rhs_,
                 int *       len,
                 MPI_Datatype *)
      {
        (void)len;
        const MinMaxAvg *in_lhs    = static_cast<const MinMaxAvg *>(in_lhs_);
        MinMaxAvg *      inout_rhs = static_cast<MinMaxAvg *>(inout_rhs_);

        Assert(*len == 1, ExcInternalError());

        inout_rhs->sum += in_lhs->sum;
        if (inout_rhs->min > in_lhs->min)
          {
            inout_rhs->min       = in_lhs->min;
            inout_rhs->min_index = in_lhs->min_index;
          }
        else if (inout_rhs->min == in_lhs->min)
          {
            // choose lower cpu index when tied to make operator commutative
            if (inout_rhs->min_index > in_lhs->min_index)
              inout_rhs->min_index = in_lhs->min_index;
          }

        if (inout_rhs->max < in_lhs->max)
          {
            inout_rhs->max       = in_lhs->max;
            inout_rhs->max_index = in_lhs->max_index;
          }
        else if (inout_rhs->max == in_lhs->max)
          {
            // choose lower cpu index when tied to make operator commutative
            if (inout_rhs->max_index > in_lhs->max_index)
              inout_rhs->max_index = in_lhs->max_index;
          }
      }
    } // namespace



    MinMaxAvg
    min_max_avg(const double my_value, const MPI_Comm &mpi_communicator)
    {
      // If MPI was not started, we have a serial computation and cannot run
      // the other MPI commands
      if (job_supports_mpi() == false)
        {
          MinMaxAvg result;
          result.sum       = my_value;
          result.avg       = my_value;
          result.min       = my_value;
          result.max       = my_value;
          result.min_index = 0;
          result.max_index = 0;

          return result;
        }

      // To avoid uninitialized values on some MPI implementations, provide
      // result with a default value already...
      MinMaxAvg result = {0.,
                          std::numeric_limits<double>::max(),
                          -std::numeric_limits<double>::max(),
                          0,
                          0,
                          0.};

      const unsigned int my_id =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
      const unsigned int numproc =
        dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);

      MPI_Op op;
      int    ierr =
        MPI_Op_create(reinterpret_cast<MPI_User_function *>(&max_reduce),
                      true,
                      &op);
      AssertThrowMPI(ierr);

      MinMaxAvg in;
      in.sum = in.min = in.max = my_value;
      in.min_index = in.max_index = my_id;

      MPI_Datatype type;
      int          lengths[]       = {3, 2};
      MPI_Aint     displacements[] = {0, offsetof(MinMaxAvg, min_index)};
      MPI_Datatype types[]         = {MPI_DOUBLE, MPI_INT};

      ierr = MPI_Type_create_struct(2, lengths, displacements, types, &type);
      AssertThrowMPI(ierr);

      ierr = MPI_Type_commit(&type);
      AssertThrowMPI(ierr);
      ierr = MPI_Allreduce(&in, &result, 1, type, op, mpi_communicator);
      AssertThrowMPI(ierr);

      ierr = MPI_Type_free(&type);
      AssertThrowMPI(ierr);

      ierr = MPI_Op_free(&op);
      AssertThrowMPI(ierr);

      result.avg = result.sum / numproc;

      return result;
    }

#else

    unsigned int
    n_mpi_processes(const MPI_Comm &)
    {
      return 1;
    }



    unsigned int
    this_mpi_process(const MPI_Comm &)
    {
      return 0;
    }


    std::vector<IndexSet>
    create_ascending_partitioning(const MPI_Comm & /*comm*/,
                                  const IndexSet::size_type &local_size)
    {
      return std::vector<IndexSet>(1, complete_index_set(local_size));
    }


    MPI_Comm
    duplicate_communicator(const MPI_Comm &mpi_communicator)
    {
      return mpi_communicator;
    }



    MinMaxAvg
    min_max_avg(const double my_value, const MPI_Comm &)
    {
      MinMaxAvg result;

      result.sum       = my_value;
      result.avg       = my_value;
      result.min       = my_value;
      result.max       = my_value;
      result.min_index = 0;
      result.max_index = 0;

      return result;
    }

#endif



    MPI_InitFinalize::MPI_InitFinalize(int &              argc,
                                       char **&           argv,
                                       const unsigned int max_num_threads)
    {
      static bool constructor_has_already_run = false;
      (void)constructor_has_already_run;
      Assert(constructor_has_already_run == false,
             ExcMessage("You can only create a single object of this class "
                        "in a program since it initializes the MPI system."));


      int ierr = 0;
#ifdef DEAL_II_WITH_MPI
      // if we have PETSc, we will initialize it and let it handle MPI.
      // Otherwise, we will do it.
      int MPI_has_been_started = 0;
      ierr                     = MPI_Initialized(&MPI_has_been_started);
      AssertThrowMPI(ierr);
      AssertThrow(MPI_has_been_started == 0,
                  ExcMessage("MPI error. You can only start MPI once!"));

      int provided;
      // this works like ierr = MPI_Init (&argc, &argv); but tells MPI that
      // we might use several threads but never call two MPI functions at the
      // same time. For an explanation see on why we do this see
      // http://www.open-mpi.org/community/lists/users/2010/03/12244.php
      int wanted = MPI_THREAD_SERIALIZED;
      ierr       = MPI_Init_thread(&argc, &argv, wanted, &provided);
      AssertThrowMPI(ierr);

      // disable for now because at least some implementations always return
      // MPI_THREAD_SINGLE.
      // Assert(max_num_threads==1 || provided != MPI_THREAD_SINGLE,
      //    ExcMessage("MPI reports that we are not allowed to use multiple
      //    threads."));
#else
      // make sure the compiler doesn't warn about these variables
      (void)argc;
      (void)argv;
      (void)ierr;
#endif

      // we are allowed to call MPI_Init ourselves and PETScInitialize will
      // detect this. This allows us to use MPI_Init_thread instead.
#ifdef DEAL_II_WITH_PETSC
#  ifdef DEAL_II_WITH_SLEPC
      // Initialize SLEPc (with PETSc):
      ierr = SlepcInitialize(&argc, &argv, nullptr, nullptr);
      AssertThrow(ierr == 0, SLEPcWrappers::SolverBase::ExcSLEPcError(ierr));
#  else
      // or just initialize PETSc alone:
      ierr = PetscInitialize(&argc, &argv, nullptr, nullptr);
      AssertThrow(ierr == 0, ExcPETScError(ierr));
#  endif

      // Disable PETSc exception handling. This just prints a large wall
      // of text that is not particularly helpful for what we do:
      PetscPopSignalHandler();
#endif

      // Initialize zoltan
#ifdef DEAL_II_TRILINOS_WITH_ZOLTAN
      float version;
      Zoltan_Initialize(argc, argv, &version);
#endif

#ifdef DEAL_II_WITH_P4EST
      // Initialize p4est and libsc components
#  if DEAL_II_P4EST_VERSION_GTE(2, 0, 0, 0)
#  else
      // This feature is broken in version 2.0.0 for calls to
      // MPI_Comm_create_group (see cburstedde/p4est#30).
      // Disabling it leads to more verbose p4est error messages
      // which should be fine.
      sc_init(MPI_COMM_WORLD, 0, 0, nullptr, SC_LP_SILENT);
#  endif
      p4est_init(nullptr, SC_LP_SILENT);
#endif

      constructor_has_already_run = true;


      // Now also see how many threads we'd like to run
      if (max_num_threads != numbers::invalid_unsigned_int)
        {
          // set maximum number of threads (also respecting the environment
          // variable that the called function evaluates) based on what the
          // user asked
          MultithreadInfo::set_thread_limit(max_num_threads);
        }
      else
        // user wants automatic choice
        {
#ifdef DEAL_II_WITH_MPI
          // we need to figure out how many MPI processes there are on the
          // current node, as well as how many CPU cores we have. for the
          // first task, check what get_hostname() returns and then to an
          // allgather so each processor gets the answer
          //
          // in calculating the length of the string, don't forget the
          // terminating \0 on C-style strings
          const std::string  hostname = Utilities::System::get_hostname();
          const unsigned int max_hostname_size =
            Utilities::MPI::max(hostname.size() + 1, MPI_COMM_WORLD);
          std::vector<char> hostname_array(max_hostname_size);
          std::copy(hostname.c_str(),
                    hostname.c_str() + hostname.size() + 1,
                    hostname_array.begin());

          std::vector<char> all_hostnames(max_hostname_size *
                                          MPI::n_mpi_processes(MPI_COMM_WORLD));
          const int         ierr = MPI_Allgather(hostname_array.data(),
                                         max_hostname_size,
                                         MPI_CHAR,
                                         all_hostnames.data(),
                                         max_hostname_size,
                                         MPI_CHAR,
                                         MPI_COMM_WORLD);
          AssertThrowMPI(ierr);

          // search how often our own hostname appears and the how-manyth
          // instance the current process represents
          unsigned int n_local_processes   = 0;
          unsigned int nth_process_on_host = 0;
          for (unsigned int i = 0; i < MPI::n_mpi_processes(MPI_COMM_WORLD);
               ++i)
            if (std::string(all_hostnames.data() + i * max_hostname_size) ==
                hostname)
              {
                ++n_local_processes;
                if (i <= MPI::this_mpi_process(MPI_COMM_WORLD))
                  ++nth_process_on_host;
              }
          Assert(nth_process_on_host > 0, ExcInternalError());


          // compute how many cores each process gets. if the number does not
          // divide evenly, then we get one more core if we are among the
          // first few processes
          //
          // if the number would be zero, round up to one since every process
          // needs to have at least one thread
          const unsigned int n_threads =
            std::max(MultithreadInfo::n_cores() / n_local_processes +
                       (nth_process_on_host <=
                            MultithreadInfo::n_cores() % n_local_processes ?
                          1 :
                          0),
                     1U);
#else
          const unsigned int n_threads = MultithreadInfo::n_cores();
#endif

          // finally set this number of threads
          MultithreadInfo::set_thread_limit(n_threads);
        }
    }


    MPI_InitFinalize::~MPI_InitFinalize()
    {
      // make memory pool release all PETSc/Trilinos/MPI-based vectors that
      // are no longer used at this point. this is relevant because the static
      // object destructors run for these vectors at the end of the program
      // would run after MPI_Finalize is called, leading to errors

#ifdef DEAL_II_WITH_MPI
      // Start with the deal.II MPI vectors (need to do this before finalizing
      // PETSc because it finalizes MPI).  Delete vectors from the pools:
      GrowingVectorMemory<
        LinearAlgebra::distributed::Vector<double>>::release_unused_memory();
      GrowingVectorMemory<LinearAlgebra::distributed::BlockVector<double>>::
        release_unused_memory();
      GrowingVectorMemory<
        LinearAlgebra::distributed::Vector<float>>::release_unused_memory();
      GrowingVectorMemory<LinearAlgebra::distributed::BlockVector<float>>::
        release_unused_memory();

      // Next with Trilinos:
#  if defined(DEAL_II_WITH_TRILINOS)
      GrowingVectorMemory<
        TrilinosWrappers::MPI::Vector>::release_unused_memory();
      GrowingVectorMemory<
        TrilinosWrappers::MPI::BlockVector>::release_unused_memory();
#  endif
#endif


      // Now deal with PETSc (with or without MPI). Only delete the vectors if
      // finalize hasn't been called yet, otherwise this will lead to errors.
#ifdef DEAL_II_WITH_PETSC
      if ((PetscInitializeCalled == PETSC_TRUE) &&
          (PetscFinalizeCalled == PETSC_FALSE))
        {
          GrowingVectorMemory<
            PETScWrappers::MPI::Vector>::release_unused_memory();
          GrowingVectorMemory<
            PETScWrappers::MPI::BlockVector>::release_unused_memory();

#  ifdef DEAL_II_WITH_SLEPC
          // and now end SLEPc (with PETSc)
          SlepcFinalize();
#  else
          // or just end PETSc.
          PetscFinalize();
#  endif
        }
#endif

// There is a similar issue with CUDA: The destructor of static objects might
// run after the CUDA driver is unloaded. Hence, also release all memory
// related to CUDA vectors.
#ifdef DEAL_II_WITH_CUDA
      GrowingVectorMemory<
        LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>>::
        release_unused_memory();
      GrowingVectorMemory<
        LinearAlgebra::distributed::Vector<float, MemorySpace::CUDA>>::
        release_unused_memory();
#endif

#ifdef DEAL_II_WITH_P4EST
      // now end p4est and libsc
      // Note: p4est has no finalize function
      sc_finalize();
#endif


      // only MPI_Finalize if we are running with MPI. We also need to do this
      // when running PETSc, because we initialize MPI ourselves before
      // calling PetscInitialize
#ifdef DEAL_II_WITH_MPI
      if (job_supports_mpi() == true)
        {
          if (std::uncaught_exception())
            {
              std::cerr
                << "ERROR: Uncaught exception in MPI_InitFinalize on proc "
                << this_mpi_process(MPI_COMM_WORLD)
                << ". Skipping MPI_Finalize() to avoid a deadlock."
                << std::endl;
            }
          else
            {
              const int ierr = MPI_Finalize();
              (void)ierr;
              AssertNothrow(ierr == MPI_SUCCESS, dealii::ExcMPI(ierr));
            }
        }
#endif
    }



    bool
    job_supports_mpi()
    {
#ifdef DEAL_II_WITH_MPI
      int       MPI_has_been_started = 0;
      const int ierr                 = MPI_Initialized(&MPI_has_been_started);
      AssertThrowMPI(ierr);

      return (MPI_has_been_started > 0);
#else
      return false;
#endif
    }

    namespace ComputeIndexOwner
    {
      struct Dictionary
      {
        unsigned int                          dofs_per_process;
        std::vector<unsigned int>             array;
        std::pair<unsigned int, unsigned int> local_range;
        unsigned int                          local_size;
        unsigned int                          size;

        void
        reinit(const IndexSet &owned_dofs, const MPI_Comm &comm)
        {
          unsigned int n_procs = n_mpi_processes(comm);
          unsigned int my_rank = this_mpi_process(comm);

          size               = owned_dofs.size();
          dofs_per_process   = (size + n_procs - 1) / n_procs;
          local_range.first  = std::min(dofs_per_process * my_rank, size);
          local_range.second = std::min(dofs_per_process * (my_rank + 1), size);
          local_size         = local_range.second - local_range.first;

          array.resize(local_size);
        }

        unsigned int
        dof_to_dict_rank(unsigned int i)
        {
          return i / dofs_per_process;
        }
      };

      class StateMachine
      {
      public:
        Dictionary dict;

        StateMachine(MPI_Comm comm)
          : comm(comm)
        {}

        void
        run(std::vector<unsigned int> &array, const IndexSet &ghosted_dofs)
        {
          start_communication_with_dict(array, ghosted_dofs);

          while (!check_own_state())
            process_requests();

          signal_finish();

          while (!check_global_state())
            process_requests();

          end_communication_with_dict(array);
        }

      public:
        static const unsigned int tag_setup    = 11;
        static const unsigned int tag_request  = 12;
        static const unsigned int tag_delivery = 13;

      private:
        MPI_Comm comm;

        unsigned int relevant_procs_size;

        // for sending request
        std::vector<std::vector<std::pair<unsigned int, unsigned int>>>
          send_buffers;

        // for receiving answer to the request
        std::vector<MPI_Request>               recv_requests;
        std::vector<MPI_Status>                recv_statuss;
        std::vector<std::vector<unsigned int>> recv_indices;
        std::vector<std::vector<unsigned int>> recv_buffers;

        // for sending answers
        std::vector<std::vector<unsigned int>>    request_buffers;
        std::vector<std::shared_ptr<MPI_Request>> request_requests;

        // request for barrier
        MPI_Request barrier_request;

        bool
        check_own_state()
        {
          int flag;
          MPI_Testall(relevant_procs_size,
                      &recv_requests[0],
                      &flag,
                      &recv_statuss[0]);
          return flag;
        }

        void
        signal_finish()
        {
          MPI_Ibarrier(comm, &barrier_request);
        }

        bool
        check_global_state()
        {
          MPI_Status status;
          int        flag;
          MPI_Test(&barrier_request, &flag, &status);
          return flag;
        }

        void
        process_requests()
        {
          // check if there is a request pending
          MPI_Status status;
          int        flag;
          MPI_Iprobe(MPI_ANY_SOURCE, tag_request, comm, &flag, &status);

          if (flag) // request is pending
            {
              // get rank of requesting process
              int other_rank = status.MPI_SOURCE;

              // get size of of incoming message
              int number_amount;
              MPI_Get_count(&status, MPI_UNSIGNED, &number_amount);

              // allocate memory for incoming message
              std::vector<std::pair<unsigned int, unsigned int>> buffer_recv;
              buffer_recv.resize(number_amount / 2);
              MPI_Recv(&buffer_recv[0],
                       number_amount,
                       MPI_UNSIGNED,
                       other_rank,
                       tag_request,
                       comm,
                       &status);

              // allocate memory for answer message
              request_buffers.push_back({});
              request_requests.emplace_back(new MPI_Request);

              // process request
              auto &request_buffer = request_buffers.back();
              for (auto interval : buffer_recv)
                for (unsigned int i = interval.first; i < interval.second; i++)
                  request_buffer.push_back(
                    dict.array[i - dict.local_range.first]);

              // start to send answer back
              MPI_Isend(&request_buffer[0],
                        request_buffer.size(),
                        MPI_UNSIGNED,
                        other_rank,
                        tag_delivery,
                        comm,
                        &*request_requests.back());
            }
        }

        void
        start_communication_with_dict(std::vector<unsigned int> &array,
                                      const IndexSet &           ghosted_dofs)
        {
          unsigned int my_rank = this_mpi_process(comm);

          // 1) collect relevant processes and process local dict entries
          std::map<unsigned int, unsigned int> relevant_procs_map;
          {
            std::set<unsigned int> relevant_procs;
            {
              unsigned int c = 0;
              for (auto i : ghosted_dofs)
                {
                  unsigned int other_rank = dict.dof_to_dict_rank(i);
                  if (other_rank == my_rank)
                    array[c] = dict.array[i - dict.local_range.first];
                  else
                    relevant_procs.insert(other_rank);
                  c++;
                }
            }

            {
              unsigned int c = 0;
              for (auto i : relevant_procs)
                relevant_procs_map[i] = c++;
            }
          }

          // 2) allocate memory
          relevant_procs_size = relevant_procs_map.size();

          recv_buffers.resize(relevant_procs_size);
          recv_indices.resize(relevant_procs_size);
          recv_requests.resize(relevant_procs_size);
          recv_statuss.resize(relevant_procs_size);

          send_buffers.resize(relevant_procs_size);

          {
            // 3) collect indices for each process
            std::vector<std::vector<unsigned int>> temp(relevant_procs_size);

            {
              unsigned int c = 0;
              for (auto i : ghosted_dofs)
                {
                  unsigned int other_rank = dict.dof_to_dict_rank(i);
                  if (other_rank != my_rank)
                    {
                      recv_indices[relevant_procs_map[other_rank]].push_back(c);
                      temp[relevant_procs_map[other_rank]].push_back(i);
                    }
                  c++;
                }
            }

            // 4) send and receive
            for (auto rank_pair : relevant_procs_map)
              {
                const unsigned int rank  = rank_pair.first;
                const unsigned int index = relevant_procs_map[rank];

                // create index set and compress data to be sent
                auto &   indices_i = temp[index];
                IndexSet is(dict.size);
                is.add_indices(indices_i.begin(), indices_i.end());
                is.compress();

                // translate index set to a list of pairs
                auto &send_buffer = send_buffers[index];
                for (auto interval = is.begin_intervals();
                     interval != is.end_intervals();
                     interval++)
                  send_buffer.push_back(std::pair<unsigned int, unsigned int>(
                    *interval->begin(), interval->last() + 1));

                // start to send data
                MPI_Request request;
                MPI_Isend(&send_buffer[0],
                          send_buffer.size() * 2,
                          MPI_UNSIGNED,
                          rank,
                          tag_request,
                          comm,
                          &request);

                // start to receive data
                auto &recv_buffer = recv_buffers[index];
                recv_buffer.resize(indices_i.size());
                MPI_Irecv(&recv_buffer[0],
                          recv_buffer.size(),
                          MPI_UNSIGNED,
                          rank,
                          tag_delivery,
                          comm,
                          &recv_requests[index]);
              }
          }
        }

        void
        end_communication_with_dict(std::vector<unsigned int> &array)
        {
          for (unsigned i = 0; i < relevant_procs_size; i++)
            for (unsigned j = 0; j < recv_indices[i].size(); j++)
              array[recv_indices[i][j]] = recv_buffers[i][j];
        }
      };

      void
      setup_dictionary(Dictionary &    dict,
                       const IndexSet &owned_dofs,
                       const MPI_Comm &comm)
      {
        unsigned int my_rank = this_mpi_process(comm);

        // 1) setup dictionary and allocate memory
        dict.reinit(owned_dofs, comm);

        unsigned int                         dic_local_rececived = 0;
        std::map<unsigned int, unsigned int> relevant_procs_map;

        // 2) collect relevant processes and process local dict entries
        {
          std::set<unsigned int> relevant_procs;
          for (auto i : owned_dofs)
            {
              unsigned int other_rank = dict.dof_to_dict_rank(i);
              if (other_rank == my_rank)
                {
                  dict.array[i - dict.local_range.first] = my_rank;
                  dic_local_rececived++;
                }
              else
                relevant_procs.insert(other_rank);
            }

          {
            unsigned int c = 0;
            for (auto i : relevant_procs)
              relevant_procs_map[i] = c++;
          }
        }

        const unsigned int relevant_procs_size = relevant_procs_map.size();
        std::vector<std::vector<std::pair<unsigned int, unsigned int>>> buffers(
          relevant_procs_size);
        std::vector<MPI_Request> request(relevant_procs_size);

        // 3) send messages with local dofs to the right dict process
        {
          std::vector<std::vector<unsigned int>> temp(relevant_procs_size);

          // collect dofs of each dict process
          for (auto i : owned_dofs)
            {
              unsigned int other_rank = dict.dof_to_dict_rank(i);
              if (other_rank != my_rank)
                temp[relevant_procs_map[other_rank]].push_back(i);
            }

          // send dofs to each process
          for (auto rank_pair : relevant_procs_map)
            {
              int rank  = rank_pair.first;
              int index = rank_pair.second;

              // create index set and compress data to be sent
              auto &   indices_i = temp[index];
              IndexSet is(dict.size);
              is.add_indices(indices_i.begin(), indices_i.end());
              is.compress();

              // translate index set to a list of pairs
              auto &buffer = buffers[index];
              for (auto interval = is.begin_intervals();
                   interval != is.end_intervals();
                   interval++)
                buffer.push_back({*interval->begin(), interval->last() + 1});

              // send data
              MPI_Isend(&buffer[0],
                        buffer.size() * 2,
                        MPI_UNSIGNED,
                        rank,
                        StateMachine::tag_setup,
                        comm,
                        &request[index]);
            }
        }


        // 4) receive messages until all dofs in dict are processed
        while (dict.local_size != dic_local_rececived)
          {
            // wait for an incoming massage
            MPI_Status status;
            MPI_Probe(MPI_ANY_SOURCE, StateMachine::tag_setup, comm, &status);

            // retrieve size of incoming massage
            int number_amount;
            MPI_Get_count(&status, MPI_UNSIGNED, &number_amount);

            unsigned int other_rank = status.MPI_SOURCE;

            // receive massage
            std::vector<std::pair<unsigned int, unsigned int>> buffer(
              number_amount / 2);
            MPI_Recv(&buffer[0],
                     number_amount,
                     MPI_UNSIGNED,
                     status.MPI_SOURCE,
                     StateMachine::tag_setup,
                     comm,
                     &status);

            // process message: loop over all intervals
            for (auto interval : buffer)
              for (unsigned int i = interval.first; i < interval.second; i++)
                {
                  dict.array[i - dict.local_range.first] = other_rank;
                  dic_local_rececived++;
                }
          }

        // 5) make sure that all messages have been sent
        std::vector<MPI_Status> status(relevant_procs_size);
        MPI_Waitall(relevant_procs_size, &request[0], &status[0]);
      }


    } // namespace ComputeIndexOwner


    std::vector<unsigned int>
    compute_index_owner(const IndexSet &owned_dofs,
                        const IndexSet &ghosted_dofs,
                        const MPI_Comm &comm)
    {
      using namespace ComputeIndexOwner;

      StateMachine state_machine(comm);

      // Step 1: setup dictionary
      setup_dictionary(state_machine.dict, owned_dofs, comm);

      // Step 2: read dictionary
      std::vector<unsigned int> array(ghosted_dofs.n_elements());
      state_machine.run(array, ghosted_dofs);

      return array;
    }

#include "mpi.inst"
  } // end of namespace MPI
} // end of namespace Utilities

DEAL_II_NAMESPACE_CLOSE
