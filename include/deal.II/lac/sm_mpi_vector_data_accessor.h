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

#ifndef dealii_sm_mpi_vector_dataccessor_h
#define dealii_sm_mpi_vector_dataccessor_h

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace SharedMPI
  {
    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    class DataAccessor
    {
    public:
      static const types::global_dof_index dofs_per_cell =
        Utilities::pow(degree + 1, dim);
      static const types::global_dof_index dofs_per_face =
        Utilities::pow(degree + 1, dim - 1);

      static const int v_len = VectorizedArrayType::n_array_elements;

      template <unsigned int stride = 1>
      inline DEAL_II_ALWAYS_INLINE //
        static void
        gather(const Number *src, Number *dst);

      inline DEAL_II_ALWAYS_INLINE //
        static void
        gatherv(const std::array<Number *, v_len> &srcs, double *dst);

      template <unsigned int stride = 1, bool do_add = false>
      inline DEAL_II_ALWAYS_INLINE //
        static void
        scatter(Number *dst, const double *src);

      template <bool do_add = false>
      inline DEAL_II_ALWAYS_INLINE //
        static void
        scatterv(const std::array<Number *, v_len> &gids, const double *src);


      template <unsigned int stride = 1>
      static void
      gather_face(const Number *src, int surface, Number *dst, bool type);

      template <unsigned int stride = 1>
      static void
      gatherv_face(const std::array<Number *, v_len> &srcs,
                   int                                surface,
                   Number *                           dst,
                   bool                               type);

      template <unsigned int stride = 1, bool do_add = false>
      static void
      scatter_face(Number *dst, int surface, const Number *src, bool type);

      template <unsigned int stride = 1, bool do_add = false>
      static void
      scatterv_face(std::array<Number *, v_len> &dst,
                    int                          surface,
                    const Number *               src,
                    bool                         type);


      template <int surface, unsigned int stride = 1>
      inline DEAL_II_ALWAYS_INLINE //
        static void
        gather_face_internal_direction(double *__restrict dst,
                                       const double *__restrict src);

      template <int surface, unsigned int stride = 1>
      inline DEAL_II_ALWAYS_INLINE //
        static void
        gatherv_face_internal_direction(double *                           dst,
                                        const std::array<Number *, v_len> &src);

      template <unsigned int stride = 1>
      static void
      gather_face_internal(const Number *temp, int surface, Number *dst);

      template <unsigned int stride = 1>
      static void
      gatherv_face_internal(const std::array<Number *, v_len> &temp,
                            int                                surface,
                            Number *__restrict dst);

      template <int surface, unsigned int stride = 1, bool do_add = false>
      inline DEAL_II_ALWAYS_INLINE //
        static void
        scatter_face_internal_direction(double *__restrict dst,
                                        const double *__restrict src);

      template <int surface, unsigned int stride = 1, bool do_add = false>
      inline DEAL_II_ALWAYS_INLINE //
        static void
        scatterv_face_internal_direction(std::array<Number *, v_len> &dst,
                                         const double *__restrict src);

      template <unsigned int stride = 1, bool do_add = false>
      static void
      scatter_face_internal(Number *temp, int surface, const Number *dst);

      template <unsigned int stride = 1, bool do_add = false>
      static void
      scatterv_face_internal(std::array<Number *, v_len> &temp,
                             int                          surface,
                             const Number *               dst);
    };

  } // namespace SharedMPI
} // namespace LinearAlgebra


DEAL_II_NAMESPACE_CLOSE

#include <deal.II/lac/sm_mpi_vector_data_accessor.cpp>


#endif