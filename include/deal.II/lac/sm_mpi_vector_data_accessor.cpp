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

#include <deal.II/lac/sm_mpi_vector_data_accessor.h>

#include <x86intrin.h>

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace SharedMPI
  {
    template <unsigned int stride_1,
              unsigned int stride_2,
              typename Number,
              bool do_add = false>
    inline DEAL_II_ALWAYS_INLINE //
      static void
      memcpy_strided(Number *__restrict dst,
                     const Number *__restrict src,
                     const unsigned int size)
    {
      for (unsigned int i = 0; i < size; i++)
        if (do_add)
          dst[stride_1 * i] += src[stride_2 * i];
        else
          dst[stride_1 * i] = src[stride_2 * i];
    }

    template <int v_len, typename Number, bool do_add = false>
    inline DEAL_II_ALWAYS_INLINE //
      static void
      memcpy_strided_v_gather(Number *__restrict dst,
                              const std::array<Number *, v_len> &src,
                              const unsigned int                 size)
    {
      if (v_len == 1)
        {
          const Number *__restrict src0 = src[0];

          for (unsigned int i = 0; i < size; i++)
            if (do_add)
              {
                dst[v_len * i + 0] += src0[i];
              }
            else
              {
                dst[v_len * i + 0] = src0[i];
              }
        }
      else if (v_len == 2)
        {
          const Number *__restrict src0 = src[0];
          const Number *__restrict src1 = src[1];

          for (unsigned int i = 0; i < size; i++)
            if (do_add)
              {
                dst[v_len * i + 0] += src0[i];
                dst[v_len * i + 1] += src1[i];
              }
            else
              {
                dst[v_len * i + 0] = src0[i];
                dst[v_len * i + 1] = src1[i];
              }
        }
      else if (v_len == 4)
        {
          const Number *__restrict src0 = src[0];
          const Number *__restrict src1 = src[1];
          const Number *__restrict src2 = src[2];
          const Number *__restrict src3 = src[3];

          for (unsigned int i = 0; i < size; i++)
            if (do_add)
              {
                dst[v_len * i + 0] += src0[i];
                dst[v_len * i + 1] += src1[i];
                dst[v_len * i + 2] += src2[i];
                dst[v_len * i + 3] += src3[i];
              }
            else
              {
                dst[v_len * i + 0] = src0[i];
                dst[v_len * i + 1] = src1[i];
                dst[v_len * i + 2] = src2[i];
                dst[v_len * i + 3] = src3[i];
              }
        }
      else if (v_len == 8)
        {
          const Number *__restrict src0 = src[0];
          const Number *__restrict src1 = src[1];
          const Number *__restrict src2 = src[2];
          const Number *__restrict src3 = src[3];
          const Number *__restrict src4 = src[4];
          const Number *__restrict src5 = src[5];
          const Number *__restrict src6 = src[6];
          const Number *__restrict src7 = src[7];

          for (unsigned int i = 0; i < size; i++)
            if (do_add)
              {
                dst[v_len * i + 0] += src0[i];
                dst[v_len * i + 1] += src1[i];
                dst[v_len * i + 2] += src2[i];
                dst[v_len * i + 3] += src3[i];
                dst[v_len * i + 4] += src4[i];
                dst[v_len * i + 5] += src5[i];
                dst[v_len * i + 6] += src6[i];
                dst[v_len * i + 7] += src7[i];
              }
            else
              {
                dst[v_len * i + 0] = src0[i];
                dst[v_len * i + 1] = src1[i];
                dst[v_len * i + 2] = src2[i];
                dst[v_len * i + 3] = src3[i];
                dst[v_len * i + 4] = src4[i];
                dst[v_len * i + 5] = src5[i];
                dst[v_len * i + 6] = src6[i];
                dst[v_len * i + 7] = src7[i];

                //        Number temp [v_len];
                //        temp[0] = src0[i];
                //        temp[1] = src1[i];
                //        temp[2] = src2[i];
                //        temp[3] = src3[i];
                //        temp[4] = src4[i];
                //        temp[5] = src5[i];
                //        temp[6] = src6[i];
                //        temp[7] = src7[i];
                //        (*reinterpret_cast<__m512d*>(&dst[v_len * i])) =
                //        _mm512_load_pd(temp);
              }
        }
      else
        AssertThrow(false, ExcMessage("Not instantiated!"));
    }

    template <int v_len, typename Number, bool do_add = false>
    inline DEAL_II_ALWAYS_INLINE //
      static void
      memcpy_strided_v_scatter(std::array<Number *, v_len> dst,
                               const Number *__restrict src,
                               const unsigned int size)
    {
      if (v_len == 1)
        {
          Number *__restrict dst0 = dst[0];

          for (unsigned int i = 0; i < size; i++)
            if (do_add)
              {
                dst0[i] += src[v_len * i + 0];
              }
            else
              {
                dst0[i] = src[v_len * i + 0];
              }
        }
      else if (v_len == 2)
        {
          Number *__restrict dst0 = dst[0];
          Number *__restrict dst1 = dst[1];

          for (unsigned int i = 0; i < size; i++)
            if (do_add)
              {
                dst0[i] += src[v_len * i + 0];
                dst1[i] += src[v_len * i + 1];
              }
            else
              {
                dst0[i] = src[v_len * i + 0];
                dst1[i] = src[v_len * i + 1];
              }
        }
      else if (v_len == 4)
        {
          Number *__restrict dst0 = dst[0];
          Number *__restrict dst1 = dst[1];
          Number *__restrict dst2 = dst[2];
          Number *__restrict dst3 = dst[3];

          for (unsigned int i = 0; i < size; i++)
            if (do_add)
              {
                dst0[i] += src[v_len * i + 0];
                dst1[i] += src[v_len * i + 1];
                dst2[i] += src[v_len * i + 2];
                dst3[i] += src[v_len * i + 3];
              }
            else
              {
                dst0[i] = src[v_len * i + 0];
                dst1[i] = src[v_len * i + 1];
                dst2[i] = src[v_len * i + 2];
                dst3[i] = src[v_len * i + 3];
              }
        }
      else if (v_len == 8)
        {
          Number *__restrict dst0 = dst[0];
          Number *__restrict dst1 = dst[1];
          Number *__restrict dst2 = dst[2];
          Number *__restrict dst3 = dst[3];
          Number *__restrict dst4 = dst[4];
          Number *__restrict dst5 = dst[5];
          Number *__restrict dst6 = dst[6];
          Number *__restrict dst7 = dst[7];

          for (unsigned int i = 0; i < size; i++)
            if (do_add)
              {
                dst0[i] += src[v_len * i + 0];
                dst1[i] += src[v_len * i + 1];
                dst2[i] += src[v_len * i + 2];
                dst3[i] += src[v_len * i + 3];
                dst4[i] += src[v_len * i + 4];
                dst5[i] += src[v_len * i + 5];
                dst6[i] += src[v_len * i + 6];
                dst7[i] += src[v_len * i + 7];
              }
            else
              {
                dst0[i] = src[v_len * i + 0];
                dst1[i] = src[v_len * i + 1];
                dst2[i] = src[v_len * i + 2];
                dst3[i] = src[v_len * i + 3];
                dst4[i] = src[v_len * i + 4];
                dst5[i] = src[v_len * i + 5];
                dst6[i] = src[v_len * i + 6];
                dst7[i] = src[v_len * i + 7];
              }
        }
      else
        AssertThrow(false, ExcMessage("Not instantiated!"));
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <unsigned int stride>
    inline DEAL_II_ALWAYS_INLINE //
      void
      DataAccessor<dim, degree, Number, VectorizedArrayType>::gather(
        const Number *src,
        Number *      dst)
    {
      memcpy_strided<stride, 1, Number, false>(dst, src, dofs_per_cell);
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>

    inline DEAL_II_ALWAYS_INLINE //
      void
      DataAccessor<dim, degree, Number, VectorizedArrayType>::gatherv(
        const std::array<Number *, v_len> &srcs,
        double *                           dst)
    {
      memcpy_strided_v_gather<v_len, Number, false>(dst, srcs, dofs_per_cell);
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <unsigned int stride, bool do_add>
    inline DEAL_II_ALWAYS_INLINE //
      void
      DataAccessor<dim, degree, Number, VectorizedArrayType>::scatter(
        Number *      dst,
        const double *src)
    {
      memcpy_strided<1, stride, Number, do_add>(dst, src, dofs_per_cell);
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <bool do_add>
    inline DEAL_II_ALWAYS_INLINE //
      void
      DataAccessor<dim, degree, Number, VectorizedArrayType>::scatterv(
        const std::array<Number *, v_len> &dsts,
        const double *                     src)
    {
      memcpy_strided_v_scatter<v_len, Number, do_add>(dsts, src, dofs_per_cell);
    }



    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <int surface, unsigned int stride>
    inline DEAL_II_ALWAYS_INLINE //
      void
      DataAccessor<dim, degree, Number, VectorizedArrayType>::
        gather_face_internal_direction(double *__restrict dst,
                                       const double *__restrict src)
    {
      static const int points = degree + 1;

      // create indices for one surfaces
      static const int d = surface / 2; // direction
      static const int s = surface % 2; // left or right surface

      static const int b1 = Utilities::pow(points, d + 1);
      static const int b2 =
        (s == 0 ? 0 : (points - 1)) * Utilities::pow(points, d);

      static const unsigned int r1 = Utilities::pow(points, dim - d - 1);
      static const unsigned int r2 = Utilities::pow(points, d);

      // collapsed iteration
      for (unsigned int i = 0, k = 0; i < r1; i++)
        for (unsigned int j = 0; j < r2; j++)
          dst[stride * k++] = src[i * b1 + b2 + j];
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <int surface, unsigned int v_len>
    inline DEAL_II_ALWAYS_INLINE //
      void
      DataAccessor<dim, degree, Number, VectorizedArrayType>::
        gatherv_face_internal_direction(double *__restrict dst,
                                        const std::array<Number *, v_len> &src)
    {
      static const int points = degree + 1;

      // create indices for one surfaces
      static const int d = surface / 2; // direction
      static const int s = surface % 2; // left or right surface

      static const int b1 = Utilities::pow(points, d + 1);
      static const int b2 =
        (s == 0 ? 0 : (points - 1)) * Utilities::pow(points, d);

      static const unsigned int r1 = Utilities::pow(points, dim - d - 1);
      static const unsigned int r2 = Utilities::pow(points, d);

      if (v_len == 1)
        {
          const Number *__restrict src0 = src[0];

          // collapsed iteration
          for (unsigned int i = 0, k = 0; i < r1; i++)
            for (unsigned int j = 0; j < r2; j++, k++)
              {
                dst[v_len * k + 0] = src0[i * b1 + b2 + j];
              }
        }
      else if (v_len == 2)
        {
          const Number *__restrict src0 = src[0];
          const Number *__restrict src1 = src[1];

          // collapsed iteration
          for (unsigned int i = 0, k = 0; i < r1; i++)
            for (unsigned int j = 0; j < r2; j++, k++)
              {
                dst[v_len * k + 0] = src0[i * b1 + b2 + j];
                dst[v_len * k + 1] = src1[i * b1 + b2 + j];
              }
        }
      else if (v_len == 4)
        {
          const Number *__restrict src0 = src[0];
          const Number *__restrict src1 = src[1];
          const Number *__restrict src2 = src[2];
          const Number *__restrict src3 = src[3];

          // collapsed iteration
          for (unsigned int i = 0, k = 0; i < r1; i++)
            for (unsigned int j = 0; j < r2; j++, k++)
              {
                dst[v_len * k + 0] = src0[i * b1 + b2 + j];
                dst[v_len * k + 1] = src1[i * b1 + b2 + j];
                dst[v_len * k + 2] = src2[i * b1 + b2 + j];
                dst[v_len * k + 3] = src3[i * b1 + b2 + j];
              }
        }
      else if (v_len == 8)
        {
          const Number *__restrict src0 = src[0];
          const Number *__restrict src1 = src[1];
          const Number *__restrict src2 = src[2];
          const Number *__restrict src3 = src[3];
          const Number *__restrict src4 = src[4];
          const Number *__restrict src5 = src[5];
          const Number *__restrict src6 = src[6];
          const Number *__restrict src7 = src[7];

          // collapsed iteration
          for (unsigned int i = 0, k = 0; i < r1; i++)
            for (unsigned int j = 0; j < r2; j++, k++)
              {
                dst[v_len * k + 0] = src0[i * b1 + b2 + j];
                dst[v_len * k + 1] = src1[i * b1 + b2 + j];
                dst[v_len * k + 2] = src2[i * b1 + b2 + j];
                dst[v_len * k + 3] = src3[i * b1 + b2 + j];
                dst[v_len * k + 4] = src4[i * b1 + b2 + j];
                dst[v_len * k + 5] = src5[i * b1 + b2 + j];
                dst[v_len * k + 6] = src6[i * b1 + b2 + j];
                dst[v_len * k + 7] = src7[i * b1 + b2 + j];
              }
        }
      else
        AssertThrow(false, ExcMessage("Not instantiated!"));
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <unsigned int stride>
    void
    DataAccessor<dim, degree, Number, VectorizedArrayType>::
      gather_face_internal(const Number *temp, int surface, Number *dst)
    {
      if (dim >= 1 && surface == 0)
        gather_face_internal_direction<0, stride>(dst, temp);
      else if (dim >= 1 && surface == 1)
        gather_face_internal_direction<1, stride>(dst, temp);
      else if (dim >= 2 && surface == 2)
        gather_face_internal_direction<2, stride>(dst, temp);
      else if (dim >= 2 && surface == 3)
        gather_face_internal_direction<3, stride>(dst, temp);
      else if (dim >= 3 && surface == 4)
        gather_face_internal_direction<4, stride>(dst, temp);
      else if (dim >= 3 && surface == 5)
        gather_face_internal_direction<5, stride>(dst, temp);
      else if (dim >= 4 && surface == 6)
        gather_face_internal_direction<6, stride>(dst, temp);
      else if (dim >= 4 && surface == 7)
        gather_face_internal_direction<7, stride>(dst, temp);
      else if (dim >= 5 && surface == 8)
        gather_face_internal_direction<8, stride>(dst, temp);
      else if (dim >= 5 && surface == 9)
        gather_face_internal_direction<9, stride>(dst, temp);
      else if (dim >= 6 && surface == 10)
        gather_face_internal_direction<10, stride>(dst, temp);
      else if (dim >= 6 && surface == 11)
        gather_face_internal_direction<11, stride>(dst, temp);
      else
        AssertThrow(false, ExcMessage("Not instantiated!"));
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <unsigned int stride>
    void
    DataAccessor<dim, degree, Number, VectorizedArrayType>::
      gatherv_face_internal(const std::array<Number *, v_len> &temp,
                            int                                surface,
                            Number *                           dst)
    {
      if (dim >= 1 && surface == 0)
        gatherv_face_internal_direction<0, stride>(dst, temp);
      else if (dim >= 1 && surface == 1)
        gatherv_face_internal_direction<1, stride>(dst, temp);
      else if (dim >= 2 && surface == 2)
        gatherv_face_internal_direction<2, stride>(dst, temp);
      else if (dim >= 2 && surface == 3)
        gatherv_face_internal_direction<3, stride>(dst, temp);
      else if (dim >= 3 && surface == 4)
        gatherv_face_internal_direction<4, stride>(dst, temp);
      else if (dim >= 3 && surface == 5)
        gatherv_face_internal_direction<5, stride>(dst, temp);
      else if (dim >= 4 && surface == 6)
        gatherv_face_internal_direction<6, stride>(dst, temp);
      else if (dim >= 4 && surface == 7)
        gatherv_face_internal_direction<7, stride>(dst, temp);
      else if (dim >= 5 && surface == 8)
        gatherv_face_internal_direction<8, stride>(dst, temp);
      else if (dim >= 5 && surface == 9)
        gatherv_face_internal_direction<9, stride>(dst, temp);
      else if (dim >= 6 && surface == 10)
        gatherv_face_internal_direction<10, stride>(dst, temp);
      else if (dim >= 6 && surface == 11)
        gatherv_face_internal_direction<11, stride>(dst, temp);
      else
        AssertThrow(false, ExcMessage("Not instantiated!"));
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <int surface, unsigned int stride, bool do_add>
    inline DEAL_II_ALWAYS_INLINE //
      void
      DataAccessor<dim, degree, Number, VectorizedArrayType>::
        scatter_face_internal_direction(double *__restrict dst,
                                        const double *__restrict src)
    {
      static const int points = degree + 1;

      // create indices for one surfaces
      static const int d = surface / 2; // direction
      static const int s = surface % 2; // left or right surface

      static const int b1 = Utilities::pow(points, d + 1);
      static const int b2 =
        (s == 0 ? 0 : (points - 1)) * Utilities::pow(points, d);

      static const unsigned int r1 = Utilities::pow(points, dim - d - 1);
      static const unsigned int r2 = Utilities::pow(points, d);

      // collapsed iteration
      for (unsigned int i = 0, k = 0; i < r1; i++)
        for (unsigned int j = 0; j < r2; j++)
          if (do_add)
            dst[i * b1 + b2 + j] += src[stride * k++];
          else
            dst[i * b1 + b2 + j] = src[stride * k++];
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <int surface, unsigned int v_len, bool do_add>
    inline DEAL_II_ALWAYS_INLINE //
      void
      DataAccessor<dim, degree, Number, VectorizedArrayType>::
        scatterv_face_internal_direction(std::array<Number *, v_len> &dst,
                                         const double *__restrict src)
    {
      static const int points = degree + 1;

      // create indices for one surfaces
      static const int d = surface / 2; // direction
      static const int s = surface % 2; // left or right surface

      static const int b1 = Utilities::pow(points, d + 1);
      static const int b2 =
        (s == 0 ? 0 : (points - 1)) * Utilities::pow(points, d);

      static const unsigned int r1 = Utilities::pow(points, dim - d - 1);
      static const unsigned int r2 = Utilities::pow(points, d);

      if (v_len == 1)
        {
          Number *__restrict dst0 = dst[0];
          // collapsed iteration
          for (unsigned int i = 0, k = 0; i < r1; i++)
            for (unsigned int j = 0; j < r2; j++, k++)
              if (do_add)
                {
                  dst0[i * b1 + b2 + j] += src[v_len * k + 0];
                }
              else
                {
                  dst0[i * b1 + b2 + j] = src[v_len * k + 0];
                }
        }
      else if (v_len == 2)
        {
          Number *__restrict dst0 = dst[0];
          Number *__restrict dst1 = dst[1];
          // collapsed iteration
          for (unsigned int i = 0, k = 0; i < r1; i++)
            for (unsigned int j = 0; j < r2; j++, k++)
              if (do_add)
                {
                  dst0[i * b1 + b2 + j] += src[v_len * k + 0];
                  dst1[i * b1 + b2 + j] += src[v_len * k + 1];
                }
              else
                {
                  dst0[i * b1 + b2 + j] = src[v_len * k + 0];
                  dst1[i * b1 + b2 + j] = src[v_len * k + 1];
                }
        }
      else if (v_len == 4)
        {
          Number *__restrict dst0 = dst[0];
          Number *__restrict dst1 = dst[1];
          Number *__restrict dst2 = dst[2];
          Number *__restrict dst3 = dst[3];
          // collapsed iteration
          for (unsigned int i = 0, k = 0; i < r1; i++)
            for (unsigned int j = 0; j < r2; j++, k++)
              if (do_add)
                {
                  dst0[i * b1 + b2 + j] += src[v_len * k + 0];
                  dst1[i * b1 + b2 + j] += src[v_len * k + 1];
                  dst2[i * b1 + b2 + j] += src[v_len * k + 2];
                  dst3[i * b1 + b2 + j] += src[v_len * k + 3];
                }
              else
                {
                  dst0[i * b1 + b2 + j] = src[v_len * k + 0];
                  dst1[i * b1 + b2 + j] = src[v_len * k + 1];
                  dst2[i * b1 + b2 + j] = src[v_len * k + 2];
                  dst3[i * b1 + b2 + j] = src[v_len * k + 3];
                }
        }
      else if (v_len == 8)
        {
          Number *__restrict dst0 = dst[0];
          Number *__restrict dst1 = dst[1];
          Number *__restrict dst2 = dst[2];
          Number *__restrict dst3 = dst[3];
          Number *__restrict dst4 = dst[4];
          Number *__restrict dst5 = dst[5];
          Number *__restrict dst6 = dst[6];
          Number *__restrict dst7 = dst[7];
          // collapsed iteration
          for (unsigned int i = 0, k = 0; i < r1; i++)
            for (unsigned int j = 0; j < r2; j++, k++)
              if (do_add)
                {
                  dst0[i * b1 + b2 + j] += src[v_len * k + 0];
                  dst1[i * b1 + b2 + j] += src[v_len * k + 1];
                  dst2[i * b1 + b2 + j] += src[v_len * k + 2];
                  dst3[i * b1 + b2 + j] += src[v_len * k + 3];
                  dst4[i * b1 + b2 + j] += src[v_len * k + 4];
                  dst5[i * b1 + b2 + j] += src[v_len * k + 5];
                  dst6[i * b1 + b2 + j] += src[v_len * k + 6];
                  dst7[i * b1 + b2 + j] += src[v_len * k + 7];
                }
              else
                {
                  dst0[i * b1 + b2 + j] = src[v_len * k + 0];
                  dst1[i * b1 + b2 + j] = src[v_len * k + 1];
                  dst2[i * b1 + b2 + j] = src[v_len * k + 2];
                  dst3[i * b1 + b2 + j] = src[v_len * k + 3];
                  dst4[i * b1 + b2 + j] = src[v_len * k + 4];
                  dst5[i * b1 + b2 + j] = src[v_len * k + 5];
                  dst6[i * b1 + b2 + j] = src[v_len * k + 6];
                  dst7[i * b1 + b2 + j] = src[v_len * k + 7];
                }
        }
      else
        AssertThrow(false, ExcMessage("Not instantiated!"));
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <unsigned int stride, bool do_add>
    void
    DataAccessor<dim, degree, Number, VectorizedArrayType>::
      scatter_face_internal(Number *temp, int surface, const Number *src)
    {
      if (dim >= 1 && surface == 0)
        scatter_face_internal_direction<0, stride, do_add>(temp, src);
      else if (dim >= 1 && surface == 1)
        scatter_face_internal_direction<1, stride, do_add>(temp, src);
      else if (dim >= 2 && surface == 2)
        scatter_face_internal_direction<2, stride, do_add>(temp, src);
      else if (dim >= 2 && surface == 3)
        scatter_face_internal_direction<3, stride, do_add>(temp, src);
      else if (dim >= 3 && surface == 4)
        scatter_face_internal_direction<4, stride, do_add>(temp, src);
      else if (dim >= 3 && surface == 5)
        scatter_face_internal_direction<5, stride, do_add>(temp, src);
      else if (dim >= 4 && surface == 6)
        scatter_face_internal_direction<6, stride, do_add>(temp, src);
      else if (dim >= 4 && surface == 7)
        scatter_face_internal_direction<7, stride, do_add>(temp, src);
      else if (dim >= 5 && surface == 8)
        scatter_face_internal_direction<8, stride, do_add>(temp, src);
      else if (dim >= 5 && surface == 9)
        scatter_face_internal_direction<9, stride, do_add>(temp, src);
      else if (dim >= 6 && surface == 10)
        scatter_face_internal_direction<10, stride, do_add>(temp, src);
      else if (dim >= 6 && surface == 11)
        scatter_face_internal_direction<11, stride, do_add>(temp, src);
      else
        AssertThrow(false, ExcMessage("Not instantiated!"));
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <unsigned int stride, bool do_add>
    void
    DataAccessor<dim, degree, Number, VectorizedArrayType>::
      scatterv_face_internal(std::array<Number *, v_len> &temp,
                             int                          surface,
                             const Number *               src)
    {
      if (dim >= 1 && surface == 0)
        scatterv_face_internal_direction<0, stride, do_add>(temp, src);
      else if (dim >= 1 && surface == 1)
        scatterv_face_internal_direction<1, stride, do_add>(temp, src);
      else if (dim >= 2 && surface == 2)
        scatterv_face_internal_direction<2, stride, do_add>(temp, src);
      else if (dim >= 2 && surface == 3)
        scatterv_face_internal_direction<3, stride, do_add>(temp, src);
      else if (dim >= 3 && surface == 4)
        scatterv_face_internal_direction<4, stride, do_add>(temp, src);
      else if (dim >= 3 && surface == 5)
        scatterv_face_internal_direction<5, stride, do_add>(temp, src);
      else if (dim >= 4 && surface == 6)
        scatterv_face_internal_direction<6, stride, do_add>(temp, src);
      else if (dim >= 4 && surface == 7)
        scatterv_face_internal_direction<7, stride, do_add>(temp, src);
      else if (dim >= 5 && surface == 8)
        scatterv_face_internal_direction<8, stride, do_add>(temp, src);
      else if (dim >= 5 && surface == 9)
        scatterv_face_internal_direction<9, stride, do_add>(temp, src);
      else if (dim >= 6 && surface == 10)
        scatterv_face_internal_direction<10, stride, do_add>(temp, src);
      else if (dim >= 6 && surface == 11)
        scatterv_face_internal_direction<11, stride, do_add>(temp, src);
      else
        AssertThrow(false, ExcMessage("Not instantiated!"));
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <unsigned int stride>
    void
    DataAccessor<dim, degree, Number, VectorizedArrayType>::gather_face(
      const Number *src,
      int           surface,
      Number *      dst,
      bool          type)
    {
      if (type) // case 1: read from buffers
        memcpy_strided<stride, 1, Number, false>(dst, src, dofs_per_face);
      else // case 2: read from shared memory
        gather_face_internal<stride>(src, surface, dst);
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <unsigned int stride>
    void
    DataAccessor<dim, degree, Number, VectorizedArrayType>::gatherv_face(
      const std::array<Number *, v_len> &srcs,
      int                                surface,
      Number *                           dst,
      bool                               type)
    {
      if (type) // case 1: read from buffers
        memcpy_strided_v_gather<stride, Number, false>(dst,
                                                       srcs,
                                                       dofs_per_face);
      else // case 2: read from shared memory
        gatherv_face_internal<stride>(srcs, surface, dst);
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <unsigned int stride, bool do_add>
    void
    DataAccessor<dim, degree, Number, VectorizedArrayType>::scatter_face(
      Number *      dst,
      int           surface,
      const Number *src,
      bool          type)
    {
      if (type) // case 1: write to buffers
        memcpy_strided<1, stride, Number, false>(dst, src, dofs_per_face);
      else // case 2: write to shared memory
        scatter_face_internal<stride, do_add>(dst, surface, src);
    }

    template <int dim,
              int degree,
              typename Number,
              typename VectorizedArrayType>
    template <unsigned int stride, bool do_add>
    void
    DataAccessor<dim, degree, Number, VectorizedArrayType>::scatterv_face(
      std::array<Number *, v_len> &dsts,
      int                          surface,
      const Number *               src,
      bool                         type)
    {
      if (type) // case 1: write to buffers
        memcpy_strided_v_scatter<stride, Number, false>(dsts,
                                                        src,
                                                        dofs_per_face);
      else // case 2: write to shared memory
        scatterv_face_internal<stride, do_add>(dsts, surface, src);
    }



  } // namespace SharedMPI
} // namespace LinearAlgebra


DEAL_II_NAMESPACE_CLOSE