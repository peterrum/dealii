// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2022 by the deal.II authors
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

#ifndef dealii_base_floating_point_copmerator_h
#define dealii_base_floating_point_copmerator_h

#include <deal.II/base/config.h>

#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/vectorization.h>

#include <bitset>
#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * A class that is used to compare floating point arrays (e.g. std::vector,
 * Tensor<1,dim>, etc.). The most common use case of this comparator
 * is for detecting arrays with the same content for the sake of
 * compression. The idea of this class is to consider two arrays as
 * equal if they are the same within a given tolerance. We use this
 * comparator class within a std::map<> of the given arrays. Note that this
 * comparison operator does not satisfy all the mathematical properties one
 * usually wants to have (consider e.g. the numbers a=0, b=0.1, c=0.2 with
 * tolerance 0.15; the operator gives a<c, but neither a<b? nor b<c? is
 * satisfied). This is not a problem in the use cases for this class, but be
 * careful when using it in other contexts.
 */
template <typename Number>
struct FloatingPointComparator
{
  using ScalarNumber =
    typename dealii::internal::VectorizedArrayTrait<Number>::value_type;
  static constexpr std::size_t width =
    dealii::internal::VectorizedArrayTrait<Number>::width;

  /**
   * Constructor.
   *
   * TODO: What is scaling?
   */
  FloatingPointComparator(
    const ScalarNumber       scaling,
    const std::bitset<width> mask = std::bitset<width>().flip());

  /**
   * Compare two vectors of numbers (not necessarily of the same length),
   * where vectors of different lengths are first sorted by their length and
   * then by the entries.
   */
  bool
  operator()(const std::vector<ScalarNumber> &v1,
             const std::vector<ScalarNumber> &v2) const;

  /**
   * Compare two vectorized arrays (stored as tensors to avoid alignment
   * issues).
   */
  bool
  operator()(const Tensor<1, width, ScalarNumber> &t1,
             const Tensor<1, width, ScalarNumber> &t2) const;

  /**
   * Compare two rank-1 tensors of vectorized arrays (stored as tensors to
   * avoid alignment issues).
   */
  template <int dim>
  bool
  operator()(const Tensor<1, dim, Tensor<1, width, ScalarNumber>> &t1,
             const Tensor<1, dim, Tensor<1, width, ScalarNumber>> &t2) const;

  /**
   * Compare two rank-2 tensors of vectorized arrays (stored as tensors to
   * avoid alignment issues).
   */
  template <int dim>
  bool
  operator()(const Tensor<2, dim, Tensor<1, width, ScalarNumber>> &t1,
             const Tensor<2, dim, Tensor<1, width, ScalarNumber>> &t2) const;

  /**
   * Compare two arrays of tensors.
   */
  template <int dim>
  bool
  operator()(const std::array<Tensor<2, dim, ScalarNumber>, dim + 1> &t1,
             const std::array<Tensor<2, dim, ScalarNumber>, dim + 1> &t2) const;

  /**
   * Compare two tables.
   */
  template <typename T>
  bool
  operator()(const Table<2, T> &t1, const Table<2, T> &t2) const;

  bool
  operator()(const ScalarNumber &s1, const ScalarNumber &s2) const;

  bool
  operator()(const VectorizedArray<ScalarNumber, width> &v1,
             const VectorizedArray<ScalarNumber, width> &v2) const;

private:
  const ScalarNumber       tolerance;
  const std::bitset<width> mask;
};


/* ------------------------------------------------------------------ */


template <typename Number>
FloatingPointComparator<Number>::FloatingPointComparator(
  const ScalarNumber       scaling,
  const std::bitset<width> mask)
  : tolerance(scaling * std::numeric_limits<double>::epsilon() * 1024.)
  , mask(mask)
{}



template <typename Number>
bool
FloatingPointComparator<Number>::operator()(
  const std::vector<ScalarNumber> &v1,
  const std::vector<ScalarNumber> &v2) const
{
  const unsigned int s1 = v1.size(), s2 = v2.size();
  if (s1 < s2)
    return true;
  else if (s1 > s2)
    return false;
  else
    for (unsigned int i = 0; i < s1; ++i)
      if (this->operator()(v1[i], v2[i]))
        return true;
      else if (this->operator()(v2[i], v1[i]))
        return false;
  return false;
}



template <typename Number>
bool
FloatingPointComparator<Number>::operator()(
  const Tensor<1, width, ScalarNumber> &t1,
  const Tensor<1, width, ScalarNumber> &t2) const
{
  for (unsigned int k = 0; k < width; ++k)
    if (this->operator()(t1[k], t2[k]))
      return true;
    else if (this->operator()(t2[k], t1[k]))
      return false;
  return false;
}



template <typename Number>
template <int dim>
bool
FloatingPointComparator<Number>::operator()(
  const Tensor<1, dim, Tensor<1, width, ScalarNumber>> &t1,
  const Tensor<1, dim, Tensor<1, width, ScalarNumber>> &t2) const
{
  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int k = 0; k < width; ++k)
      if (this->operator()(t1[d][k], t2[d][k]))
        return true;
      else if (this->operator()(t2[d][k], t1[d][k]))
        return false;
  return false;
}



template <typename Number>
template <int dim>
bool
FloatingPointComparator<Number>::operator()(
  const Tensor<2, dim, Tensor<1, width, ScalarNumber>> &t1,
  const Tensor<2, dim, Tensor<1, width, ScalarNumber>> &t2) const
{
  for (unsigned int d = 0; d < dim; ++d)
    for (unsigned int e = 0; e < dim; ++e)
      for (unsigned int k = 0; k < width; ++k)
        if (this->operator()(t1[d][e][k], t2[d][e][k]))
          return true;
        else if (this->operator()(t2[d][e][k], t1[d][e][k]))
          return false;
  return false;
}



template <typename Number>
template <int dim>
bool
FloatingPointComparator<Number>::operator()(
  const std::array<Tensor<2, dim, ScalarNumber>, dim + 1> &t1,
  const std::array<Tensor<2, dim, ScalarNumber>, dim + 1> &t2) const
{
  for (unsigned int i = 0; i < t1.size(); ++i)
    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int e = 0; e < dim; ++e)
        if (this->operator()(t1[i][d][e], t2[i][d][e]))
          return true;
        else if (this->operator()(t2[i][d][e], t1[i][d][e]))
          return false;
  return false;
}


template <typename Number>
template <typename T>
bool
FloatingPointComparator<Number>::operator()(const Table<2, T> &t1,
                                            const Table<2, T> &t2) const
{
  AssertDimension(t1.size(0), t2.size(0));
  AssertDimension(t1.size(1), t2.size(1));

  for (unsigned int i = 0; i < t1.size(0); ++i)
    for (unsigned int j = 0; j < t1.size(1); ++j)
      if (this->operator()(t1[i][j], t2[i][j]))
        return true;
      else if (this->operator()(t2[i][j], t1[i][j]))
        return false;
  return false;
}

template <typename Number>
bool
FloatingPointComparator<Number>::operator()(const ScalarNumber &s1,
                                            const ScalarNumber &s2) const
{
  if (mask[0] && (s1 < s2 - tolerance))
    return true;
  else
    return false;
}

template <typename Number>
bool
FloatingPointComparator<Number>::operator()(
  const VectorizedArray<ScalarNumber, width> &v1,
  const VectorizedArray<ScalarNumber, width> &v2) const
{
  for (unsigned int v = 0; v < width; ++v)
    if (mask[v])
      {
        if (v1[v] < v2[v] - tolerance)
          return true;
        if (v1[v] > v2[v] + tolerance)
          return false;
      }

  return false;
}


DEAL_II_NAMESPACE_CLOSE

#endif
