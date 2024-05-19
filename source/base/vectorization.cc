// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2018 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#include <deal.II/base/vectorization.h>

DEAL_II_NAMESPACE_OPEN

// VectorizedArray must be a POD (plain old data) type to make sure it
// can use maximum level of compiler optimization.
// A type is POD if it has standard layout (similar to a C struct)
// and it is trivial (can be statically default initialized)
// Here, the trait std::is_pod cannot be used because it is deprecated
// in C++20.
static_assert(std::is_standard_layout_v<VectorizedArray<double>> &&
                std::is_trivial_v<VectorizedArray<double>>,
              "VectorizedArray<double> must be a POD type");
static_assert(std::is_standard_layout_v<VectorizedArray<float>> &&
                std::is_trivial_v<VectorizedArray<float>>,
              "VectorizedArray<float> must be a POD type");


static bool          precon_flag   = false;
static std::uint64_t counter_mul_0 = 0;
static std::uint64_t counter_mul_1 = 0;
static std::uint64_t counter_fma_0 = 0;
static std::uint64_t counter_fma_1 = 0;

bool
set_state(const bool flag)
{
  const bool old_flag = precon_flag;
  precon_flag         = flag;
  return old_flag;
}

std::uint64_t
get_n_fma(const bool flag)
{
  if (flag)
    return counter_fma_0;
  else
    return counter_fma_1;
}

void
inc_fma()
{
  if (precon_flag)
    counter_fma_0++;
  else
    counter_fma_1++;
}

std::uint64_t
get_n_mul(const bool flag)
{
  if (flag)
    return counter_mul_0;
  else
    return counter_mul_1;
}

void
inc_mul()
{
  if (precon_flag)
    counter_mul_0++;
  else
    counter_mul_1++;
}

DEAL_II_NAMESPACE_CLOSE
