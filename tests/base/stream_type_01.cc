// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
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


// test operator<< for different types and different streams

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include "../tests.h"



class FormattedOutput
{
public:
  template <typename T>
  FormattedOutput &
  operator<<(const T &output)
  {
    std::cout << std::fixed << std::right << std::setw(7)
              << std::setprecision(3) << output;

    return *this;
  }

  FormattedOutput &
  operator<<(std::ostream &(*func)(std::ostream &))
  {
    func(std::cout);
    return *this;
  }

private:
};



template <typename StreamType>
void
run_test(StreamType &stream)
{
  stream << 1.0 << std::endl;

  Point<2> point(1.0, 1.5);
  stream << point << std::endl;

  Tensor<1, 2, double> tensor = point;
  stream << tensor << std::endl;
}



int
main()
{
  run_test(std::cout);
  std::cout << std::endl;

  FormattedOutput mystream;
  run_test(mystream);
}
