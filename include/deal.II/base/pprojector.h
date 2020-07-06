// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
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

#ifndef dealii_base_pprojector_h
#define dealii_base_pprojector_h

#include <deal.II/base/config.h>

#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe_poly.h>

#include <deal.II/tet/polynomials.h>

DEAL_II_NAMESPACE_OPEN

template <int dim>
struct PProjector
{
  static Quadrature<3>
  project_to_all_faces(const Quadrature<2> quad)
  {
    const auto &sub_quadrature_points  = quad.get_points();
    const auto &sub_quadrature_weights = quad.get_weights();

    const std::array<std::pair<std::array<Point<3>, 3>, unsigned int>, 4>
      faces = {{{{Point<3>(0.0, 0.0, 0.0),
                  Point<3>(1.0, 0.0, 0.0),
                  Point<3>(0.0, 1.0, 0.0)},
                 1.0},
                {{Point<3>(1.0, 0.0, 0.0),
                  Point<3>(0.0, 0.0, 0.0),
                  Point<3>(0.0, 0.0, 1.0)},
                 1.0},
                {{Point<3>(0.0, 0.0, 0.0),
                  Point<3>(0.0, 1.0, 0.0),
                  Point<3>(0.0, 0.0, 1.0)},
                 1.0},
                {{Point<3>(0.0, 1.0, 0.0),
                  Point<3>(1.0, 0.0, 0.0),
                  Point<3>(0.0, 0.0, 1.0)},
                 2.0}}};

    Tet::ScalarPolynomial<2> poly(1);

    std::vector<Point<3>> points;
    std::vector<double>   weights;

    for (const auto &face : faces)
      {
        for (unsigned int o = 0; o < 6; ++o)
          {
            std::array<Point<3>, 3> support_points;

            switch (o)
              {
                case 1:
                  support_points = {face.first[0],
                                    face.first[1],
                                    face.first[2]};
                  break;
                case 3:
                  support_points = {face.first[1],
                                    face.first[0],
                                    face.first[2]};
                  break;
                case 5:
                  support_points = {face.first[2],
                                    face.first[0],
                                    face.first[1]};
                  break;
                case 0:
                  support_points = {face.first[0],
                                    face.first[2],
                                    face.first[1]};
                  break;
                case 2:
                  support_points = {face.first[1],
                                    face.first[2],
                                    face.first[0]};
                  break;
                case 4:
                  support_points = {face.first[2],
                                    face.first[1],
                                    face.first[0]};
                  break;
                default:
                  Assert(false, ExcNotImplemented());
              }

            for (unsigned int j = 0; j < sub_quadrature_points.size(); ++j)
              {
                Point<3> mapped_point;

                for (unsigned int i = 0; i < 3; ++i)
                  mapped_point +=
                    support_points[i] *
                    poly.compute_value(i, sub_quadrature_points[j]);

                points.push_back(mapped_point);
                weights.push_back(sub_quadrature_weights[j] * face.second);
              }
          }
      }

    return {points, weights};
  }

  static Quadrature<2>
  project_to_all_faces(const Quadrature<1> quad)
  {
    const auto &sub_quadrature_points  = quad.get_points();
    const auto &sub_quadrature_weights = quad.get_weights();

    const std::array<std::pair<std::array<Point<2>, 2>, unsigned int>, 3>
      faces = {{{{Point<2>(0.0, 0.0), Point<2>(1.0, 0.0)}, 1.0},
                {{Point<2>(1.0, 0.0), Point<2>(0.0, 1.0)}, 1.41421356237},
                {{Point<2>(0.0, 1.0), Point<2>(0.0, 0.0)}, 1.0}}};

    // const std::array<std::pair<std::array<Point<2>, 2>, unsigned int>, 3>
    //  faces = {{{{Point<2>(1.0, 0.0), Point<2>(0.0, 1.0)}, 1.41421356237},
    //            {{Point<2>(0.0, 1.0), Point<2>(0.0, 0.0)}, 1.0},
    //            {{Point<2>(0.0, 0.0), Point<2>(1.0, 0.0)}, 1.0}}};

    Tet::ScalarPolynomial<1> poly(1);

    std::vector<Point<2>> points;
    std::vector<double>   weights;

    for (const auto &face : faces)
      {
        for (unsigned int o = 0; o < 2; ++o)
          {
            std::array<Point<2>, 2> support_points;

            switch (o)
              {
                case 0:
                  support_points = {face.first[1], face.first[0]};
                  break;
                case 1:
                  support_points = {face.first[0], face.first[1]};
                  break;
                default:
                  Assert(false, ExcNotImplemented());
              }

            for (unsigned int j = 0; j < sub_quadrature_points.size(); ++j)
              {
                Point<2> mapped_point;

                for (unsigned int i = 0; i < 2; ++i)
                  mapped_point +=
                    support_points[i] *
                    poly.compute_value(i, sub_quadrature_points[j]);

                points.push_back(mapped_point);
                weights.push_back(sub_quadrature_weights[j] * face.second);
              }
          }
      }

    return {points, weights};
  }

  static Quadrature<1>
  project_to_all_faces(const Quadrature<0> quad)
  {
    Assert(false, ExcNotImplemented());

    (void)quad;

    return Quadrature<1>();
  }

  static Quadrature<3>
  project_to_all_subfaces(const Quadrature<2> quad)
  {
    Assert(false, ExcNotImplemented());

    (void)quad;

    return Quadrature<3>();
  }

  static Quadrature<2>
  project_to_all_subfaces(const Quadrature<1> quad)
  {
    Assert(false, ExcNotImplemented());

    (void)quad;

    return Quadrature<2>();
  }

  static Quadrature<1>
  project_to_all_subfaces(const Quadrature<0> quad)
  {
    Assert(false, ExcNotImplemented());

    (void)quad;

    return Quadrature<1>();
  }


  struct DataSetDescriptor
  {
    using SubQuadrature = Quadrature<dim - 1>;

    DataSetDescriptor()
      : dataset_offset(0)
    {}

    operator unsigned int() const
    {
      return dataset_offset;
    }

    static DataSetDescriptor
    cell()
    {
      return {0};
    }

    static DataSetDescriptor
    face(const unsigned int face_no,
         const bool         face_orientation,
         const bool         face_flip,
         const bool         face_rotation,
         const unsigned int n_quadrature_points)
    {
      if (dim == 2)
        return {(2 * face_no + face_orientation) * n_quadrature_points};
      else if (dim == 3)
        {
          const unsigned int orientation =
            (face_flip * 2 + face_rotation) * 2 + face_orientation;
          return {(6 * face_no + orientation) * n_quadrature_points};
        }

      Assert(false, ExcNotImplemented());
    }


    static DataSetDescriptor
    subface(const unsigned int               face_no,
            const unsigned int               subface_no,
            const bool                       face_orientation,
            const bool                       face_flip,
            const bool                       face_rotation,
            const unsigned int               n_quadrature_points,
            const internal::SubfaceCase<dim> ref_case =
              internal::SubfaceCase<dim>::case_isotropic)
    {
      Assert(false, ExcNotImplemented());

      (void)face_no;
      (void)subface_no;
      (void)face_orientation;
      (void)face_flip;
      (void)face_rotation;
      (void)n_quadrature_points;
      (void)ref_case;

      return {0};
    }

  private:
    const unsigned int dataset_offset;

    DataSetDescriptor(const unsigned int dataset_offset)
      : dataset_offset(dataset_offset)
    {}
  };
};

DEAL_II_NAMESPACE_CLOSE

#endif
