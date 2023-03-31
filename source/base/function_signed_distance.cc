// ---------------------------------------------------------------------
//
// Copyright (C) 2019 - 2022 by the deal.II authors
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
#include <deal.II/base/function_signed_distance.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/utilities.h>

#include <algorithm>
#include <numeric>

DEAL_II_NAMESPACE_OPEN

namespace Functions
{
  namespace SignedDistance
  {
    namespace internal
    {
      /**
       * Compute the minimal distance between a point @p p and an infinite line described by two support
       * points a (@p bottom_left) and b (@p top_right) according to
       *
       * @f[
       * d = \frac{|| (b - a) \times (f1 - p)||}{||b-a||}
       * @f]
       */
      template <int dim>
      double
      distance_to_line(const Point<dim> &p,
                       const Point<dim> &bottom_left,
                       const Point<dim> &top_right)
      {
        Assert(
          (top_right - bottom_left).norm() >=
            std::numeric_limits<double>::epsilon(),
          ExcMessage(
            "The support points must not lie on top of each other! Abort.."));
        if (dim == 3)
          return cross_product_3d(top_right - bottom_left, bottom_left - p)
                   .norm() /
                 (top_right - bottom_left).norm();
        else if (dim == 2)
          return std::abs((top_right - bottom_left) *
                          cross_product_2d(bottom_left - p)) /
                 (top_right - bottom_left).norm();
        else
          AssertThrow(false,
                      ExcMessage(
                        "distance to infinite line: dim must be 2 or 3."));
      }
    } // namespace internal
    template <int dim>
    Sphere<dim>::Sphere(const Point<dim> &center, const double radius)
      : center(center)
      , radius(radius)
    {
      Assert(radius > 0, ExcMessage("Radius must be positive."))
    }



    template <int dim>
    double
    Sphere<dim>::value(const Point<dim> & point,
                       const unsigned int component) const
    {
      AssertIndexRange(component, this->n_components);
      (void)component;

      return point.distance(center) - radius;
    }



    template <int dim>
    Tensor<1, dim>
    Sphere<dim>::gradient(const Point<dim> & point,
                          const unsigned int component) const
    {
      AssertIndexRange(component, this->n_components);
      (void)component;

      const Tensor<1, dim> center_to_point = point - center;
      const Tensor<1, dim> grad = center_to_point / center_to_point.norm();
      return grad;
    }



    template <int dim>
    SymmetricTensor<2, dim>
    Sphere<dim>::hessian(const Point<dim> & point,
                         const unsigned int component) const
    {
      AssertIndexRange(component, this->n_components);
      (void)component;

      const Tensor<1, dim> center_to_point = point - center;
      const double         distance        = center_to_point.norm();

      const SymmetricTensor<2, dim> hess =
        unit_symmetric_tensor<dim>() / distance -
        symmetrize(outer_product(center_to_point, center_to_point)) /
          Utilities::fixed_power<3>(distance);

      return hess;
    }



    template <int dim>
    Plane<dim>::Plane(const Point<dim> &point, const Tensor<1, dim> &normal)
      : point_in_plane(point)
      , normal(normal)
    {
      Assert(normal.norm() > 0, ExcMessage("Plane normal must not be 0."))
    }



    template <int dim>
    double
    Plane<dim>::value(const Point<dim> & point,
                      const unsigned int component) const
    {
      AssertIndexRange(component, this->n_components);
      (void)component;

      return normal * (point - point_in_plane);
    }



    template <int dim>
    Tensor<1, dim>
    Plane<dim>::gradient(const Point<dim> &, const unsigned int component) const
    {
      AssertIndexRange(component, this->n_components);
      (void)component;

      return normal;
    }



    template <int dim>
    SymmetricTensor<2, dim>
    Plane<dim>::hessian(const Point<dim> &, const unsigned int component) const
    {
      AssertIndexRange(component, this->n_components);
      (void)component;

      return SymmetricTensor<2, dim>();
    }



    template <int dim>
    Ellipsoid<dim>::Ellipsoid(const Point<dim> &             center,
                              const std::array<double, dim> &radii,
                              const double                   tolerance,
                              const unsigned int             max_iter)
      : center(center)
      , radii(radii)
      , tolerance(tolerance)
      , max_iter(max_iter)
    {
      for (unsigned int d = 0; d < dim; ++d)
        Assert(radii[d] > 0, ExcMessage("All radii must be positive."))
    }



    template <int dim>
    double
    Ellipsoid<dim>::value(const Point<dim> & point,
                          const unsigned int component) const
    {
      AssertIndexRange(component, this->n_components);
      (void)component;

      if (dim == 1)
        return point.distance(center) - radii[0];
      else if (dim == 2)
        return compute_signed_distance_ellipse(point);
      else
        Assert(false, ExcNotImplemented());

      return 0.0;
    }



    template <int dim>
    Tensor<1, dim>
    Ellipsoid<dim>::gradient(const Point<dim> & point,
                             const unsigned int component) const
    {
      AssertIndexRange(component, this->n_components);
      (void)component;

      Tensor<1, dim> grad;
      if (dim == 1)
        grad = point - center;
      else if (dim == 2)
        {
          const Point<dim> point_in_centered_coordinate_system =
            Point<dim>(compute_closest_point_ellipse(point) - center);
          grad = compute_analyical_normal_vector_on_ellipse(
            point_in_centered_coordinate_system);
        }
      else
        AssertThrow(false, ExcNotImplemented());

      if (grad.norm() > 1e-12)
        return grad / grad.norm();
      else
        return grad;
    }



    template <int dim>
    double
    Ellipsoid<dim>::evaluate_ellipsoid(const Point<dim> &point) const
    {
      double val = 0.0;
      for (unsigned int d = 0; d < dim; ++d)
        val += Utilities::fixed_power<2>((point[d] - center[d]) / radii[d]);
      return val - 1.0;
    }



    template <int dim>
    Point<dim>
    Ellipsoid<dim>::compute_closest_point_ellipse(const Point<dim> &point) const
    {
      AssertDimension(dim, 2);

      /*
       * Function to compute the closest point on an ellipse (adopted from
       * https://wet-robots.ghost.io/simple-method-for-distance-to-ellipse/ and
       * https://github.com/0xfaded/ellipse_demo):
       *
       * Since the ellipse is symmetric to the two major axes through its
       * center, the point is moved so the center coincides with the origin and
       * into the first quadrant.
       * 1. Choose a point on the ellipse (x), here x = a*cos(pi/4) and y =
       * b*sin(pi/4).
       * 2. Find second point on the ellipse, that has the same distance.
       * 3. Find midpoint on the ellipse (must be closer).
       * 4. Repeat 2.-4. until convergence.
       */
      // get equivalent point in first quadrant of centered ellipse
      const double px      = std::abs(point[0] - center[0]);
      const double py      = std::abs(point[1] - center[1]);
      const double sign_px = std::copysign(1.0, point[0] - center[0]);
      const double sign_py = std::copysign(1.0, point[1] - center[1]);
      // get semi axes radii
      const double &a = radii[0];
      const double &b = radii[1];
      // initial guess (t = angle from x-axis)
      double t = numbers::PI_4;
      double x = a * std::cos(t);
      double y = b * std::sin(t);

      unsigned int iter = 0;
      double       delta_t;
      do
        {
          // compute the ellipse evolute (center of curvature) for the current t
          const double ex =
            (a * a - b * b) * Utilities::fixed_power<3>(std::cos(t)) / a;
          const double ey =
            (b * b - a * a) * Utilities::fixed_power<3>(std::sin(t)) / b;
          // compute distances from current point on ellipse to its evolute
          const double rx = x - ex;
          const double ry = y - ey;
          // compute distances from point to the current evolute
          const double qx = px - ex;
          const double qy = py - ey;
          // compute the curvature radius at the current point on the ellipse
          const double r = std::hypot(rx, ry);
          // compute the distance from evolute to the point
          const double q = std::hypot(qx, qy);
          // compute step size on ellipse
          const double delta_c = r * std::asin((rx * qy - ry * qx) / (r * q));
          // compute approximate angle step
          delta_t = delta_c / std::sqrt(a * a + b * b - x * x - y * y);
          t += delta_t;
          // make sure the angle stays in first quadrant
          t = std::min(numbers::PI_2, std::max(0.0, t));
          x = a * std::cos(t);
          y = b * std::sin(t);
          iter++;
        }
      while (std::abs(delta_t) > tolerance && iter < max_iter);
      AssertIndexRange(iter, max_iter);

      AssertIsFinite(x);
      AssertIsFinite(y);

      return center + Point<dim>(sign_px * x, sign_py * y);
    }



    template <int dim>
    Tensor<1, dim, double>
    Ellipsoid<dim>::compute_analyical_normal_vector_on_ellipse(
      const Point<dim> &) const
    {
      AssertThrow(false, ExcNotImplemented());
      return Tensor<1, dim, double>();
    }



    template <>
    Tensor<1, 2, double>
    Ellipsoid<2>::compute_analyical_normal_vector_on_ellipse(
      const Point<2> &point) const
    {
      const auto &a = radii[0];
      const auto &b = radii[1];
      const auto &x = point[0];
      const auto &y = point[1];
      return Tensor<1, 2, double>({b * x / a, a * y / b});
    }



    template <int dim>
    double
    Ellipsoid<dim>::compute_signed_distance_ellipse(const Point<dim> &) const
    {
      AssertThrow(false, ExcNotImplemented());
      return 0;
    }



    template <>
    double
    Ellipsoid<2>::compute_signed_distance_ellipse(const Point<2> &point) const
    {
      // point corresponds to center
      if (point.distance(center) < tolerance)
        return *std::min_element(radii.begin(), radii.end()) * -1.;

      const Point<2> &closest_point = compute_closest_point_ellipse(point);

      const double distance =
        std::hypot(closest_point[0] - point[0], closest_point[1] - point[1]);

      return evaluate_ellipsoid(point) < 0.0 ? -distance : distance;
    }



    template <int dim>
    Rectangle<dim>::Rectangle(const Point<dim> &bottom_left,
                              const Point<dim> &top_right)
      : bounding_box({bottom_left, top_right})
    {
      if constexpr (dim == 3)
        {
          boundary_faces.emplace_back(Functions::SignedDistance::Plane(
            bottom_left, -Point<dim>::unit_vector(0))); // left
          boundary_faces.emplace_back(Functions::SignedDistance::Plane(
            top_right, Point<dim>::unit_vector(0))); // right
          boundary_faces.emplace_back(Functions::SignedDistance::Plane(
            bottom_left,
            -Point<dim>::unit_vector(1))); // front
          boundary_faces.emplace_back(Functions::SignedDistance::Plane(
            top_right, Point<dim>::unit_vector(1))); // back
          boundary_faces.emplace_back(Functions::SignedDistance::Plane(
            bottom_left,
            -Point<dim>::unit_vector(2))); // bottom
          boundary_faces.emplace_back(Functions::SignedDistance::Plane(
            top_right, Point<dim>::unit_vector(2))); // top
        }
    }



    template <int dim>
    double
    Rectangle<dim>::value(const Point<dim> & p,
                          const unsigned int component) const
    {
      (void)component;
      const Point<dim> &bottom_left = bounding_box.get_boundary_points().first;
      const Point<dim> &top_right   = bounding_box.get_boundary_points().second;

      if constexpr (dim == 3)
        {
          // inside (1 case)
          if (bounding_box.point_inside(p))
            {
              double signed_distance = std::numeric_limits<double>::lowest();

              for (const auto &plane : boundary_faces)
                signed_distance = std::max(plane.value(p), signed_distance);

              return signed_distance;
            }
          // boundary faces (6 cases)
          for (unsigned int i = 0; i < boundary_faces.size(); ++i)
            {
              // check if the point has a positive value for the signed distance
              // to the current face
              if (boundary_faces[i].value(p) >= 1e-16)
                {
                  // check if all other faces have negative distance values
                  bool all_other_faces_negative = true;
                  for (unsigned int j = 0; j < boundary_faces.size(); ++j)
                    {
                      if ((i != j) && boundary_faces[j].value(p) > 0)
                        {
                          all_other_faces_negative = false;
                          break;
                        }
                    }

                  if (all_other_faces_negative)
                    return boundary_faces[i].value(p);
                }
            }
          // corners (8 cases)
          for (unsigned int i = 0; i < 8; i++)
            {
              const auto face_indices = GeometryInfo<dim>::vertex_to_face[i];

              bool found = true;

              std::vector<unsigned int> processed_faces(6);
              std::iota(processed_faces.begin(), processed_faces.end(), 0);

              // check if the point has a positive value for the signed distance
              // to adjoining faces of the current corner
              for (const auto &f : face_indices)
                {
                  if (boundary_faces[f].value(p) < 0)
                    {
                      found = false;
                      break;
                    }

                  processed_faces.erase(std::find(processed_faces.begin(),
                                                  processed_faces.end(),
                                                  f));
                }

              // check if the point has a negative value for the signed distance
              // to all remaining faces
              if (found)
                {
                  for (const auto &f : processed_faces)
                    {
                      if (boundary_faces[f].value(p) > 0)
                        {
                          found = false;
                          break;
                        }
                    }
                }

              if (found)
                return bounding_box.vertex(i).distance(p);
            }
          // boundary lines (12 cases)
          for (unsigned int i = 0; i < 12; i++)
            {
              bool found = true;

              std::vector<unsigned int> processed_faces(6);
              std::iota(processed_faces.begin(), processed_faces.end(), 0);

              // check if the point has a positive value for the signed distance
              // to adjoining faces of the current edge
              for (const auto &f : GeometryInfo<dim>::line_to_face[i])
                {
                  if (boundary_faces[f].value(p) < 0)
                    {
                      found = false;
                      break;
                    }

                  processed_faces.erase(std::find(processed_faces.begin(),
                                                  processed_faces.end(),
                                                  f));
                }

              // check if the point has a negative value for the signed distance
              // to all remaining faces
              if (found)
                {
                  for (const auto &f : processed_faces)
                    {
                      if (boundary_faces[f].value(p) > 0)
                        {
                          found = false;
                          break;
                        }
                    }
                }

              if (found)
                return internal::distance_to_line(
                  p,
                  bounding_box.vertex(
                    GeometryInfo<dim>::line_to_cell_vertices(i, 0)),
                  bounding_box.vertex(
                    GeometryInfo<dim>::line_to_cell_vertices(i, 1)));
            }
          AssertThrow(
            false,
            ExcMessage(
              "The distance of your requested point could not be calculated."));
          return 0;
        }
      else if constexpr (dim == 2)
        {
          // inside
          if (bounding_box.point_inside(p))
            return -std::min({p[0] - bottom_left[0],
                              top_right[0] - p[0],
                              p[1] - bottom_left[1],
                              top_right[1] - p[1]});
          // top or bottom
          else if (bottom_left[0] <= p[0] && p[0] <= top_right[0])
            return std::min(std::abs(bottom_left[1] - p[1]),
                            std::abs(p[1] - top_right[1]));
          // left or right
          else if (bottom_left[1] <= p[1] && p[1] <= top_right[1])
            return std::min(std::abs(bottom_left[0] - p[0]),
                            std::abs(p[0] - top_right[0]));
          else
            // corner
            return std::min(
              {p.distance(bottom_left),
               p.distance(top_right),
               p.distance(Point<2>(bottom_left[0], top_right[1])),
               p.distance(Point<2>(top_right[0], bottom_left[1]))});
        }
      else if constexpr (dim == 1)
        {
          // left
          if (p[0] <= bottom_left[0])
            return p.distance(bottom_left);
          // right
          else if (p[0] >= top_right[0])
            return p.distance(top_right);
          // inside left
          else if (p[0] <= bounding_box.center()[0])
            return -p.distance(bottom_left);
          // inside right
          else
            return -p.distance(top_right);
        }
      else
        return 0;
    }

  } // namespace SignedDistance
} // namespace Functions

#include "function_signed_distance.inst"

DEAL_II_NAMESPACE_CLOSE
