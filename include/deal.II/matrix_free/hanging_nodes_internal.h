// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#ifndef dealii_matrix_free_hanging_nodes_internal_h
#define dealii_matrix_free_hanging_nodes_internal_h

#include <deal.II/base/config.h>

DEAL_II_NAMESPACE_OPEN

namespace internal
{
  // Here is the system for how we store constraint types in a binary mask.
  // This is not a complete contradiction-free system, i.e., there are
  // invalid states that we just assume that we never get.

  // If the mask is zero, there are no constraints. Then, there are three
  // different fields with one bit per dimension. The first field determines
  // the type, or the position of an element along each direction. The
  // second field determines if there is a constrained face with that
  // direction as normal. The last field determines if there is a
  // constrained edge of a given pair of coordinate planes, but where
  // neither of the corresponding faces are constrained (only valid in 3D).

  // The element is placed in the 'first position' along *-axis. These also
  // determine which face is constrained. For example, in 2D, if
  // constr_face_x and constr_type_x are set, then x = 0 is constrained.
  constexpr unsigned int constr_type_x = 1 << 0;
  constexpr unsigned int constr_type_y = 1 << 1;
  constexpr unsigned int constr_type_z = 1 << 2;

  // Element has as a constraint at * = 0 or * = fe_degree face
  constexpr unsigned int constr_face_x = 1 << 3;
  constexpr unsigned int constr_face_y = 1 << 4;
  constexpr unsigned int constr_face_z = 1 << 5;

  // Element has as a constraint at * = 0 or * = fe_degree edge
  constexpr unsigned int constr_edge_xy = 1 << 6;
  constexpr unsigned int constr_edge_yz = 1 << 7;
  constexpr unsigned int constr_edge_zx = 1 << 8;


  /**
   * This class creates the mask used in the treatment of hanging nodes in
   * CUDAWrappers::MatrixFree.
   * The implementation of this class is explained in <em>Section 3 of
   * Matrix-Free Finite-Element Computations On Graphics Processors With
   * Adaptively Refined Unstructured Meshes</em> by Karl Ljungkvist,
   * SpringSim-HPC, 2017 April 23-26.
   */
  template <int dim>
  class HangingNodes
  {
  public:
    /**
     * Constructor.
     */
    HangingNodes(const Triangulation<dim> &dof_handler);

    /**
     * Compute the value of the constraint mask for a given cell.
     */
    template <typename CellIterator>
    void
    setup_constraints(
      const CellIterator &                                      cell,
      const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
      const std::vector<unsigned int> &lexicographic_mapping,
      const ArrayView<unsigned int> &  dof_indices,
      unsigned int &                   mask) const;

  private:
    /**
     * Set up line-to-cell mapping for edge constraints in 3D.
     */
    void
    setup_line_to_cell(const Triangulation<dim> &dof_handler);

    void
    rotate_subface_index(int times, unsigned int &subface_index) const;

    void
    rotate_face(int                                   times,
                unsigned int                          n_dofs_1d,
                std::vector<types::global_dof_index> &dofs) const;

    unsigned int
    line_dof_idx(int          local_line,
                 unsigned int dof,
                 unsigned int n_dofs_1d) const;

    void
    transpose_face(const unsigned int                    fe_degree,
                   std::vector<types::global_dof_index> &dofs) const;

    void
    transpose_subface_index(unsigned int &subface) const;

    std::vector<std::vector<
      std::pair<typename Triangulation<dim>::cell_iterator, unsigned int>>>
      line_to_cells;
  };



  template <int dim>
  inline HangingNodes<dim>::HangingNodes(
    const Triangulation<dim> &triangulation)
  {
    // Set up line-to-cell mapping for edge constraints (only if dim = 3)
    setup_line_to_cell(triangulation);
  }



  template <int dim>
  inline void
  HangingNodes<dim>::setup_line_to_cell(const Triangulation<dim> &triangulation)
  {
    (void)triangulation;
  }



  template <>
  inline void
  HangingNodes<3>::setup_line_to_cell(const Triangulation<3> &triangulation)
  {
    const unsigned int n_raw_lines = triangulation.n_raw_lines();
    this->line_to_cells.resize(n_raw_lines);

    // In 3D, we can have DoFs on only an edge being constrained (e.g. in a
    // cartesian 2x2x2 grid, where only the upper left 2 cells are refined).
    // This sets up a helper data structure in the form of a mapping from
    // edges (i.e. lines) to neighboring cells.

    // Mapping from an edge to which children that share that edge.
    const unsigned int line_to_children[12][2] = {{0, 2},
                                                  {1, 3},
                                                  {0, 1},
                                                  {2, 3},
                                                  {4, 6},
                                                  {5, 7},
                                                  {4, 5},
                                                  {6, 7},
                                                  {0, 4},
                                                  {1, 5},
                                                  {2, 6},
                                                  {3, 7}};

    std::vector<std::vector<
      std::pair<typename Triangulation<3>::cell_iterator, unsigned int>>>
      line_to_inactive_cells(n_raw_lines);

    // First add active and inactive cells to their lines:
    for (const auto &cell : triangulation.cell_iterators())
      {
        for (unsigned int line = 0; line < GeometryInfo<3>::lines_per_cell;
             ++line)
          {
            const unsigned int line_idx = cell->line(line)->index();
            if (cell->is_active())
              line_to_cells[line_idx].push_back(std::make_pair(cell, line));
            else
              line_to_inactive_cells[line_idx].push_back(
                std::make_pair(cell, line));
          }
      }

    // Now, we can access edge-neighboring active cells on same level to also
    // access of an edge to the edges "children". These are found from looking
    // at the corresponding edge of children of inactive edge neighbors.
    for (unsigned int line_idx = 0; line_idx < n_raw_lines; ++line_idx)
      {
        if ((line_to_cells[line_idx].size() > 0) &&
            line_to_inactive_cells[line_idx].size() > 0)
          {
            // We now have cells to add (active ones) and edges to which they
            // should be added (inactive cells).
            const auto &inactive_cell =
              line_to_inactive_cells[line_idx][0].first;
            const unsigned int neighbor_line =
              line_to_inactive_cells[line_idx][0].second;

            for (unsigned int c = 0; c < 2; ++c)
              {
                const auto &child =
                  inactive_cell->child(line_to_children[neighbor_line][c]);
                const unsigned int child_line_idx =
                  child->line(neighbor_line)->index();

                // Now add all active cells
                for (const auto cl : line_to_cells[line_idx])
                  line_to_cells[child_line_idx].push_back(cl);
              }
          }
      }
  }



  template <int dim>
  template <typename CellIterator>
  inline void
  HangingNodes<dim>::setup_constraints(
    const CellIterator &                                      cell,
    const std::shared_ptr<const Utilities::MPI::Partitioner> &partitioner,
    const std::vector<unsigned int> &lexicographic_mapping,
    const ArrayView<unsigned int> &  dof_indices,
    unsigned int &                   mask) const
  {
    mask                         = 0;
    const unsigned int fe_degree = cell->get_fe().tensor_degree();
    const unsigned int n_dofs_1d = fe_degree + 1;
    const unsigned int dofs_per_face =
      Utilities::fixed_power<dim - 1>(n_dofs_1d);

    std::vector<types::global_dof_index> neighbor_dofs(dofs_per_face);

    const auto lex_face_mapping =
      FETools::lexicographic_to_hierarchic_numbering<dim - 1>(fe_degree);

    for (const unsigned int face : GeometryInfo<dim>::face_indices())
      {
        if ((!cell->at_boundary(face)) &&
            (cell->neighbor(face)->has_children() == false))
          {
            const auto &neighbor = cell->neighbor(face);

            // Neighbor is coarser than us, i.e., face is constrained
            if (neighbor->level() < cell->level())
              {
                const unsigned int neighbor_face = cell->neighbor_face_no(face);

                // Find position of face on neighbor
                unsigned int subface = 0;
                for (; subface < GeometryInfo<dim>::max_children_per_face;
                     ++subface)
                  if (neighbor->neighbor_child_on_subface(neighbor_face,
                                                          subface) == cell)
                    break;

                // Get indices to read
                DoFAccessor<dim - 1, dim, dim, false>(
                  &neighbor->face(neighbor_face)->get_triangulation(),
                  neighbor->face(neighbor_face)->level(),
                  neighbor->face(neighbor_face)->index(),
                  &cell->get_dof_handler())
                  .get_dof_indices(neighbor_dofs);

                // If the vector is distributed, we need to transform the
                // global indices to local ones.
                if (partitioner)
                  for (auto &index : neighbor_dofs)
                    index = partitioner->global_to_local(index);

                if (dim == 2)
                  {
                    if (face < 2)
                      {
                        mask |= internal::constr_face_x;
                        if (face == 0)
                          mask |= internal::constr_type_x;
                        if (subface == 0)
                          mask |= internal::constr_type_y;
                      }
                    else
                      {
                        mask |= internal::constr_face_y;
                        if (face == 2)
                          mask |= internal::constr_type_y;
                        if (subface == 0)
                          mask |= internal::constr_type_x;
                      }

                    // Reorder neighbor_dofs and copy into faceth face of
                    // dof_indices

                    // Offset if upper/right face
                    unsigned int offset = (face % 2 == 1) ? fe_degree : 0;

                    for (unsigned int i = 0; i < n_dofs_1d; ++i)
                      {
                        unsigned int idx = 0;
                        // If X-line, i.e., if y = 0 or y = fe_degree
                        if (face > 1)
                          idx = n_dofs_1d * offset + i;
                        // If Y-line, i.e., if x = 0 or x = fe_degree
                        else
                          idx = n_dofs_1d * i + offset;

                        dof_indices[idx] = neighbor_dofs[lex_face_mapping[i]];
                      }
                  }
                else if (dim == 3)
                  {
                    const bool transpose = !(cell->face_orientation(face));

                    int rotate = 0;

                    if (cell->face_rotation(face))
                      rotate -= 1;
                    if (cell->face_flip(face))
                      rotate -= 2;

                    rotate_face(rotate, n_dofs_1d, neighbor_dofs);
                    rotate_subface_index(rotate, subface);

                    if (transpose)
                      {
                        transpose_face(fe_degree, neighbor_dofs);
                        transpose_subface_index(subface);
                      }

                    // YZ-plane
                    if (face < 2)
                      {
                        mask |= internal::constr_face_x;
                        if (face == 0)
                          mask |= internal::constr_type_x;
                        if (subface % 2 == 0)
                          mask |= internal::constr_type_y;
                        if (subface / 2 == 0)
                          mask |= internal::constr_type_z;
                      }
                    // XZ-plane
                    else if (face < 4)
                      {
                        mask |= internal::constr_face_y;
                        if (face == 2)
                          mask |= internal::constr_type_y;
                        if (subface % 2 == 0)
                          mask |= internal::constr_type_z;
                        if (subface / 2 == 0)
                          mask |= internal::constr_type_x;
                      }
                    // XY-plane
                    else
                      {
                        mask |= internal::constr_face_z;
                        if (face == 4)
                          mask |= internal::constr_type_z;
                        if (subface % 2 == 0)
                          mask |= internal::constr_type_x;
                        if (subface / 2 == 0)
                          mask |= internal::constr_type_y;
                      }

                    // Offset if upper/right/back face
                    unsigned int offset = (face % 2 == 1) ? fe_degree : 0;

                    for (unsigned int i = 0; i < n_dofs_1d; ++i)
                      {
                        for (unsigned int j = 0; j < n_dofs_1d; ++j)
                          {
                            unsigned int idx = 0;
                            // If YZ-plane, i.e., if x = 0 or x = fe_degree,
                            // and orientation standard
                            if (face < 2)
                              idx = n_dofs_1d * n_dofs_1d * i + n_dofs_1d * j +
                                    offset;
                            // If XZ-plane, i.e., if y = 0 or y = fe_degree,
                            // and orientation standard
                            else if (face < 4)
                              idx = n_dofs_1d * n_dofs_1d * j +
                                    n_dofs_1d * offset + i;
                            // If XY-plane, i.e., if z = 0 or z = fe_degree,
                            // and orientation standard
                            else
                              idx = n_dofs_1d * n_dofs_1d * offset +
                                    n_dofs_1d * i + j;

                            dof_indices[idx] =
                              neighbor_dofs[lex_face_mapping[n_dofs_1d * i +
                                                             j]];
                          }
                      }
                  }
                else
                  ExcNotImplemented();
              }
          }
      }

    // In 3D we can have a situation where only DoFs on an edge are
    // constrained. Append these here.
    if (dim == 3)
      {
        // For each line on cell, which faces does it belong to, what is the
        // edge mask, what is the types of the faces it belong to, and what is
        // the type along the edge.
        const unsigned int line_to_edge[12][4] = {
          {internal::constr_face_x | internal::constr_face_z,
           internal::constr_edge_zx,
           internal::constr_type_x | internal::constr_type_z,
           internal::constr_type_y},
          {internal::constr_face_x | internal::constr_face_z,
           internal::constr_edge_zx,
           internal::constr_type_z,
           internal::constr_type_y},
          {internal::constr_face_y | internal::constr_face_z,
           internal::constr_edge_yz,
           internal::constr_type_y | internal::constr_type_z,
           internal::constr_type_x},
          {internal::constr_face_y | internal::constr_face_z,
           internal::constr_edge_yz,
           internal::constr_type_z,
           internal::constr_type_x},
          {internal::constr_face_x | internal::constr_face_z,
           internal::constr_edge_zx,
           internal::constr_type_x,
           internal::constr_type_y},
          {internal::constr_face_x | internal::constr_face_z,
           internal::constr_edge_zx,
           0,
           internal::constr_type_y},
          {internal::constr_face_y | internal::constr_face_z,
           internal::constr_edge_yz,
           internal::constr_type_y,
           internal::constr_type_x},
          {internal::constr_face_y | internal::constr_face_z,
           internal::constr_edge_yz,
           0,
           internal::constr_type_x},
          {internal::constr_face_x | internal::constr_face_y,
           internal::constr_edge_xy,
           internal::constr_type_x | internal::constr_type_y,
           internal::constr_type_z},
          {internal::constr_face_x | internal::constr_face_y,
           internal::constr_edge_xy,
           internal::constr_type_y,
           internal::constr_type_z},
          {internal::constr_face_x | internal::constr_face_y,
           internal::constr_edge_xy,
           internal::constr_type_x,
           internal::constr_type_z},
          {internal::constr_face_x | internal::constr_face_y,
           internal::constr_edge_xy,
           0,
           internal::constr_type_z}};

        for (unsigned int local_line = 0;
             local_line < GeometryInfo<dim>::lines_per_cell;
             ++local_line)
          {
            // If we don't already have a constraint for as part of a face
            if (!(mask & line_to_edge[local_line][0]))
              {
                // For each cell which share that edge
                const unsigned int line = cell->line(local_line)->index();
                for (const auto edge_neighbor : line_to_cells[line])
                  {
                    // If one of them is coarser than us
                    const auto neighbor_cell = edge_neighbor.first;
                    if (neighbor_cell->level() < cell->level())
                      {
                        const unsigned int local_line_neighbor =
                          edge_neighbor.second;
                        mask |= line_to_edge[local_line][1] |
                                line_to_edge[local_line][2];

                        bool flipped = false;
                        if (cell->line(local_line)->vertex_index(0) ==
                            neighbor_cell->line(local_line_neighbor)
                              ->vertex_index(0))
                          {
                            // Assuming line directions match axes directions,
                            // we have an unflipped edge of first type
                            mask |= line_to_edge[local_line][3];
                          }
                        else if (cell->line(local_line)->vertex_index(1) ==
                                 neighbor_cell->line(local_line_neighbor)
                                   ->vertex_index(1))
                          {
                            // We have an unflipped edge of second type
                          }
                        else if (cell->line(local_line)->vertex_index(1) ==
                                 neighbor_cell->line(local_line_neighbor)
                                   ->vertex_index(0))
                          {
                            // We have a flipped edge of second type
                            flipped = true;
                          }
                        else if (cell->line(local_line)->vertex_index(0) ==
                                 neighbor_cell->line(local_line_neighbor)
                                   ->vertex_index(1))
                          {
                            // We have a flipped edge of first type
                            mask |= line_to_edge[local_line][3];
                            flipped = true;
                          }
                        else
                          ExcInternalError();

                        // Copy the unconstrained values
                        neighbor_dofs.resize(n_dofs_1d * n_dofs_1d * n_dofs_1d);
                        DoFCellAccessor<dim, dim, false>(
                          &neighbor_cell->get_triangulation(),
                          neighbor_cell->level(),
                          neighbor_cell->index(),
                          &cell->get_dof_handler())
                          .get_dof_indices(neighbor_dofs);
                        // If the vector is distributed, we need to transform
                        // the global indices to local ones.
                        if (partitioner)
                          for (auto &index : neighbor_dofs)
                            index = partitioner->global_to_local(index);

                        for (unsigned int i = 0; i < n_dofs_1d; ++i)
                          {
                            // Get local dof index along line
                            const unsigned int idx =
                              line_dof_idx(local_line, i, n_dofs_1d);
                            dof_indices[idx] =
                              neighbor_dofs[lexicographic_mapping[line_dof_idx(
                                local_line_neighbor,
                                flipped ? fe_degree - i : i,
                                n_dofs_1d)]];
                          }

                        // Stop looping over edge neighbors
                        break;
                      }
                  }
              }
          }
      }
  }



  template <int dim>
  inline void
  HangingNodes<dim>::rotate_subface_index(int           times,
                                          unsigned int &subface_index) const
  {
    const unsigned int rot_mapping[4] = {2, 0, 3, 1};

    times = times % 4;
    times = times < 0 ? times + 4 : times;
    for (int t = 0; t < times; ++t)
      subface_index = rot_mapping[subface_index];
  }



  template <int dim>
  inline void
  HangingNodes<dim>::rotate_face(
    int                                   times,
    unsigned int                          n_dofs_1d,
    std::vector<types::global_dof_index> &dofs) const
  {
    const unsigned int rot_mapping[4] = {2, 0, 3, 1};

    times = times % 4;
    times = times < 0 ? times + 4 : times;

    std::vector<types::global_dof_index> copy(dofs.size());
    for (int t = 0; t < times; ++t)
      {
        std::swap(copy, dofs);

        // Vertices
        for (unsigned int i = 0; i < 4; ++i)
          dofs[rot_mapping[i]] = copy[i];

        // Edges
        const unsigned int n_int  = n_dofs_1d - 2;
        unsigned int       offset = 4;
        for (unsigned int i = 0; i < n_int; ++i)
          {
            // Left edge
            dofs[offset + i] = copy[offset + 2 * n_int + (n_int - 1 - i)];
            // Right edge
            dofs[offset + n_int + i] =
              copy[offset + 3 * n_int + (n_int - 1 - i)];
            // Bottom edge
            dofs[offset + 2 * n_int + i] = copy[offset + n_int + i];
            // Top edge
            dofs[offset + 3 * n_int + i] = copy[offset + i];
          }

        // Interior points
        offset += 4 * n_int;

        for (unsigned int i = 0; i < n_int; ++i)
          for (unsigned int j = 0; j < n_int; ++j)
            dofs[offset + i * n_int + j] =
              copy[offset + j * n_int + (n_int - 1 - i)];
      }
  }



  template <int dim>
  inline unsigned int
  HangingNodes<dim>::line_dof_idx(int          local_line,
                                  unsigned int dof,
                                  unsigned int n_dofs_1d) const
  {
    unsigned int x, y, z;

    const unsigned int fe_degree = n_dofs_1d - 1;

    if (local_line < 8)
      {
        x = (local_line % 4 == 0) ? 0 : (local_line % 4 == 1) ? fe_degree : dof;
        y = (local_line % 4 == 2) ? 0 : (local_line % 4 == 3) ? fe_degree : dof;
        z = (local_line / 4) * fe_degree;
      }
    else
      {
        x = ((local_line - 8) % 2) * fe_degree;
        y = ((local_line - 8) / 2) * fe_degree;
        z = dof;
      }

    return n_dofs_1d * n_dofs_1d * z + n_dofs_1d * y + x;
  }



  template <int dim>
  inline void
  HangingNodes<dim>::transpose_face(
    const unsigned int                    fe_degree,
    std::vector<types::global_dof_index> &dofs) const
  {
    const std::vector<types::global_dof_index> copy(dofs);

    // Vertices
    dofs[1] = copy[2];
    dofs[2] = copy[1];

    // Edges
    const unsigned int n_int  = fe_degree - 1;
    unsigned int       offset = 4;
    for (unsigned int i = 0; i < n_int; ++i)
      {
        // Right edge
        dofs[offset + i] = copy[offset + 2 * n_int + i];
        // Left edge
        dofs[offset + n_int + i] = copy[offset + 3 * n_int + i];
        // Bottom edge
        dofs[offset + 2 * n_int + i] = copy[offset + i];
        // Top edge
        dofs[offset + 3 * n_int + i] = copy[offset + n_int + i];
      }

    // Interior
    offset += 4 * n_int;
    for (unsigned int i = 0; i < n_int; ++i)
      for (unsigned int j = 0; j < n_int; ++j)
        dofs[offset + i * n_int + j] = copy[offset + j * n_int + i];
  }



  template <int dim>
  void
  HangingNodes<dim>::transpose_subface_index(unsigned int &subface) const
  {
    if (subface == 1)
      subface = 2;
    else if (subface == 2)
      subface = 1;
  }

} // namespace internal

DEAL_II_NAMESPACE_CLOSE

#endif
