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

#include <deal.II/grid/connectivity.h>

#include <iomanip>
#include <set>

DEAL_II_NAMESPACE_OPEN


namespace Tet
{
  namespace
  {
    template <int dim>
    static std::array<CellTypeEntities, dim + 1>
    convert_to_crs(
      const std::vector<std::vector<std::vector<unsigned int>>> &entities_in)
    {
      std::array<CellTypeEntities, dim + 1> result;

      AssertDimension(dim, entities_in.size());

      for (int d = 0; d < dim; d++)
        {
          const auto &entities = entities_in[d];

          for (auto entity : entities)
            {
              for (auto vertex : entity)
                result[dim - d].vertices.push_back(vertex);

              result[dim - d].vertices_ptr.push_back(
                result[dim - d].vertices.size());
            }
        }

      return result;
    }
  } // namespace

  template <int dim>
  CellTypeBase<dim>::CellTypeBase(
    const std::string                                          name,
    const std::vector<std::vector<std::vector<unsigned int>>> &entities_in)
    : name(name)
    , entities(convert_to_crs<dim>(entities_in))
    , n_vertices_(entities[dim].vertices.size())
  {}

  template <int dim>
  unsigned int
  CellTypeBase<dim>::n_entities(const unsigned int d) const
  {
    AssertIndexRange(d, dim + 1);
    return entities[d].vertices_ptr.size() - 1;
  }

  template <int dim>
  unsigned int
  CellTypeBase<dim>::n_vertices()
  {
    return n_vertices_;
  }

  template <int dim>
  dealii::ArrayView<const unsigned int>
  CellTypeBase<dim>::vertices_of_entity(const unsigned int d,
                                        const unsigned int e) const
  {
    AssertIndexRange(d, dim + 1);
    return dealii::ArrayView<const unsigned int>(
      entities[d].vertices.data() + entities[d].vertices_ptr[e],
      entities[d].vertices_ptr[e + 1] - entities[d].vertices_ptr[e]);
  }

  template <int dim>
  std::string
  CellTypeBase<dim>::get_name() const
  {
    return name;
  }

  template <int dim>
  struct CellTypeTet : public CellTypeBase<dim>
  {
    CellTypeTet()
      : CellTypeBase<dim>(
          "tet",
          dim == 2 ? (std::vector<std::vector<std::vector<unsigned int>>>{
                       {{0, 1, 2}},             // cell
                       {{0, 1}, {1, 2}, {2, 0}} // edges
                     }) :
                     (std::vector<std::vector<std::vector<unsigned int>>>{
                       {{0, 1, 2, 3}},                                  // cell
                       {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}},    // faces
                       {{0, 1}, {1, 2}, {0, 2}, {0, 3}, {1, 3}, {2, 3}} // edges
                     }))
    {}
  };

  template <int dim>
  struct CellTypeQuad : public CellTypeBase<dim>
  {
    CellTypeQuad()
      : CellTypeBase<dim>("quad",
                          {
                            {{0, 1, 2, 3}},                  // cell
                            {{0, 1}, {1, 2}, {2, 3}, {3, 0}} // edges
                          })
    {
      AssertDimension(dim, 2);
    }
  };

  template <int dim>
  std::shared_ptr<CellTypeBase<dim>>
  CellTypeFactory<dim>::build(const CellTypeEnum type)
  {
    std::shared_ptr<CellTypeBase<dim>> result;

    switch (type)
      {
        // clang-format off
        case CellTypeEnum::tet:  result.reset(new CellTypeTet<dim>());  break;
        case CellTypeEnum::quad: result.reset(new CellTypeQuad<dim>()); break;
        // clang-format on
        default:
          Assert(false,
                 dealii::StandardExceptions::ExcMessage(
                   "Element type is not supported!"));
      }

    return result;
  }


} // namespace Tet

template class Tet::CellTypeBase<1>;
template class Tet::CellTypeBase<2>;
template class Tet::CellTypeBase<3>;

template class Tet::CellTypeFactory<1>;
template class Tet::CellTypeFactory<2>;
template class Tet::CellTypeFactory<3>;


namespace Tet
{
  namespace internal
  {
    void
    transpose(const CRS<unsigned int> &in, CRS<unsigned int> &out)
    {
      auto        in_   = in;
      auto &      col   = in_.col;
      const auto &ptr   = in_.ptr;
      auto &      col_t = out.col;
      auto &      ptr_t = out.ptr;

      for (unsigned int i = 0; i < ptr.size() - 1; i++)
        std::sort(col.data() + ptr[i], col.data() + ptr[i + 1]);

      col_t.clear();
      ptr_t.clear();

      col_t.reserve(col.size());
      ptr_t.reserve(
        col.size() == 0 ? 0 : *std::max_element(col.begin(), col.end()));

      std::vector<std::pair<unsigned int, unsigned int>> temp;
      temp.reserve(col.size());

      for (unsigned int i = 0; i < ptr.size() - 1; i++)
        for (std::size_t j = ptr[i]; j < ptr[i + 1]; j++)
          temp.emplace_back(col[j], i);

      std::sort(temp.begin(), temp.end());

      auto r = dealii::numbers::invalid_unsigned_int;

      for (unsigned int i = 0; i < temp.size(); i++)
        {
          if (r != temp[i].first)
            {
              r = temp[i].first;
              ptr_t.push_back(col_t.size());
            }

          col_t.push_back(temp[i].second);
        }

      ptr_t.push_back(col_t.size());
    }

    void
    transitive_closure(const CRS<unsigned int> &in_0,
                       const CRS<unsigned int> &in_1,
                       CRS<unsigned int> &      out)
    {
      const auto &col_0 = in_0.col;
      const auto &ptr_0 = in_0.ptr;
      const auto &col_1 = in_1.col;
      const auto &ptr_1 = in_1.ptr;
      auto &      col   = out.col;
      auto &      ptr   = out.ptr;

      col.clear();
      ptr.clear();

      ptr.push_back(0);

      for (unsigned int i_0 = 0; i_0 < ptr_0.size() - 1; i_0++)
        {
          std::vector<unsigned int> temp;

          for (std::size_t j_0 = ptr_0[i_0]; j_0 < ptr_0[i_0 + 1]; j_0++)
            {
              for (std::size_t j_1 = ptr_1[col_0[j_0]];
                   j_1 < ptr_1[col_0[j_0] + 1];
                   j_1++)
                if (i_0 != col_1[j_1])
                  temp.emplace_back(col_1[j_1]);
            }

          std::sort(temp.begin(), temp.end());
          temp.erase(std::unique(temp.begin(), temp.end()), temp.end());

          for (const auto i : temp)
            col.emplace_back(i);
          ptr.push_back(col.size());
        }
    }

    template <class StreamType>
    void
    print(const CRS<unsigned int> &in, StreamType &out)
    {
      unsigned w = 3;

      for (unsigned int i = 0; i < in.ptr.size() - 1; i++)
        for (unsigned int j = in.ptr[i]; j < in.ptr[i + 1]; j++)
          {
            if (j == in.ptr[i])
              out << std::setw(w) << in.ptr[i] << " ";
            else
              out << std::setw(w) << ""
                  << " ";
          }
      out << std::setw(w) << in.ptr.back() << std::endl;


      for (const auto i : in.col)
        out << std::setw(w) << i << " ";
      out << std::endl;
      out << std::endl;
    }

    template <int dim, class StreamType>
    void
    print(const std::array<std::array<CRS<unsigned int>, dim>, dim> &table,
          StreamType &                                               out)
    {
      for (unsigned int i = 0; i < dim; i++)
        for (unsigned int j = 0; j < dim; j++)
          {
            out << "Table " << i << " -> " << j << ":" << std::endl;
            print(table[i][j], out);
          }
    }

    template <int key_length, int dim>
    void
    build_entity_templated(
      const unsigned int                                     d,
      const std::vector<std::shared_ptr<CellTypeBase<dim>>> &cell_types,
      const std::vector<unsigned int> &                      cell_types_index,
      const CRS<unsigned int> &                              crs,
      CRS<unsigned int> &                                    crs_d,
      CRS<unsigned int> &                                    crs_0)
    {
      const std::vector<std::size_t> & cell_ptr      = crs.ptr;
      const std::vector<unsigned int> &cell_vertices = crs.col;
      std::vector<std::size_t> &       ptr_d         = crs_d.ptr;
      std::vector<unsigned int> &      col_d         = crs_d.col;
      std::vector<std::size_t> &       ptr_0 = crs_0.ptr = {};
      std::vector<unsigned int> &      col_0             = crs_0.col;

      std::vector<std::pair<std::array<unsigned int, key_length>, unsigned int>>
        keys;

      ptr_d.resize(cell_types_index.size() + 1);
      ptr_d[0] = 0;

      for (unsigned int c = 0, counter = 0; c < cell_types_index.size(); c++)
        {
          const auto &cell_type = cell_types[cell_types_index[c]];
          ptr_d[c + 1]          = ptr_d[c] + cell_type->n_entities(d);

          const dealii::ArrayView<const unsigned int> cell_vertice(
            cell_vertices.data() + cell_ptr[c], cell_ptr[c + 1] - cell_ptr[c]);

          for (unsigned int e = 0; e < cell_type->n_entities(d); e++)
            {
              const auto &local_entity_vertices =
                cell_type->vertices_of_entity(d, e);

              std::vector<unsigned int> global_entity_vertices(
                local_entity_vertices.size());

              for (unsigned int i = 0; i < local_entity_vertices.size(); i++)
                global_entity_vertices[i] =
                  cell_vertice[local_entity_vertices[i]];

              // TODO: save this permutation

              std::sort(global_entity_vertices.begin(),
                        global_entity_vertices.end());


              std::array<unsigned int, key_length> key;
              std::fill(key.begin(), key.end(), 0);
              for (unsigned int i = 0; i < global_entity_vertices.size(); i++)
                key[i] = global_entity_vertices[i] + 1;

              keys.emplace_back(key, counter++);
            }
        }

      std::sort(keys.begin(), keys.end());

      std::vector<std::pair<unsigned int, unsigned int>> keys_unique;
      keys_unique.reserve(keys.size());

      std::array<unsigned int, key_length> temp;
      std::fill(temp.begin(), temp.end(), 0);

      for (unsigned int i = 0, counter = dealii::numbers::invalid_unsigned_int;
           i < keys.size();
           i++)
        {
          if (temp != keys[i].first)
            {
              counter++;
              temp = keys[i].first;

              ptr_0.push_back(col_0.size());
              for (const auto j : keys[i].first)
                if (j != 0)
                  col_0.push_back(j - 1);
            }
          keys_unique.emplace_back(keys[i].second, counter);
        }
      ptr_0.push_back(col_0.size());

      std::sort(keys_unique.begin(), keys_unique.end());

      col_d.resize(keys_unique.size());
      for (unsigned int i = 0; i < keys_unique.size(); i++)
        col_d[i] = keys_unique[i].second;
    }

    template <int dim>
    void
    build_entity(
      const unsigned int                                     d,
      const std::vector<std::shared_ptr<CellTypeBase<dim>>> &cell_types,
      const std::vector<unsigned int> &                      cell_types_index,
      const CRS<unsigned int> &                              crs,
      CRS<unsigned int> &                                    crs_d,
      CRS<unsigned int> &                                    crs_0)
    {
      if (!(0 < d && d < dim))
        AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented());

      std::size_t key_length = 0;

      for (unsigned int c = 0; c < cell_types_index.size(); c++)
        {
          const auto &cell_type = cell_types[cell_types_index[c]];
          for (unsigned int e = 0; e < cell_type->n_entities(d); e++)
            key_length =
              std::max(key_length, cell_type->vertices_of_entity(d, e).size());
        }

      // clang-format off
    if(key_length == 2)
      build_entity_templated<2>(d, cell_types, cell_types_index, crs, crs_d, crs_0);
    else if(key_length == 3)
      build_entity_templated<3>(d, cell_types, cell_types_index, crs, crs_d, crs_0);
    else
      AssertThrow(false, dealii::StandardExceptions::ExcNotImplemented ());
      // clang-format on
    }


    template <int dim>
    void
    build_table(
      const std::vector<std::shared_ptr<CellTypeBase<dim>>> &cell_types,
      const std::vector<unsigned int> &                      cell_types_index,
      const CRS<unsigned int> &                              crs,
      std::array<std::array<CRS<unsigned int>, dim + 1>, dim + 1> &table)
    {
      table[dim][0] = crs;

      transpose(table[dim][0], table[0][dim]);
      transitive_closure(table[dim][0], table[0][dim], table[dim][dim]);

      for (unsigned int d = 1; d < dim; d++)
        {
          build_entity(d,
                       cell_types,
                       cell_types_index,
                       table[dim][0],
                       table[dim][d],
                       table[d][0]);
          transpose(table[d][0], table[0][d]);
          transpose(table[dim][d], table[d][dim]);
        }

      const auto fu = [](const auto &T_i0,
                         const auto &T_j0,
                         const auto &T_id,
                         const auto &T_dj,
                         auto &      T_ij) {
        T_ij.ptr = {0};
        T_ij.col = {};

        // loop over all i entities
        for (unsigned int i = 0; i < T_i0.ptr.size() - 1; ++i)
          {
            // collect the vertices of entity i
            std::set<unsigned int> set_i(T_i0.col.begin() + T_i0.ptr[i],
                                         T_i0.col.begin() + T_i0.ptr[i + 1]);

            // extract a cell
            const unsigned int d = T_id.col[T_id.ptr[i]];

            for (unsigned int j_ = T_dj.ptr[d]; j_ < T_dj.ptr[d + 1]; ++j_)
              {
                const unsigned int     j = T_dj.col[j_];
                std::set<unsigned int> set_j(T_j0.col.begin() + T_j0.ptr[j],
                                             T_j0.col.begin() +
                                               T_j0.ptr[j + 1]);

                std::set<int> intersect;

                std::set_intersection(set_i.begin(),
                                      set_i.end(),
                                      set_j.begin(),
                                      set_j.end(),
                                      std::inserter(intersect,
                                                    intersect.begin()));

                if (intersect.size() == set_j.size())
                  T_ij.col.push_back(j);
              }

            T_ij.ptr.push_back(T_ij.col.size());
          }
      };

      for (unsigned int i = 1; i < dim; i++)
        for (unsigned int j = 1; j < i; j++)
          fu(table[i][0],
             table[j][0],
             table[i][dim],
             table[dim][j],
             table[i][j]);
    }
  } // namespace internal

  template <int dim>
  void
  Connectivity<dim>::build(const std::vector<CellTypeEnum> &cell_types,
                           const std::vector<unsigned int> &cell_vertices)
  {
    std::set<CellTypeEnum> cell_types_set;
    for (const auto cell_type : cell_types)
      cell_types_set.insert(cell_type);

    std::map<CellTypeEnum, unsigned int>            cell_types_map;
    std::vector<std::shared_ptr<CellTypeBase<dim>>> cell_types_impl;

    for (const auto cell_type : cell_types_set)
      {
        cell_types_map[cell_type] = cell_types_impl.size();
        cell_types_impl.push_back(CellTypeFactory<dim>::build(cell_type));
      }

    std::vector<unsigned int> cell_types_indices(cell_types.size());
    std::vector<std::size_t>  cell_ptr(cell_types.size() + 1);
    cell_ptr[0] = 0;

    for (unsigned int i = 0; i < cell_types.size(); i++)
      {
        const unsigned int index = cell_types_map[cell_types[i]];
        cell_types_indices[i]    = index;
        cell_ptr[i + 1] = cell_ptr[i] + cell_types_impl[index]->n_vertices();
      }

    CRS<unsigned int> crs;
    crs.col = cell_vertices;
    crs.ptr = cell_ptr;

    internal::build_table(cell_types_impl, cell_types_indices, crs, table);
  }

  template <int dim>
  void
  Connectivity<dim>::print(std::ostream &out) const
  {
    internal::print<dim + 1>(table, out);
  }


} // namespace Tet

template class Tet::Connectivity<1>;
template class Tet::Connectivity<2>;
template class Tet::Connectivity<3>;


DEAL_II_NAMESPACE_CLOSE
