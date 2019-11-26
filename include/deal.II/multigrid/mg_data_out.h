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

#ifndef dealii_mg_data_out_h
#define dealii_mg_data_out_h

//#include <deal.II/multigrid/mg_base.h>
#include <deal.II/base/mg_level_object.h>

#include <deal.II/numerics/data_out.h>

DEAL_II_NAMESPACE_OPEN

/*

- test subdomain id
- test level
- no dofhandler
- with dofhandler
- more than one component
- parallel
- matrix-free
- trilinos

 */


/*!@addtogroup mg */
/*@{*/


template <int dim, typename DoFHandlerType = DoFHandler<dim>>
class SingleLevelDataOut : public DataOut<dim, DoFHandlerType>
{
public:
  SingleLevelDataOut()
    : level_subdomain_id(-1)
    , level(-1)
  {}

  SingleLevelDataOut(const unsigned int level_subdomain_id, unsigned int level)
    : level_subdomain_id(level_subdomain_id)
    , level(level)
  {}

  void
  reinit(const unsigned int level_subdomain_id, unsigned int level)
  {
    this->level_subdomain_id = level_subdomain_id;
    this->level              = level;
  }


  virtual typename DataOut<dim>::cell_iterator
  first_cell()
  {
    typename DataOut<dim>::cell_iterator cell = this->dofs->begin(level);
    while ((cell != this->dofs->end(level)) &&
           (cell->level_subdomain_id() != level_subdomain_id))
      ++cell;

    if (cell == this->dofs->end(level))
      return this->dofs->end();

    return cell;
  }

  virtual typename DataOut<dim>::cell_iterator
  next_cell(const typename DataOut<dim>::cell_iterator &old_cell)
  {
    if (old_cell != this->dofs->end(level))
      {
        typename DataOut<dim>::cell_iterator cell = old_cell;
        ++cell;
        while ((cell != this->dofs->end(level)) &&
               (cell->level_subdomain_id() != level_subdomain_id))
          ++cell;
        if (cell == this->dofs->end(level))
          return this->dofs->end();

        return cell;
      }
    else
      return this->dofs->end();
  }

  virtual typename DataOut<dim>::cell_iterator
  first_locally_owned_cell()
  {
    return first_cell();
  }


  virtual typename DataOut<dim>::cell_iterator
  next_locally_owned_cell(const typename DataOut<dim>::cell_iterator &old_cell)
  {
    return next_cell(old_cell);
  }



private:
  unsigned int level_subdomain_id;
  unsigned int level;
};


/**
 * A specialized version of DataOut<dim> that supports output of cell
 * and dof data on multigrid levels.
 */
template <int dim>
class LevelDataOut : public DataOut<dim>
{
public:
  /**
   * Construct this output class. We will only output cells with the given
   * @p subdomain_id and between level @p min_lvl and @p max_lvl. The default
   * for @p max_lvl will output up to the finest existing level.
   */
  LevelDataOut(const unsigned int subdomain_id,
               unsigned int       min_lvl = 0,
               unsigned int       max_lvl = numbers::invalid_unsigned_int)
    : subdomain_id(subdomain_id)
    , min_lvl(min_lvl)
    , max_lvl(max_lvl)
  {
    Assert(min_lvl <= max_lvl, ExcMessage("invalid levels in LvlDataOut"));
  }

  /**
   * Returns the lowest level that will be output.
   */
  unsigned int
  get_min_level() const
  {
    return min_lvl;
  }

  /**
   * Returns the finest level that will be output. This value might be
   * numbers::invalid_unsigned_int denoting that all levels up to the finest
   * one will be output.
   */
  unsigned int
  get_max_level() const
  {
    return max_lvl;
  }

  /**
   * Create a Vector<double> for each level with as many entries as
   * cells. This can be used to output cell-wise data using
   * add_data_vector(MGLevelObject<VectorType>, std::string)
   *
   * @param[out] result The filled set of vectors
   */
  void
  make_level_cell_data_vector(MGLevelObject<Vector<double>> &result)
  {
    // Note: we need to use n_levels, not n_global_levels here!
    const unsigned int real_max_lvl =
      std::min(this->triangulation->n_levels() - 1, max_lvl);
    result.resize(min_lvl, real_max_lvl);
    for (unsigned int lvl = min_lvl; lvl <= real_max_lvl; ++lvl)
      result[lvl].reinit(this->triangulation->n_cells(lvl));
  }

  /**
   * Create a ghosted vector on each level to be used for DoFData for
   * the given @p dof_handler.
   */
  template <typename VectorType>
  void
  make_level_dof_data_vector(MGLevelObject<VectorType> &result,
                             const DoFHandler<dim> &    dof_handler)
  {
    // Note: we need to use n_global_levels so we construct the same number
    // of parallel vectors on each rank.
    const unsigned int real_max_lvl =
      std::min(this->triangulation->n_global_levels() - 1, max_lvl);
    result.resize(min_lvl, real_max_lvl);
    for (unsigned int lvl = min_lvl; lvl <= real_max_lvl; ++lvl)
      {
        result[lvl].reinit(dof_handler.n_dofs(lvl));
      }
  }

  /**
   * Create a ghosted vector on each level to be used for DoFData for
   * the given @p dof_handler.
   */
  template <typename VectorType>
  void
  make_level_dof_data_vector(MGLevelObject<VectorType> &result,
                             const DoFHandler<dim> &    dof_handler,
                             const MPI_Comm             mpi_comm)
  {
    // Note: we need to use n_global_levels so we construct the same number
    // of parallel vectors on each rank.
    const unsigned int real_max_lvl =
      std::min(this->triangulation->n_global_levels(), max_lvl);
    result.resize(min_lvl, real_max_lvl);
    for (unsigned int lvl = min_lvl; lvl < real_max_lvl; ++lvl)
      { /*
         IndexSet relevant;
         DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                       lvl,
                                                       relevant);
         IndexSet owned = dof_handler.locally_owned_mg_dofs(lvl);
         result[lvl].reinit(owned, relevant, mpi_comm);*/
      }
  }

  /**
   * Return the first cell including non-owned cells (required by our base
   * class).
   */
  virtual typename DataOut<dim>::cell_iterator
  first_cell() // const
  {
    typename DataOut<dim>::cell_iterator cell =
      this->triangulation->begin(min_lvl);

    if (cell == end_iterator())
      return this->triangulation->end();

    return cell;
  }

  /**
   * Return the next cell including non-owned cells (required by our base
   * class).
   */
  virtual typename DataOut<dim>::cell_iterator
  next_cell(const typename DataOut<dim>::cell_iterator &old_cell) // const
  {
    if (old_cell != end_iterator())
      {
        typename DataOut<dim>::cell_iterator cell = old_cell;
        ++cell;
        if (cell == end_iterator())
          return this->triangulation->end();

        return cell;
      }
    else
      return this->triangulation->end();
  }

  /**
   * Return the first cell we want to output (required by our base class)
   */
  virtual typename DataOut<dim>::cell_iterator
  first_locally_owned_cell() // const
  {
    typename DataOut<dim>::cell_iterator cell = first_cell();
    while ((cell != end_iterator()) &&
           (cell->level_subdomain_id() != subdomain_id))
      ++cell;
    if (cell == end_iterator())
      return this->triangulation->end();
    return cell;
  }

  /**
   * Return the next cell we want to output (required by our base class)
   */
  virtual typename DataOut<dim>::cell_iterator
  next_locally_owned_cell(
    const typename DataOut<dim>::cell_iterator &old_cell) // const
  {
    typename DataOut<dim>::cell_iterator cell = next_cell(old_cell);
    while ((cell != end_iterator()) &&
           (cell->level_subdomain_id() != subdomain_id))
      ++cell;
    if (cell == end_iterator())
      return this->triangulation->end();
    return cell;
  }

protected:
  /**
   * Helper function returning the end iterator (computed from max_level).
   */
  typename DataOut<dim>::cell_iterator
  end_iterator() const
  {
    if (max_lvl == numbers::invalid_unsigned_int)
      return this->triangulation->end();
    else
      return this->triangulation->end(max_lvl);
  }
  /**
   * The subdomain_id we want to output on
   */
  const unsigned int subdomain_id;
  /**
   * The minimum level where cells are output from.
   */
  const unsigned int min_lvl;
  /**
   * The maximum level where cells are output from.
   */
  const unsigned int max_lvl;
};



/**
 *
 *
 * @author
 */
template <int dim, typename DoFHandlerType = DoFHandler<dim>>
class MGDataOut : public DataOutInterface<dim, DoFHandlerType::space_dimension>
{
public:
  static const unsigned int dimension       = dim;
  static const unsigned int space_dimension = DoFHandlerType::space_dimension;

  MGDataOut(
    const Triangulation<DoFHandlerType::dimension,
                        DoFHandlerType::space_dimension> &triangulation);

  //  void
  // attach_dof_handler(const DoFHandlerType &);

  // dof_data:
  template <class VectorType>
  void
  add_data_vector(const DoFHandlerType &           dof_handler,
                  const MGLevelObject<VectorType> &data,
                  const std::string &              name);

  /*

    cell_data

    template <class VectorType>
void add_data_vector (
    const MGLevelObject<VectorType>               &data,
                      const std::string &name);
  */

  // TODO, vector data

  /**
   * Add the level_subdomain_id as a cell-wise output with name
   * "level_subdomain_id" to the output.
   */
  void
  add_level_subdomain_id();


  virtual void
  build_patches(const unsigned int n_subdivisions = 0);
  /*  virtual void
build_patches(const Mapping<DoFHandlerType::dimension,
                            DoFHandlerType::space_dimension> &mapping,
              const unsigned int     n_subdivisions = 0,
              const CurvedCellRegion curved_region  = curved_boundary);
  */



  /**
   * Create a Vector<double> for each level with as many entries as
   * cells. This can be used to output cell-wise data using
   * add_data_vector(MGLevelObject<VectorType>, std::string)
   *
   * @param[out] result The filled set of vectors
   */
  void
  make_level_cell_data_vector(MGLevelObject<Vector<double>> &result)
  {
    /*    // Note: we need to use n_levels, not n_global_levels here!
    const unsigned int real_max_lvl =
    std::min(this->triangulation->n_levels()-1, max_lvl); result.resize(min_lvl,
    real_max_lvl); for (unsigned int lvl = min_lvl; lvl <= real_max_lvl; ++lvl)
    result[lvl].reinit(this->triangulation->n_cells(lvl));*/
  }

  /**
   * Create a ghosted vector on each level to be used for DoFData for
   * the given @p dof_handler.
   */
  template <typename VectorType>
  void
  make_level_dof_data_vector(MGLevelObject<VectorType> &result,
                             const DoFHandler<dim> &    dof_handler)
  {
    // Note: we need to use n_global_levels so we construct the same number
    // of parallel vectors on each rank.
    const unsigned int min_lvl = 0;
    const unsigned int max_lvl = this->triangulation->n_global_levels() - 1;

    result.resize(min_lvl, max_lvl);
    for (unsigned int lvl = min_lvl; lvl <= max_lvl; ++lvl)
      {
        result[lvl].reinit(dof_handler.n_dofs(lvl));
      }
  }

protected:
  void
  add_level();

  using patch_t = DataOutBase::Patch<dim, space_dimension>;

  std::vector<patch_t> patches;

  /**
   *
   */
  virtual const std::vector<DataOutBase::Patch<dim, space_dimension>> &
  get_patches() const
  {
    return patches;
  }


  /**
   *
   */
  virtual std::vector<std::string>
  get_dataset_names() const;

private:
  SmartPointer<const Triangulation<dimension, space_dimension>> triangulation;
  MGLevelObject<SingleLevelDataOut<dim, DoFHandlerType>>        data_out;
};


/*@}*/


// -------------------- template and inline functions ------------------------

template <int dim, typename DoFHandlerType>
MGDataOut<dim, DoFHandlerType>::MGDataOut(
  const Triangulation<DoFHandlerType::dimension,
                      DoFHandlerType::space_dimension> &triangulation)
  :

  triangulation(&triangulation)
{
  const unsigned int max_lvl = this->triangulation->n_global_levels() - 1;
  data_out.resize(0, max_lvl);

  const unsigned int id = this->triangulation->locally_owned_subdomain();

  data_out.apply([&](const unsigned int                       level,
                     SingleLevelDataOut<dim, DoFHandlerType> &object) {
    object.reinit(id, level);
    object.attach_triangulation(triangulation);
  });
}

template <int dim, typename DoFHandlerType>
template <class VectorType>
void
MGDataOut<dim, DoFHandlerType>::add_data_vector(
  const DoFHandlerType &           dof_handler,
  const MGLevelObject<VectorType> &data,
  const std::string &              name)
{
  data_out.apply([&](const unsigned int                       level,
                     SingleLevelDataOut<dim, DoFHandlerType> &object) {
    object.add_data_vector(dof_handler, data[level], name);
  });
}

template <int dim, typename DoFHandlerType>
void
MGDataOut<dim, DoFHandlerType>::build_patches(const unsigned int n_subdivisions)
{
  patches.clear();
  unsigned int offset = 0;

  data_out.apply([&](const unsigned int                       level,
                     SingleLevelDataOut<dim, DoFHandlerType> &object) {
    const unsigned int offset = patches.size();
    object.build_patches(n_subdivisions);
    auto new_patches = object.get_patches();
    auto it =
      patches.insert(patches.end(), new_patches.begin(), new_patches.end());

    // now shift cell and neighbor info:
    while (it != patches.end())
      {
        it->patch_index += offset;

        for (unsigned int n = 0; n < GeometryInfo<dim>::faces_per_cell; ++n)
          if (it->neighbors[n] != patch_t::no_neighbor)
            it->neighbors[n] += offset;

        ++it;
      }
  });
}


template <int dim, typename DoFHandlerType>
std::vector<std::string>

MGDataOut<dim, DoFHandlerType>::get_dataset_names() const
{
  return data_out[0].get_dataset_names();
}



DEAL_II_NAMESPACE_CLOSE

#endif
