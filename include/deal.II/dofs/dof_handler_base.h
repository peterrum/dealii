// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2019 by the deal.II authors
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

#ifndef dealii_dof_handler_base_h
#define dealii_dof_handler_base_h



#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/std_cxx14/memory.h>

#include <deal.II/distributed/cell_data_transfer.templates.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/block_info.h>
#include <deal.II/dofs/deprecated_function_map.h>
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_iterator_selector.h>
#include <deal.II/dofs/dof_levels.h>
#include <deal.II/dofs/number_cache.h>

#include <deal.II/hp/dof_faces.h>
#include <deal.II/hp/dof_level.h>
#include <deal.II/hp/fe_collection.h>

#include <boost/serialization/split_member.hpp>

#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
template <int dim, int spacedim>
class FiniteElement;
template <int dim, int spacedim>
class Triangulation;

namespace internal
{
  namespace DoFHandlerImplementation
  {
    struct Implementation;

    namespace Policy
    {
      template <int dim, int spacedim>
      class PolicyBase;
      struct Implementation;
    } // namespace Policy
  }   // namespace DoFHandlerImplementation

  namespace DoFAccessorImplementation
  {
    struct Implementation;
  }

  namespace DoFCellAccessorImplementation
  {
    struct Implementation;
  }
} // namespace internal


#endif

// Forward declarations
#ifndef DOXYGEN
template <int dim, int spacedim>
class Triangulation;

namespace parallel
{
  namespace distributed
  {
    template <int dim, int spacedim, typename VectorType>
    class CellDataTransfer;
  }
} // namespace parallel

namespace internal
{
  namespace DoFHandlerImplementation
  {
    struct Implementation;

    namespace Policy
    {
      template <int dim, int spacedim>
      class PolicyBase;
      struct Implementation;
    } // namespace Policy
  }   // namespace DoFHandlerImplementation

  namespace hp
  {
    class DoFLevel;

    namespace DoFHandlerImplementation
    {
      struct Implementation;
    }
  } // namespace hp
} // namespace internal

namespace internal
{
  namespace DoFAccessorImplementation
  {
    struct Implementation;
  }

  namespace DoFCellAccessorImplementation
  {
    struct Implementation;
  }
} // namespace internal
#endif

/**
 * @author Wolfgang Bangerth, Markus Buerg, Timo Heister, Guido Kanschat, 1998, 1999, 2000, 2012
 * @author Wolfgang Bangerth, 2003, 2004, 2017, 2018
 * @author Oliver Kayser-Herold, 2003, 2004
 * @author Marc Fehling, 2018
 * @author Peter Munch, 2020
 */
template <int dim, int spacedim, typename T>
class DoFHandlerBase : public Subscriptor
{
  using ActiveSelector =
    dealii::internal::DoFHandlerImplementation::Iterators<T, false>;
  using LevelSelector =
    dealii::internal::DoFHandlerImplementation::Iterators<T, true>;

public:
  using cell_accessor = typename ActiveSelector::CellAccessor;

  using face_accessor = typename ActiveSelector::FaceAccessor;

  using line_iterator = typename ActiveSelector::line_iterator;

  using active_line_iterator = typename ActiveSelector::active_line_iterator;

  using quad_iterator = typename ActiveSelector::quad_iterator;

  using active_quad_iterator = typename ActiveSelector::active_quad_iterator;

  using hex_iterator = typename ActiveSelector::hex_iterator;

  using active_hex_iterator = typename ActiveSelector::active_hex_iterator;

  using active_cell_iterator = typename ActiveSelector::active_cell_iterator;

  using cell_iterator = typename ActiveSelector::cell_iterator;

  using face_iterator = typename ActiveSelector::face_iterator;

  using active_face_iterator = typename ActiveSelector::active_face_iterator;

  using level_cell_accessor = typename LevelSelector::CellAccessor;
  using level_face_accessor = typename LevelSelector::FaceAccessor;

  using level_cell_iterator = typename LevelSelector::cell_iterator;
  using level_face_iterator = typename LevelSelector::face_iterator;


  static const unsigned int dimension = dim;

  static const unsigned int space_dimension = spacedim;

  static const bool is_hp_dof_handler = T::is_hp_dof_handler;

  static const unsigned int default_fe_index =
    is_hp_dof_handler ? numbers::invalid_unsigned_int : 0;

  DoFHandlerBase()
    : tria(nullptr, typeid(*this).name())
    , faces(nullptr)
    , mg_faces(nullptr)
    , faces_hp(nullptr)
  {}

  DoFHandlerBase(const Triangulation<dim, spacedim> &tria)
    : tria(&tria, typeid(*this).name())
    , faces(nullptr)
    , mg_faces(nullptr)
    , faces_hp(nullptr)
  {
    if (is_hp_dof_handler)
      {
        this->setup_policy_and_listeners();
        this->create_active_fe_table();
      }
    else
      {
        this->setup_policy();
      }
  }

  // DoFHandlerBase(const DoFHandlerBase &) = delete;
  //
  // DoFHandlerBase &
  // operator=(const DoFHandlerBase &) = delete;

  virtual ~DoFHandlerBase();

  void
  initialize(const Triangulation<dim, spacedim> &tria,
             const FiniteElement<dim, spacedim> &fe);
  void
  initialize(const Triangulation<dim, spacedim> &   tria,
             const hp::FECollection<dim, spacedim> &fe);

  void
  set_fe(const FiniteElement<dim, spacedim> &fe);

  void
  set_fe(const hp::FECollection<dim, spacedim> &fe);

  void
  distribute_dofs(const FiniteElement<dim, spacedim> &fe);

  void
  distribute_dofs(const hp::FECollection<dim, spacedim> &fe);

  void
  set_active_fe_indices(const std::vector<unsigned int> &active_fe_indices);

  void
  get_active_fe_indices(std::vector<unsigned int> &active_fe_indices) const;

  DEAL_II_DEPRECATED
  void
  distribute_mg_dofs(const FiniteElement<dim, spacedim> &fe);

  DEAL_II_DEPRECATED
  void
  distribute_mg_dofs(const hp::FECollection<dim, spacedim> &fe);

  void
  distribute_mg_dofs();

  bool
  has_level_dofs() const;

  bool
  has_active_dofs() const;

  void
  initialize_local_block_info();

  void
  clear();

  void
  renumber_dofs(const std::vector<types::global_dof_index> &new_numbers);

  void
  renumber_dofs(const unsigned int                          level,
                const std::vector<types::global_dof_index> &new_numbers);

  unsigned int
  max_couplings_between_dofs() const;

  unsigned int
  max_couplings_between_boundary_dofs() const;

  cell_iterator
  begin(const unsigned int level = 0) const;

  active_cell_iterator
  begin_active(const unsigned int level = 0) const;

  cell_iterator
  end() const;

  cell_iterator
  end(const unsigned int level) const;

  active_cell_iterator
  end_active(const unsigned int level) const;

  level_cell_iterator
  begin_mg(const unsigned int level = 0) const;

  level_cell_iterator
  end_mg(const unsigned int level) const;

  level_cell_iterator
  end_mg() const;

  IteratorRange<cell_iterator>
  cell_iterators() const;

  IteratorRange<active_cell_iterator>
  active_cell_iterators() const;

  IteratorRange<level_cell_iterator>
  mg_cell_iterators() const;

  IteratorRange<cell_iterator>
  cell_iterators_on_level(const unsigned int level) const;

  IteratorRange<active_cell_iterator>
  active_cell_iterators_on_level(const unsigned int level) const;

  IteratorRange<level_cell_iterator>
  mg_cell_iterators_on_level(const unsigned int level) const;

  types::global_dof_index
  n_dofs() const;

  types::global_dof_index
  n_dofs(const unsigned int level) const;

  types::global_dof_index
  n_boundary_dofs() const;

  template <typename number>
  types::global_dof_index
  n_boundary_dofs(
    const std::map<types::boundary_id, const Function<spacedim, number> *>
      &boundary_ids) const;

  types::global_dof_index
  n_boundary_dofs(const std::set<types::boundary_id> &boundary_ids) const;

  virtual const BlockInfo &
  block_info() const;

  types::global_dof_index
  n_locally_owned_dofs() const;

  const IndexSet &
  locally_owned_dofs() const;

  const IndexSet &
  locally_owned_mg_dofs(const unsigned int level) const;

  std::vector<IndexSet>
  compute_locally_owned_dofs_per_processor() const;

  std::vector<types::global_dof_index>
  compute_n_locally_owned_dofs_per_processor() const;

  std::vector<IndexSet>
  compute_locally_owned_mg_dofs_per_processor(const unsigned int level) const;

  DEAL_II_DEPRECATED virtual const std::vector<IndexSet> &
  locally_owned_dofs_per_processor() const;

  DEAL_II_DEPRECATED virtual const std::vector<types::global_dof_index> &
  n_locally_owned_dofs_per_processor() const;

  DEAL_II_DEPRECATED virtual const std::vector<IndexSet> &
  locally_owned_mg_dofs_per_processor(const unsigned int level) const;

  const FiniteElement<dim, spacedim> &
  get_fe(const unsigned int index = 0) const;

  const hp::FECollection<dim, spacedim> &
  get_fe_collection() const;

  const Triangulation<dim, spacedim> &
  get_triangulation() const;

  virtual std::size_t
  memory_consumption() const;

  void
  prepare_for_serialization_of_active_fe_indices();

  void
  deserialize_active_fe_indices();

  struct ActiveFEIndexTransfer
  {
    std::map<const cell_iterator, const unsigned int> persisting_cells_fe_index;

    std::map<const cell_iterator, const unsigned int> refined_cells_fe_index;

    std::map<const cell_iterator, const unsigned int> coarsened_cells_fe_index;

    std::vector<unsigned int> active_fe_indices;

    std::unique_ptr<
      parallel::distributed::
        CellDataTransfer<dim, spacedim, std::vector<unsigned int>>>
      cell_data_transfer;
  };

  std::unique_ptr<ActiveFEIndexTransfer> active_fe_index_transfer;

  std::vector<boost::signals2::connection> tria_listeners;

  template <class Archive>
  void
  save(Archive &ar, const unsigned int version) const;

  template <class Archive>
  void
  load(Archive &ar, const unsigned int version);

  BOOST_SERIALIZATION_SPLIT_MEMBER()

  DeclException0(ExcNoFESelected);
  DeclException0(ExcInvalidBoundaryIndicator);
  DeclException1(ExcInvalidLevel,
                 int,
                 << "The given level " << arg1
                 << " is not in the valid range!");
  DeclException1(ExcNewNumbersNotConsecutive,
                 types::global_dof_index,
                 << "The given list of new dof indices is not consecutive: "
                 << "the index " << arg1 << " does not exist.");
  DeclException2(ExcInvalidFEIndex,
                 int,
                 int,
                 << "The mesh contains a cell with an active_fe_index of "
                 << arg1 << ", but the finite element collection only has "
                 << arg2 << " elements");

protected:
  BlockInfo block_info_object;

  SmartPointer<const Triangulation<dim, spacedim>, DoFHandler<dim, spacedim>>
    tria;

  hp::FECollection<dim, spacedim> fe_collection;

  std::unique_ptr<dealii::internal::DoFHandlerImplementation::Policy::
                    PolicyBase<dim, spacedim>>
    policy;

  dealii::internal::DoFHandlerImplementation::NumberCache number_cache;

  std::vector<dealii::internal::DoFHandlerImplementation::NumberCache>
    mg_number_cache;


  void
  clear_space();


  class MGVertexDoFs
  {
  public:
    MGVertexDoFs();

    void
    init(const unsigned int coarsest_level,
         const unsigned int finest_level,
         const unsigned int dofs_per_vertex);

    unsigned int
    get_coarsest_level() const;

    unsigned int
    get_finest_level() const;

    types::global_dof_index
    get_index(const unsigned int level,
              const unsigned int dof_number,
              const unsigned int dofs_per_vertex) const;

    void
    set_index(const unsigned int            level,
              const unsigned int            dof_number,
              const unsigned int            dofs_per_vertex,
              const types::global_dof_index index);

  private:
    unsigned int coarsest_level;

    unsigned int finest_level;

    std::unique_ptr<types::global_dof_index[]> indices;
  };

  void
  setup_policy();

  void
  setup_policy_and_listeners();

  void
  pre_refinement_action();

  void
  post_refinement_action();

  void
  pre_active_fe_index_transfer();

  void
  pre_distributed_active_fe_index_transfer();

  void
  post_active_fe_index_transfer();

  void
  post_distributed_active_fe_index_transfer();

  void
  post_distributed_serialization_of_active_fe_indices();

  void
  create_active_fe_table();

  void
  clear_mg_space();

  template <int structdim>
  types::global_dof_index
  get_dof_index(const unsigned int obj_level,
                const unsigned int obj_index,
                const unsigned int fe_index,
                const unsigned int local_index) const;

  template <int structdim>
  void
  set_dof_index(const unsigned int            obj_level,
                const unsigned int            obj_index,
                const unsigned int            fe_index,
                const unsigned int            local_index,
                const types::global_dof_index global_index) const;

  std::vector<types::global_dof_index> vertex_dofs;

  std::vector<unsigned int> vertex_dof_offsets; // for hp

  std::vector<MGVertexDoFs> mg_vertex_dofs;

  std::vector<
    std::unique_ptr<dealii::internal::DoFHandlerImplementation::DoFLevel<dim>>>
    levels;

  std::vector<
    std::unique_ptr<dealii::internal::DoFHandlerImplementation::DoFLevel<dim>>>
    mg_levels;

  std::vector<std::unique_ptr<dealii::internal::hp::DoFLevel>>
    levels_hp; // TODO: rename hp_levels

  std::unique_ptr<dealii::internal::DoFHandlerImplementation::DoFFaces<dim>>
    faces;

  std::unique_ptr<dealii::internal::DoFHandlerImplementation::DoFFaces<dim>>
    mg_faces;

  std::unique_ptr<dealii::internal::hp::DoFIndicesOnFaces<dim>>
    faces_hp; // TODO: rename hp_faces


  template <int, class, bool>
  friend class dealii::DoFAccessor;
  template <class, bool>
  friend class dealii::DoFCellAccessor;
  friend struct dealii::internal::DoFAccessorImplementation::Implementation;
  friend struct dealii::internal::DoFCellAccessorImplementation::Implementation;

  // Likewise for DoFLevel objects since they need to access the vertex dofs
  // in the functions that set and retrieve vertex dof indices.
  template <int>
  friend class dealii::internal::hp::DoFIndicesOnFacesOrEdges;
  friend struct dealii::internal::hp::DoFHandlerImplementation::Implementation;
  friend struct dealii::internal::DoFHandlerImplementation::Policy::
    Implementation;

  template <int, class, bool>
  friend class DoFAccessor;
  template <class, bool>
  friend class DoFCellAccessor;
  friend struct dealii::internal::DoFAccessorImplementation::Implementation;
  friend struct dealii::internal::DoFCellAccessorImplementation::Implementation;

  friend struct dealii::internal::DoFHandlerImplementation::Implementation;
  friend struct dealii::internal::DoFHandlerImplementation::Policy::
    Implementation;
};


namespace internal
{
  namespace hp
  {
    namespace DoFHandlerImplementation
    {
      /**
       * A class with the same purpose as the similarly named class of the
       * Triangulation class. See there for more information.
       */
      struct Implementation
      {
        template <int dim, int spacedim, typename T>
        static void
        ensure_absence_of_future_fe_indices(
          DoFHandlerBase<dim, spacedim, T> &dof_handler);



        template <int dim, int spacedim, typename T>
        static void
        reserve_space_release_space(
          DoFHandlerBase<dim, spacedim, T> &dof_handler);



        template <int dim, int spacedim, typename T>
        static void
        reserve_space_vertices(DoFHandlerBase<dim, spacedim, T> &dof_handler);



        template <int dim, int spacedim, typename T>
        static void
        reserve_space_cells(DoFHandlerBase<dim, spacedim, T> &dof_handler);



        template <int dim, int spacedim, typename T>
        static void
        reserve_space_faces(DoFHandlerBase<dim, spacedim, T> &dof_handler);



        template <int spacedim, typename T>
        static void
          reserve_space(dealii::DoFHandlerBase<1, spacedim, T> &dof_handler);



        template <int spacedim, typename T>
        static void
          reserve_space(dealii::DoFHandlerBase<2, spacedim, T> &dof_handler);



        template <int spacedim, typename T>
        static void
          reserve_space(dealii::DoFHandlerBase<3, spacedim, T> &dof_handler);



        template <int spacedim>
        static unsigned int
        max_couplings_between_dofs(
          const dealii::
            DoFHandlerBase<1, spacedim, dealii::hp::DoFHandler<1, spacedim>>
              &dof_handler);



        template <int spacedim>
        static unsigned int
        max_couplings_between_dofs(
          const dealii::
            DoFHandlerBase<2, spacedim, dealii::hp::DoFHandler<2, spacedim>>
              &dof_handler);



        template <int spacedim>
        static unsigned int
        max_couplings_between_dofs(
          const dealii::
            DoFHandlerBase<3, spacedim, dealii::hp::DoFHandler<3, spacedim>>
              &dof_handler);



        template <int dim, int spacedim>
        static void
        communicate_active_fe_indices(
          DoFHandlerBase<dim, spacedim, dealii::DoFHandler<dim, spacedim>>
            &dof_handler);

        template <int dim, int spacedim>
        static void
        communicate_active_fe_indices(
          DoFHandlerBase<dim, spacedim, dealii::hp::DoFHandler<dim, spacedim>>
            &dof_handler);



        template <int dim, int spacedim, typename T>
        static void
        collect_fe_indices_on_cells_to_be_refined(
          DoFHandlerBase<dim, spacedim, T> &dof_handler);



        template <int dim, int spacedim, typename T>
        static void
        distribute_fe_indices_on_refined_cells(
          DoFHandlerBase<dim, spacedim, T> &dof_handler);


        template <int dim, int spacedim>
        static unsigned int
        determine_fe_from_children(
          const std::vector<unsigned int> &        children_fe_indices,
          dealii::hp::FECollection<dim, spacedim> &fe_collection);
      };
    } // namespace DoFHandlerImplementation
  }   // namespace hp
} // namespace internal


namespace internal
{
  template <int dim, int spacedim>
  std::string
  policy_to_string(const dealii::internal::DoFHandlerImplementation::Policy::
                     PolicyBase<dim, spacedim> &policy);


  namespace DoFHandlerImplementation
  {
    struct Implementation
    {
      template <int spacedim, typename T>
      static unsigned int
      max_couplings_between_dofs(
        const DoFHandlerBase<1, spacedim, T> &dof_handler)
      {
        return std::min(static_cast<types::global_dof_index>(
                          3 * dof_handler.get_fe().dofs_per_vertex +
                          2 * dof_handler.get_fe().dofs_per_line),
                        dof_handler.n_dofs());
      }

      template <int spacedim, typename T>
      static unsigned int
      max_couplings_between_dofs(
        const DoFHandlerBase<2, spacedim, T> &dof_handler)
      {
        // get these numbers by drawing pictures
        // and counting...
        // example:
        //   |     |     |
        // --x-----x--x--X--
        //   |     |  |  |
        //   |     x--x--x
        //   |     |  |  |
        // --x--x--*--x--x--
        //   |  |  |     |
        //   x--x--x     |
        //   |  |  |     |
        // --X--x--x-----x--
        //   |     |     |
        // x = vertices connected with center vertex *;
        //   = total of 19
        // (the X vertices are connected with * if
        // the vertices adjacent to X are hanging
        // nodes)
        // count lines -> 28 (don't forget to count
        // mother and children separately!)
        types::global_dof_index max_couplings;
        switch (dof_handler.tria->max_adjacent_cells())
          {
            case 4:
              max_couplings = 19 * dof_handler.get_fe().dofs_per_vertex +
                              28 * dof_handler.get_fe().dofs_per_line +
                              8 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 5:
              max_couplings = 21 * dof_handler.get_fe().dofs_per_vertex +
                              31 * dof_handler.get_fe().dofs_per_line +
                              9 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 6:
              max_couplings = 28 * dof_handler.get_fe().dofs_per_vertex +
                              42 * dof_handler.get_fe().dofs_per_line +
                              12 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 7:
              max_couplings = 30 * dof_handler.get_fe().dofs_per_vertex +
                              45 * dof_handler.get_fe().dofs_per_line +
                              13 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 8:
              max_couplings = 37 * dof_handler.get_fe().dofs_per_vertex +
                              56 * dof_handler.get_fe().dofs_per_line +
                              16 * dof_handler.get_fe().dofs_per_quad;
              break;

            // the following numbers are not based on actual counting but by
            // extrapolating the number sequences from the previous ones (for
            // example, for dofs_per_vertex, the sequence above is 19, 21, 28,
            // 30, 37, and is continued as follows):
            case 9:
              max_couplings = 39 * dof_handler.get_fe().dofs_per_vertex +
                              59 * dof_handler.get_fe().dofs_per_line +
                              17 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 10:
              max_couplings = 46 * dof_handler.get_fe().dofs_per_vertex +
                              70 * dof_handler.get_fe().dofs_per_line +
                              20 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 11:
              max_couplings = 48 * dof_handler.get_fe().dofs_per_vertex +
                              73 * dof_handler.get_fe().dofs_per_line +
                              21 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 12:
              max_couplings = 55 * dof_handler.get_fe().dofs_per_vertex +
                              84 * dof_handler.get_fe().dofs_per_line +
                              24 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 13:
              max_couplings = 57 * dof_handler.get_fe().dofs_per_vertex +
                              87 * dof_handler.get_fe().dofs_per_line +
                              25 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 14:
              max_couplings = 63 * dof_handler.get_fe().dofs_per_vertex +
                              98 * dof_handler.get_fe().dofs_per_line +
                              28 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 15:
              max_couplings = 65 * dof_handler.get_fe().dofs_per_vertex +
                              103 * dof_handler.get_fe().dofs_per_line +
                              29 * dof_handler.get_fe().dofs_per_quad;
              break;
            case 16:
              max_couplings = 72 * dof_handler.get_fe().dofs_per_vertex +
                              114 * dof_handler.get_fe().dofs_per_line +
                              32 * dof_handler.get_fe().dofs_per_quad;
              break;

            default:
              Assert(false, ExcNotImplemented());
              max_couplings = 0;
          }
        return std::min(max_couplings, dof_handler.n_dofs());
      }

      template <int spacedim, typename T>
      static unsigned int
      max_couplings_between_dofs(
        const DoFHandlerBase<3, spacedim, T> &dof_handler)
      {
        // TODO:[?] Invent significantly better estimates than the ones in this
        // function

        // doing the same thing here is a
        // rather complicated thing, compared
        // to the 2d case, since it is hard
        // to draw pictures with several
        // refined hexahedra :-) so I
        // presently only give a coarse
        // estimate for the case that at most
        // 8 hexes meet at each vertex
        //
        // can anyone give better estimate
        // here?
        const unsigned int max_adjacent_cells =
          dof_handler.tria->max_adjacent_cells();

        types::global_dof_index max_couplings;
        if (max_adjacent_cells <= 8)
          max_couplings = 7 * 7 * 7 * dof_handler.get_fe().dofs_per_vertex +
                          7 * 6 * 7 * 3 * dof_handler.get_fe().dofs_per_line +
                          9 * 4 * 7 * 3 * dof_handler.get_fe().dofs_per_quad +
                          27 * dof_handler.get_fe().dofs_per_hex;
        else
          {
            Assert(false, ExcNotImplemented());
            max_couplings = 0;
          }

        return std::min(max_couplings, dof_handler.n_dofs());
      }

      template <int spacedim, typename T>
      static void reserve_space(DoFHandlerBase<1, spacedim, T> &dof_handler);

      template <int spacedim, typename T>
      static void reserve_space(DoFHandlerBase<2, spacedim, T> &dof_handler);

      template <int spacedim, typename T>
      static void reserve_space(DoFHandlerBase<3, spacedim, T> &dof_handler);

      template <int spacedim>
      static void reserve_space_mg(
        DoFHandlerBase<1, spacedim, DoFHandler<1, spacedim>> &dof_handler);

      template <int spacedim>
      static void reserve_space_mg(
        DoFHandlerBase<2, spacedim, DoFHandler<2, spacedim>> &dof_handler);

      template <int spacedim>
      static void reserve_space_mg(
        DoFHandlerBase<3, spacedim, DoFHandler<3, spacedim>> &dof_handler);

      template <int dim, int spacedim>
      static types::global_dof_index
      get_dof_index(
        const DoFHandlerBase<dim,
                             spacedim,
                             dealii::hp::DoFHandler<dim, spacedim>>
          &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<dim>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<dim>>
          &,
        const unsigned int obj_index,
        const unsigned int fe_index,
        const unsigned int local_index,
        const std::integral_constant<int, 1>);

      template <int dim, int spacedim>
      static types::global_dof_index
      get_dof_index(
        const DoFHandlerBase<dim,
                             spacedim,
                             dealii::hp::DoFHandler<dim, spacedim>>
          &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<dim>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<dim>>
          &,
        const unsigned int obj_index,
        const unsigned int fe_index,
        const unsigned int local_index,
        const std::integral_constant<int, 2>);

      template <int dim, int spacedim>
      static types::global_dof_index
      get_dof_index(
        const DoFHandlerBase<dim,
                             spacedim,
                             dealii::hp::DoFHandler<dim, spacedim>>
          &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<dim>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<dim>>
          &,
        const unsigned int obj_index,
        const unsigned int fe_index,
        const unsigned int local_index,
        const std::integral_constant<int, 3>);

      template <int spacedim>
      static types::global_dof_index
      get_dof_index(
        const DoFHandlerBase<1, spacedim, DoFHandler<1, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<1>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<1>>
          &,
        const unsigned int obj_index,
        const unsigned int fe_index,
        const unsigned int local_index,
        const std::integral_constant<int, 1>);

      template <int spacedim>
      static types::global_dof_index
      get_dof_index(
        const DoFHandlerBase<2, spacedim, DoFHandler<2, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<2>>
          &,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<2>>
          &                mg_faces,
        const unsigned int obj_index,
        const unsigned int fe_index,
        const unsigned int local_index,
        const std::integral_constant<int, 1>);

      template <int spacedim>
      static types::global_dof_index
      get_dof_index(
        const DoFHandlerBase<2, spacedim, DoFHandler<2, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<2>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<2>>
          &,
        const unsigned int obj_index,
        const unsigned int fe_index,
        const unsigned int local_index,
        const std::integral_constant<int, 2>);

      template <int spacedim>
      static types::global_dof_index
      get_dof_index(
        const DoFHandlerBase<3, spacedim, DoFHandler<3, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<3>>
          &,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<3>>
          &                mg_faces,
        const unsigned int obj_index,
        const unsigned int fe_index,
        const unsigned int local_index,
        const std::integral_constant<int, 1>);

      template <int spacedim>
      static types::global_dof_index
      get_dof_index(
        const DoFHandlerBase<3, spacedim, DoFHandler<3, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<3>>
          &,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<3>>
          &                mg_faces,
        const unsigned int obj_index,
        const unsigned int fe_index,
        const unsigned int local_index,
        const std::integral_constant<int, 2>);

      template <int spacedim>
      static types::global_dof_index
      get_dof_index(
        const DoFHandlerBase<3, spacedim, DoFHandler<3, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<3>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<3>>
          &,
        const unsigned int obj_index,
        const unsigned int fe_index,
        const unsigned int local_index,
        const std::integral_constant<int, 3>);

      template <int dim, int spacedim>
      static void
      set_dof_index(
        const DoFHandlerBase<dim,
                             spacedim,
                             dealii::hp::DoFHandler<dim, spacedim>>
          &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<dim>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<dim>>
          &,
        const unsigned int            obj_index,
        const unsigned int            fe_index,
        const unsigned int            local_index,
        const types::global_dof_index global_index,
        const std::integral_constant<int, 1>);

      template <int dim, int spacedim>
      static void
      set_dof_index(
        const DoFHandlerBase<dim,
                             spacedim,
                             dealii::hp::DoFHandler<dim, spacedim>>
          &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<dim>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<dim>>
          &,
        const unsigned int            obj_index,
        const unsigned int            fe_index,
        const unsigned int            local_index,
        const types::global_dof_index global_index,
        const std::integral_constant<int, 2>);

      template <int dim, int spacedim>
      static void
      set_dof_index(
        const DoFHandlerBase<dim,
                             spacedim,
                             dealii::hp::DoFHandler<dim, spacedim>>
          &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<dim>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<dim>>
          &,
        const unsigned int            obj_index,
        const unsigned int            fe_index,
        const unsigned int            local_index,
        const types::global_dof_index global_index,
        const std::integral_constant<int, 3>);

      template <int spacedim>
      static void
      set_dof_index(
        const DoFHandlerBase<1, spacedim, DoFHandler<1, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<1>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<1>>
          &,
        const unsigned int            obj_index,
        const unsigned int            fe_index,
        const unsigned int            local_index,
        const types::global_dof_index global_index,
        const std::integral_constant<int, 1>);

      template <int spacedim>
      static void
      set_dof_index(
        const DoFHandlerBase<2, spacedim, DoFHandler<2, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<2>>
          &,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<2>>
          &                           mg_faces,
        const unsigned int            obj_index,
        const unsigned int            fe_index,
        const unsigned int            local_index,
        const types::global_dof_index global_index,
        const std::integral_constant<int, 1>);

      template <int spacedim>
      static void
      set_dof_index(
        const DoFHandlerBase<2, spacedim, DoFHandler<2, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<2>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<2>>
          &,
        const unsigned int            obj_index,
        const unsigned int            fe_index,
        const unsigned int            local_index,
        const types::global_dof_index global_index,
        const std::integral_constant<int, 2>);

      template <int spacedim>
      static void
      set_dof_index(
        const DoFHandlerBase<3, spacedim, DoFHandler<3, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<3>>
          &,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<3>>
          &                           mg_faces,
        const unsigned int            obj_index,
        const unsigned int            fe_index,
        const unsigned int            local_index,
        const types::global_dof_index global_index,
        const std::integral_constant<int, 1>);

      template <int spacedim>
      static void
      set_dof_index(
        const DoFHandlerBase<3, spacedim, DoFHandler<3, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<3>>
          &,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<3>>
          &                           mg_faces,
        const unsigned int            obj_index,
        const unsigned int            fe_index,
        const unsigned int            local_index,
        const types::global_dof_index global_index,
        const std::integral_constant<int, 2>);

      template <int spacedim>
      static void
      set_dof_index(
        const DoFHandlerBase<3, spacedim, DoFHandler<3, spacedim>> &dof_handler,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<3>>
          &mg_level,
        const std::unique_ptr<internal::DoFHandlerImplementation::DoFFaces<3>>
          &,
        const unsigned int            obj_index,
        const unsigned int            fe_index,
        const unsigned int            local_index,
        const types::global_dof_index global_index,
        const std::integral_constant<int, 3>);
    };
  } // namespace DoFHandlerImplementation
} // namespace internal

template <int dim, int spacedim, typename T>
inline const FiniteElement<dim, spacedim> &
DoFHandlerBase<dim, spacedim, T>::get_fe(const unsigned int number) const
{
  Assert(fe_collection.size() > 0,
         ExcMessage("No finite element collection is associated with "
                    "this DoFHandler"));
  return fe_collection[number];
}



template <int dim, int spacedim, typename T>
inline const hp::FECollection<dim, spacedim> &
DoFHandlerBase<dim, spacedim, T>::get_fe_collection() const
{
  Assert(fe_collection.size() > 0,
         ExcMessage("No finite element collection is associated with "
                    "this DoFHandler"));
  return fe_collection;
}



template <int dim, int spacedim, typename T>
inline const BlockInfo &
DoFHandlerBase<dim, spacedim, T>::block_info() const
{
  Assert(T::is_hp_dof_handler == false, ExcNotImplemented());

  return block_info_object;
}



template <int dim, int spacedim, typename T>
types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::n_boundary_dofs() const
{
  Assert(!(dim == 2 && spacedim == 3) || T::is_hp_dof_handler == false,
         ExcNotImplemented());

  Assert(this->fe_collection.size() > 0, ExcNoFESelected());

  std::unordered_set<types::global_dof_index> boundary_dofs;
  std::vector<types::global_dof_index>        dofs_on_face;
  dofs_on_face.reserve(this->get_fe_collection().max_dofs_per_face());

  const IndexSet &owned_dofs = locally_owned_dofs();

  // loop over all faces to check whether they are at a
  // boundary. note that we need not take special care of single
  // lines in 3d (using @p{cell->has_boundary_lines}), since we do
  // not support boundaries of dimension dim-2, and so every
  // boundary line is also part of a boundary face.
  for (const auto &cell : this->active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      {
        for (auto f : GeometryInfo<dim>::face_indices())
          if (cell->at_boundary(f))
            {
              const unsigned int dofs_per_face = cell->get_fe().dofs_per_face;
              dofs_on_face.resize(dofs_per_face);

              cell->face(f)->get_dof_indices(dofs_on_face,
                                             cell->active_fe_index());
              for (unsigned int i = 0; i < dofs_per_face; ++i)
                {
                  const unsigned int global_idof_index = dofs_on_face[i];
                  if (owned_dofs.is_element(global_idof_index))
                    {
                      boundary_dofs.insert(global_idof_index);
                    }
                }
            }
      }
  return boundary_dofs.size();
}



template <int dim, int spacedim, typename T>
types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::n_boundary_dofs(
  const std::set<types::boundary_id> &boundary_ids) const
{
  Assert(!(dim == 2 && spacedim == 3) || T::is_hp_dof_handler == false,
         ExcNotImplemented());

  Assert(this->fe_collection.size() > 0, ExcNoFESelected());
  Assert(boundary_ids.find(numbers::internal_face_boundary_id) ==
           boundary_ids.end(),
         ExcInvalidBoundaryIndicator());

  // same as above, but with additional checks for set of boundary
  // indicators
  std::unordered_set<types::global_dof_index> boundary_dofs;
  std::vector<types::global_dof_index>        dofs_on_face;
  dofs_on_face.reserve(this->get_fe_collection().max_dofs_per_face());

  const IndexSet &owned_dofs = locally_owned_dofs();

  for (const auto &cell : this->active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      {
        for (auto f : GeometryInfo<dim>::face_indices())
          if (cell->at_boundary(f) &&
              (boundary_ids.find(cell->face(f)->boundary_id()) !=
               boundary_ids.end()))
            {
              const unsigned int dofs_per_face = cell->get_fe().dofs_per_face;
              dofs_on_face.resize(dofs_per_face);

              cell->face(f)->get_dof_indices(dofs_on_face,
                                             cell->active_fe_index());
              for (unsigned int i = 0; i < dofs_per_face; ++i)
                {
                  const unsigned int global_idof_index = dofs_on_face[i];
                  if (owned_dofs.is_element(global_idof_index))
                    {
                      boundary_dofs.insert(global_idof_index);
                    }
                }
            }
      }
  return boundary_dofs.size();
}


template <int dim, int spacedim, typename T>
template <typename number>
types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::n_boundary_dofs(
  const std::map<types::boundary_id, const Function<spacedim, number> *>
    &boundary_ids) const
{
  Assert(!(dim == 2 && spacedim == 3) || T::is_hp_dof_handler == false,
         ExcNotImplemented());

  // extract the set of boundary ids and forget about the function object
  // pointers
  std::set<types::boundary_id> boundary_ids_only;
  for (typename std::map<types::boundary_id,
                         const Function<spacedim, number> *>::const_iterator p =
         boundary_ids.begin();
       p != boundary_ids.end();
       ++p)
    boundary_ids_only.insert(p->first);

  // then just hand everything over to the other function that does the work
  return n_boundary_dofs(boundary_ids_only);
}



template <int dim, int spacedim, typename T>
inline types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::n_dofs() const
{
  return number_cache.n_global_dofs;
}



template <int dim, int spacedim, typename T>
inline types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::n_dofs(const unsigned int level) const
{
  Assert(has_level_dofs(),
         ExcMessage(
           "n_dofs(level) can only be called after distribute_mg_dofs()"));
  Assert(level < mg_number_cache.size(), ExcInvalidLevel(level));
  return mg_number_cache[level].n_global_dofs;
}



template <int dim, int spacedim, typename T>
inline bool
DoFHandlerBase<dim, spacedim, T>::has_level_dofs() const
{
  return this->mg_number_cache.size() > 0;
}



template <int dim, int spacedim, typename T>
inline bool
DoFHandlerBase<dim, spacedim, T>::has_active_dofs() const
{
  return this->number_cache.n_global_dofs > 0;
}



template <int dim, int spacedim, typename T>
types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::n_locally_owned_dofs() const
{
  return this->number_cache.n_locally_owned_dofs;
}



template <int dim, int spacedim, typename T>
const IndexSet &
DoFHandlerBase<dim, spacedim, T>::locally_owned_dofs() const
{
  return this->number_cache.locally_owned_dofs;
}



template <int dim, int spacedim, typename T>
const IndexSet &
DoFHandlerBase<dim, spacedim, T>::locally_owned_mg_dofs(
  const unsigned int level) const
{
  Assert(level < this->get_triangulation().n_global_levels(),
         ExcMessage("The given level index exceeds the number of levels "
                    "present in the triangulation"));
  Assert(
    this->mg_number_cache.size() == this->get_triangulation().n_global_levels(),
    ExcMessage(
      "The level dofs are not set up properly! Did you call distribute_mg_dofs()?"));
  return this->mg_number_cache[level].locally_owned_dofs;
}



template <int dim, int spacedim, typename T>
const std::vector<types::global_dof_index> &
DoFHandlerBase<dim, spacedim, T>::n_locally_owned_dofs_per_processor() const
{
  if (this->number_cache.n_locally_owned_dofs_per_processor.empty() &&
      this->number_cache.n_global_dofs > 0)
    {
      const_cast<dealii::internal::DoFHandlerImplementation::NumberCache &>(
        this->number_cache)
        .n_locally_owned_dofs_per_processor =
        compute_n_locally_owned_dofs_per_processor();
    }
  return this->number_cache.n_locally_owned_dofs_per_processor;
}



template <int dim, int spacedim, typename T>
const std::vector<IndexSet> &
DoFHandlerBase<dim, spacedim, T>::locally_owned_dofs_per_processor() const
{
  if (this->number_cache.locally_owned_dofs_per_processor.empty() &&
      this->number_cache.n_global_dofs > 0)
    {
      const_cast<dealii::internal::DoFHandlerImplementation::NumberCache &>(
        this->number_cache)
        .locally_owned_dofs_per_processor =
        compute_locally_owned_dofs_per_processor();
    }
  return this->number_cache.locally_owned_dofs_per_processor;
}



template <int dim, int spacedim, typename T>
const std::vector<IndexSet> &
DoFHandlerBase<dim, spacedim, T>::locally_owned_mg_dofs_per_processor(
  const unsigned int level) const
{
  Assert(level < this->get_triangulation().n_global_levels(),
         ExcMessage("The given level index exceeds the number of levels "
                    "present in the triangulation"));
  Assert(
    this->mg_number_cache.size() == this->get_triangulation().n_global_levels(),
    ExcMessage(
      "The level dofs are not set up properly! Did you call distribute_mg_dofs()?"));
  if (this->mg_number_cache[level].locally_owned_dofs_per_processor.empty() &&
      this->mg_number_cache[level].n_global_dofs > 0)
    {
      const_cast<dealii::internal::DoFHandlerImplementation::NumberCache &>(
        this->mg_number_cache[level])
        .locally_owned_dofs_per_processor =
        compute_locally_owned_mg_dofs_per_processor(level);
    }
  return this->mg_number_cache[level].locally_owned_dofs_per_processor;
}



template <int dim, int spacedim, typename T>
std::vector<types::global_dof_index>
DoFHandlerBase<dim, spacedim, T>::compute_n_locally_owned_dofs_per_processor()
  const
{
  const parallel::TriangulationBase<dim, spacedim> *tr =
    (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
      &this->get_triangulation()));
  if (tr != nullptr)
    return this->number_cache.get_n_locally_owned_dofs_per_processor(
      tr->get_communicator());
  else
    return this->number_cache.get_n_locally_owned_dofs_per_processor(
      MPI_COMM_SELF);
}



template <int dim, int spacedim, typename T>
std::vector<IndexSet>
DoFHandlerBase<dim, spacedim, T>::compute_locally_owned_dofs_per_processor()
  const
{
  const parallel::TriangulationBase<dim, spacedim> *tr =
    (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
      &this->get_triangulation()));
  if (tr != nullptr)
    return this->number_cache.get_locally_owned_dofs_per_processor(
      tr->get_communicator());
  else
    return this->number_cache.get_locally_owned_dofs_per_processor(
      MPI_COMM_SELF);
}



template <int dim, int spacedim, typename T>
std::vector<IndexSet>
DoFHandlerBase<dim, spacedim, T>::compute_locally_owned_mg_dofs_per_processor(
  const unsigned int level) const
{
  Assert(level < this->get_triangulation().n_global_levels(),
         ExcMessage("The given level index exceeds the number of levels "
                    "present in the triangulation"));
  Assert(
    this->mg_number_cache.size() == this->get_triangulation().n_global_levels(),
    ExcMessage(
      "The level dofs are not set up properly! Did you call distribute_mg_dofs()?"));
  const parallel::TriangulationBase<dim, spacedim> *tr =
    (dynamic_cast<const parallel::TriangulationBase<dim, spacedim> *>(
      &this->get_triangulation()));
  if (tr != nullptr)
    return this->mg_number_cache[level].get_locally_owned_dofs_per_processor(
      tr->get_communicator());
  else
    return this->mg_number_cache[level].get_locally_owned_dofs_per_processor(
      MPI_COMM_SELF);
}


template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::initialize(
  const Triangulation<dim, spacedim> &tria,
  const FiniteElement<dim, spacedim> &fe)
{
  this->initialize(tria, hp::FECollection<dim, spacedim>(fe));
}

template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::initialize(
  const Triangulation<dim, spacedim> &   tria,
  const hp::FECollection<dim, spacedim> &fe)
{
  if (is_hp_dof_handler)
    {
      this->clear();

      if (this->tria != &tria)
        {
          for (auto &connection : this->tria_listeners)
            connection.disconnect();
          this->tria_listeners.clear();

          this->tria = &tria;

          this->setup_policy_and_listeners();
        }

      this->create_active_fe_table();

      this->distribute_dofs(fe);
    }
  else
    {
      this->tria                       = &tria;
      this->faces                      = nullptr;
      this->number_cache.n_global_dofs = 0;

      this->setup_policy();

      this->distribute_dofs(fe);
    }
}


template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::set_fe(const FiniteElement<dim, spacedim> &fe)
{
  this->set_fe(hp::FECollection<dim, spacedim>(fe));
}

template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::set_fe(
  const hp::FECollection<dim, spacedim> &ff)
{
  Assert(
    this->tria != nullptr,
    ExcMessage(
      "You need to set the Triangulation in the DoFHandler using initialize() or "
      "in the constructor before you can distribute DoFs."));
  Assert(this->tria->n_levels() > 0,
         ExcMessage("The Triangulation you are using is empty!"));
  Assert(ff.size() > 0, ExcMessage("The hp::FECollection given is empty!"));

  // don't create a new object if the one we have is already appropriate
  if (this->fe_collection != ff)
    this->fe_collection = hp::FECollection<dim, spacedim>(ff);

  if (is_hp_dof_handler == true)
    {
      // ensure that the active_fe_indices vectors are initialized correctly
      this->create_active_fe_table();

      // make sure every processor knows the active_fe_indices
      // on both its own cells and all ghost cells
      dealii::internal::hp::DoFHandlerImplementation::Implementation::
        communicate_active_fe_indices(*this);

      // make sure that the fe collection is large enough to
      // cover all fe indices presently in use on the mesh
      for (const auto &cell : this->active_cell_iterators())
        if (!cell->is_artificial())
          Assert(cell->active_fe_index() < this->fe_collection.size(),
                 ExcInvalidFEIndex(cell->active_fe_index(),
                                   this->fe_collection.size()));
    }
}


/*------------------------ Cell iterator functions ------------------------*/

template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::cell_iterator
DoFHandlerBase<dim, spacedim, T>::begin(const unsigned int level) const
{
  typename Triangulation<dim, spacedim>::cell_iterator cell =
    this->get_triangulation().begin(level);
  if (cell == this->get_triangulation().end(level))
    return end(level);
  return cell_iterator(*cell, static_cast<const T *>(this));
}



template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator
DoFHandlerBase<dim, spacedim, T>::begin_active(const unsigned int level) const
{
  // level is checked in begin
  cell_iterator i = begin(level);
  if (i.state() != IteratorState::valid)
    return i;
  while (i->has_children())
    if ((++i).state() != IteratorState::valid)
      return i;
  return i;
}



template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::cell_iterator
DoFHandlerBase<dim, spacedim, T>::end() const
{
  return cell_iterator(&this->get_triangulation(),
                       -1,
                       -1,
                       static_cast<const T *>(this));
}


template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::cell_iterator
DoFHandlerBase<dim, spacedim, T>::end(const unsigned int level) const
{
  typename Triangulation<dim, spacedim>::cell_iterator cell =
    this->get_triangulation().end(level);
  if (cell.state() != IteratorState::valid)
    return end();
  return cell_iterator(*cell, static_cast<const T *>(this));
}


template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator
DoFHandlerBase<dim, spacedim, T>::end_active(const unsigned int level) const
{
  typename Triangulation<dim, spacedim>::cell_iterator cell =
    this->get_triangulation().end_active(level);
  if (cell.state() != IteratorState::valid)
    return active_cell_iterator(end());
  return active_cell_iterator(*cell, static_cast<const T *>(this));
}



template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator
DoFHandlerBase<dim, spacedim, T>::begin_mg(const unsigned int level) const
{
  // Assert(this->has_level_dofs(), ExcMessage("You can only iterate over mg "
  //     "levels if mg dofs got distributed."));
  typename Triangulation<dim, spacedim>::cell_iterator cell =
    this->get_triangulation().begin(level);
  if (cell == this->get_triangulation().end(level))
    return end_mg(level);
  return level_cell_iterator(*cell, static_cast<const T *>(this));
}


template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator
DoFHandlerBase<dim, spacedim, T>::end_mg(const unsigned int level) const
{
  // Assert(this->has_level_dofs(), ExcMessage("You can only iterate over mg "
  //     "levels if mg dofs got distributed."));
  typename Triangulation<dim, spacedim>::cell_iterator cell =
    this->get_triangulation().end(level);
  if (cell.state() != IteratorState::valid)
    return end();
  return level_cell_iterator(*cell, static_cast<const T *>(this));
}


template <int dim, int spacedim, typename T>
typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator
DoFHandlerBase<dim, spacedim, T>::end_mg() const
{
  return level_cell_iterator(&this->get_triangulation(),
                             -1,
                             -1,
                             static_cast<const T *>(this));
}



template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::cell_iterator>
DoFHandlerBase<dim, spacedim, T>::cell_iterators() const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::cell_iterator>(begin(), end());
}


template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator>
DoFHandlerBase<dim, spacedim, T>::active_cell_iterators() const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator>(
    begin_active(), end());
}



template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator>
DoFHandlerBase<dim, spacedim, T>::mg_cell_iterators() const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator>(begin_mg(),
                                                                    end_mg());
}



template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::cell_iterator>
DoFHandlerBase<dim, spacedim, T>::cell_iterators_on_level(
  const unsigned int level) const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::cell_iterator>(begin(level),
                                                              end(level));
}



template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator>
DoFHandlerBase<dim, spacedim, T>::active_cell_iterators_on_level(
  const unsigned int level) const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::active_cell_iterator>(
    begin_active(level), end_active(level));
}



template <int dim, int spacedim, typename T>
IteratorRange<typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator>
DoFHandlerBase<dim, spacedim, T>::mg_cell_iterators_on_level(
  const unsigned int level) const
{
  return IteratorRange<
    typename DoFHandlerBase<dim, spacedim, T>::level_cell_iterator>(
    begin_mg(level), end_mg(level));
}



//---------------------------------------------------------------------------


template <int dim, int spacedim, typename T>
inline const Triangulation<dim, spacedim> &
DoFHandlerBase<dim, spacedim, T>::get_triangulation() const
{
  Assert(tria != nullptr,
         ExcMessage("This DoFHandler object has not been associated "
                    "with a triangulation."));
  return *tria;
}


template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::distribute_dofs(
  const FiniteElement<dim, spacedim> &fe)
{
  this->distribute_dofs(hp::FECollection<dim, spacedim>(fe));
}


template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::distribute_mg_dofs(
  const FiniteElement<dim, spacedim> &fe)
{
  (void)fe;
  this->distribute_mg_dofs();
}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::distribute_mg_dofs(
  const hp::FECollection<dim, spacedim> &fe)
{
  (void)fe;
  this->distribute_mg_dofs();
}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::set_active_fe_indices(
  const std::vector<unsigned int> &active_fe_indices)
{
  Assert(active_fe_indices.size() == this->get_triangulation().n_active_cells(),
         ExcDimensionMismatch(active_fe_indices.size(),
                              this->get_triangulation().n_active_cells()));

  this->create_active_fe_table();
  // we could set the values directly, since they are stored as
  // protected data of this object, but for simplicity we use the
  // cell-wise access. this way we also have to pass some debug-mode
  // tests which we would have to duplicate ourselves otherwise
  for (const auto &cell : this->active_cell_iterators())
    if (cell->is_locally_owned())
      cell->set_active_fe_index(active_fe_indices[cell->active_cell_index()]);
}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::get_active_fe_indices(
  std::vector<unsigned int> &active_fe_indices) const
{
  active_fe_indices.resize(this->get_triangulation().n_active_cells());

  // we could try to extract the values directly, since they are
  // stored as protected data of this object, but for simplicity we
  // use the cell-wise access.
  for (const auto &cell : this->active_cell_iterators())
    if (!cell->is_artificial())
      active_fe_indices[cell->active_cell_index()] = cell->active_fe_index();
}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::
  prepare_for_serialization_of_active_fe_indices()
{
#ifndef DEAL_II_WITH_P4EST
  Assert(false,
         ExcMessage(
           "You are attempting to use a functionality that is only available "
           "if deal.II was configured to use p4est, but cmake did not find a "
           "valid p4est library."));
#else
  Assert(
    (dynamic_cast<const parallel::distributed::Triangulation<dim, spacedim> *>(
       &this->get_triangulation()) != nullptr),
    ExcMessage(
      "This functionality requires a parallel::distributed::Triangulation object."));

  // Finite elements need to be assigned to each cell by calling
  // distribute_dofs() first to make this functionality available.
  if (this->fe_collection.size() > 0)
    {
      Assert(active_fe_index_transfer == nullptr, ExcInternalError());

      active_fe_index_transfer =
        std_cxx14::make_unique<ActiveFEIndexTransfer>();

      // Create transfer object and attach to it.
      const auto *distributed_tria = dynamic_cast<
        const parallel::distributed::Triangulation<dim, spacedim> *>(
        &this->get_triangulation());

      active_fe_index_transfer->cell_data_transfer = std_cxx14::make_unique<
        parallel::distributed::
          CellDataTransfer<dim, spacedim, std::vector<unsigned int>>>(
        *distributed_tria,
        /*transfer_variable_size_data=*/false,
        [this](const std::vector<unsigned int> &children_fe_indices) {
          return dealii::internal::hp::DoFHandlerImplementation::
            Implementation::determine_fe_from_children<dim, spacedim>(
              children_fe_indices, this->fe_collection);
        });

      // If we work on a p::d::Triangulation, we have to transfer all
      // active fe indices since ownership of cells may change.

      // Gather all current active_fe_indices
      this->get_active_fe_indices(active_fe_index_transfer->active_fe_indices);

      // Attach to transfer object
      active_fe_index_transfer->cell_data_transfer->prepare_for_serialization(
        active_fe_index_transfer->active_fe_indices);
    }
#endif
}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::deserialize_active_fe_indices()
{
#ifndef DEAL_II_WITH_P4EST
  Assert(false,
         ExcMessage(
           "You are attempting to use a functionality that is only available "
           "if deal.II was configured to use p4est, but cmake did not find a "
           "valid p4est library."));
#else
  Assert(
    (dynamic_cast<const parallel::distributed::Triangulation<dim, spacedim> *>(
       &this->get_triangulation()) != nullptr),
    ExcMessage(
      "This functionality requires a parallel::distributed::Triangulation object."));

  // Finite elements need to be assigned to each cell by calling
  // distribute_dofs() first to make this functionality available.
  if (this->fe_collection.size() > 0)
    {
      Assert(active_fe_index_transfer == nullptr, ExcInternalError());

      active_fe_index_transfer =
        std_cxx14::make_unique<ActiveFEIndexTransfer>();

      // Create transfer object and attach to it.
      const auto *distributed_tria = dynamic_cast<
        const parallel::distributed::Triangulation<dim, spacedim> *>(
        &this->get_triangulation());

      active_fe_index_transfer->cell_data_transfer = std_cxx14::make_unique<
        parallel::distributed::
          CellDataTransfer<dim, spacedim, std::vector<unsigned int>>>(
        *distributed_tria,
        /*transfer_variable_size_data=*/false,
        [this](const std::vector<unsigned int> &children_fe_indices) {
          return dealii::internal::hp::DoFHandlerImplementation::
            Implementation::determine_fe_from_children<dim, spacedim>(
              children_fe_indices, this->fe_collection);
        });

      // Unpack active_fe_indices.
      active_fe_index_transfer->active_fe_indices.resize(
        this->get_triangulation().n_active_cells(),
        numbers::invalid_unsigned_int);
      active_fe_index_transfer->cell_data_transfer->deserialize(
        active_fe_index_transfer->active_fe_indices);

      // Update all locally owned active_fe_indices.
      this->set_active_fe_indices(active_fe_index_transfer->active_fe_indices);

      // Update active_fe_indices on ghost cells.
      dealii::internal::hp::DoFHandlerImplementation::Implementation::
        communicate_active_fe_indices(*this);

      // Free memory.
      active_fe_index_transfer.reset();
    }
#endif
}



template <int dim, int spacedim, typename T>
template <int structdim>
types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::get_dof_index(
  const unsigned int obj_level,
  const unsigned int obj_index,
  const unsigned int fe_index,
  const unsigned int local_index) const
{
  if (T::is_hp_dof_handler == true)
    {
      Assert(false, ExcNotImplemented());
      return numbers::invalid_dof_index;
    }

  return internal::DoFHandlerImplementation::Implementation::get_dof_index(
    *this,
    this->mg_levels[obj_level],
    this->mg_faces,
    obj_index,
    fe_index,
    local_index,
    std::integral_constant<int, structdim>());
}



template <int dim, int spacedim, typename T>
template <int structdim>
void
DoFHandlerBase<dim, spacedim, T>::set_dof_index(
  const unsigned int            obj_level,
  const unsigned int            obj_index,
  const unsigned int            fe_index,
  const unsigned int            local_index,
  const types::global_dof_index global_index) const
{
  if (T::is_hp_dof_handler == true)
    {
      Assert(false, ExcNotImplemented());
      return;
    }

  internal::DoFHandlerImplementation::Implementation::set_dof_index(
    *this,
    this->mg_levels[obj_level],
    this->mg_faces,
    obj_index,
    fe_index,
    local_index,
    global_index,
    std::integral_constant<int, structdim>());
}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::clear_mg_space()
{
  this->mg_levels.clear();
  this->mg_faces.reset();

  std::vector<MGVertexDoFs> tmp;

  std::swap(this->mg_vertex_dofs, tmp);

  this->mg_number_cache.clear();
}



template <int dim, int spacedim, typename T>
std::size_t
DoFHandlerBase<dim, spacedim, T>::memory_consumption() const
{
  if (is_hp_dof_handler)
    {
      std::size_t mem =
        (MemoryConsumption::memory_consumption(this->tria) +
         MemoryConsumption::memory_consumption(this->fe_collection) +
         MemoryConsumption::memory_consumption(this->tria) +
         MemoryConsumption::memory_consumption(this->levels_hp) +
         MemoryConsumption::memory_consumption(*this->faces_hp) +
         MemoryConsumption::memory_consumption(this->number_cache) +
         MemoryConsumption::memory_consumption(this->vertex_dofs) +
         MemoryConsumption::memory_consumption(this->vertex_dof_offsets));
      for (unsigned int i = 0; i < this->levels_hp.size(); ++i)
        mem += MemoryConsumption::memory_consumption(*this->levels_hp[i]);
      mem += MemoryConsumption::memory_consumption(*this->faces_hp);

      return mem;
    }
  else
    {
      std::size_t mem =
        (MemoryConsumption::memory_consumption(this->tria) +
         MemoryConsumption::memory_consumption(this->fe_collection) +
         MemoryConsumption::memory_consumption(this->block_info_object) +
         MemoryConsumption::memory_consumption(this->levels) +
         MemoryConsumption::memory_consumption(*this->faces) +
         MemoryConsumption::memory_consumption(this->faces) +
         sizeof(this->number_cache) +
         MemoryConsumption::memory_consumption(this->n_dofs()) +
         MemoryConsumption::memory_consumption(this->vertex_dofs));
      for (unsigned int i = 0; i < this->levels.size(); ++i)
        mem += MemoryConsumption::memory_consumption(*this->levels[i]);

      for (unsigned int level = 0; level < this->mg_levels.size(); ++level)
        mem += this->mg_levels[level]->memory_consumption();

      if (this->mg_faces != nullptr)
        mem += MemoryConsumption::memory_consumption(*this->mg_faces);

      for (unsigned int i = 0; i < this->mg_vertex_dofs.size(); ++i)
        mem += sizeof(MGVertexDoFs) +
               (1 + this->mg_vertex_dofs[i].get_finest_level() -
                this->mg_vertex_dofs[i].get_coarsest_level()) *
                 sizeof(types::global_dof_index);

      return mem;
    }
}


template <int dim, int spacedim, typename T>
unsigned int
DoFHandlerBase<dim, spacedim, T>::max_couplings_between_boundary_dofs() const
{
  Assert(this->fe_collection.size() > 0, ExcNoFESelected());

  switch (dim)
    {
      case 1:
        return this->fe_collection.max_dofs_per_vertex();
      case 2:
        return (3 * this->fe_collection.max_dofs_per_vertex() +
                2 * this->fe_collection.max_dofs_per_line());
      case 3:
        // we need to take refinement of one boundary face into
        // consideration here; in fact, this function returns what
        // #max_coupling_between_dofs<2> returns
        //
        // we assume here, that only four faces meet at the boundary;
        // this assumption is not justified and needs to be fixed some
        // time. fortunately, omitting it for now does no harm since
        // the matrix will cry foul if its requirements are not
        // satisfied
        return (19 * this->fe_collection.max_dofs_per_vertex() +
                28 * this->fe_collection.max_dofs_per_line() +
                8 * this->fe_collection.max_dofs_per_quad());
      default:
        Assert(false, ExcNotImplemented());
        return 0;
    }
}



template <int dim, int spacedim, typename T>
inline types::global_dof_index
DoFHandlerBase<dim, spacedim, T>::MGVertexDoFs::get_index(
  const unsigned int level,
  const unsigned int dof_number,
  const unsigned int dofs_per_vertex) const
{
  Assert((level >= coarsest_level) && (level <= finest_level),
         ExcInvalidLevel(level));
  return indices[dofs_per_vertex * (level - coarsest_level) + dof_number];
}



template <int dim, int spacedim, typename T>
inline void
DoFHandlerBase<dim, spacedim, T>::MGVertexDoFs::set_index(
  const unsigned int            level,
  const unsigned int            dof_number,
  const unsigned int            dofs_per_vertex,
  const types::global_dof_index index)
{
  Assert((level >= coarsest_level) && (level <= finest_level),
         ExcInvalidLevel(level));
  indices[dofs_per_vertex * (level - coarsest_level) + dof_number] = index;
}



template <int dim, int spacedim, typename T>
DoFHandlerBase<dim, spacedim, T>::MGVertexDoFs::MGVertexDoFs()
  : coarsest_level(numbers::invalid_unsigned_int)
  , finest_level(0)
{}



template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::MGVertexDoFs::init(
  const unsigned int cl,
  const unsigned int fl,
  const unsigned int dofs_per_vertex)
{
  coarsest_level = cl;
  finest_level   = fl;

  if (coarsest_level <= finest_level)
    {
      const unsigned int n_levels  = finest_level - coarsest_level + 1;
      const unsigned int n_indices = n_levels * dofs_per_vertex;

      indices = std_cxx14::make_unique<types::global_dof_index[]>(n_indices);
      std::fill(indices.get(),
                indices.get() + n_indices,
                numbers::invalid_dof_index);
    }
  else
    indices.reset();
}



template <int dim, int spacedim, typename T>
unsigned int
DoFHandlerBase<dim, spacedim, T>::MGVertexDoFs::get_coarsest_level() const
{
  return coarsest_level;
}



template <int dim, int spacedim, typename T>
unsigned int
DoFHandlerBase<dim, spacedim, T>::MGVertexDoFs::get_finest_level() const
{
  return finest_level;
}



template <int dim, int spacedim, typename T>
template <class Archive>
void
DoFHandlerBase<dim, spacedim, T>::save(Archive &ar, const unsigned int) const
{
  if (is_hp_dof_handler)
    {
      ar & this->vertex_dofs;
      ar & this->vertex_dof_offsets;
      ar & this->number_cache;
      ar & this->mg_number_cache;

      // some versions of gcc have trouble with loading vectors of
      // std::unique_ptr objects because std::unique_ptr does not
      // have a copy constructor. do it one level at a time
      const unsigned int n_levels = this->levels_hp.size();
      ar &               n_levels;
      for (unsigned int i = 0; i < n_levels; ++i)
        ar & this->levels_hp[i];

      // boost dereferences a nullptr when serializing a nullptr
      // at least up to 1.65.1. This causes problems with clang-5.
      // Therefore, work around it.
      bool faces_is_nullptr = (this->faces_hp.get() == nullptr);
      ar & faces_is_nullptr;
      if (!faces_is_nullptr)
        ar & this->faces_hp;

      // write out the number of triangulation cells and later check during
      // loading that this number is indeed correct; same with something that
      // identifies the policy
      const unsigned int n_cells = this->tria->n_cells();
      std::string        policy_name =
        dealii::internal::policy_to_string(*this->policy);

      ar &n_cells &policy_name;
    }
  else
    {
      ar & this->block_info_object;
      ar & this->vertex_dofs;
      ar & this->number_cache;

      // some versions of gcc have trouble with loading vectors of
      // std::unique_ptr objects because std::unique_ptr does not
      // have a copy constructor. do it one level at a time
      unsigned int n_levels = this->levels.size();
      ar &         n_levels;
      for (unsigned int i = 0; i < this->levels.size(); ++i)
        ar & this->levels[i];

      // boost dereferences a nullptr when serializing a nullptr
      // at least up to 1.65.1. This causes problems with clang-5.
      // Therefore, work around it.
      bool faces_is_nullptr = (this->faces.get() == nullptr);
      ar & faces_is_nullptr;
      if (!faces_is_nullptr)
        ar & this->faces;

      // write out the number of triangulation cells and later check during
      // loading that this number is indeed correct; same with something that
      // identifies the FE and the policy
      unsigned int n_cells     = this->tria->n_cells();
      std::string  fe_name     = this->get_fe(0).get_name();
      std::string  policy_name = internal::policy_to_string(*this->policy);

      ar &n_cells &fe_name &policy_name;
    }
}



template <int dim, int spacedim, typename T>
template <class Archive>
void
DoFHandlerBase<dim, spacedim, T>::load(Archive &ar, const unsigned int)
{
  if (is_hp_dof_handler)
    {
      ar & this->vertex_dofs;
      ar & this->vertex_dof_offsets;
      ar & this->number_cache;
      ar & this->mg_number_cache;

      // boost::serialization can restore pointers just fine, but if the
      // pointer object still points to something useful, that object is not
      // destroyed and we end up with a memory leak. consequently, first delete
      // previous content before re-loading stuff
      this->levels_hp.clear();
      this->faces_hp.reset();

      // some versions of gcc have trouble with loading vectors of
      // std::unique_ptr objects because std::unique_ptr does not
      // have a copy constructor. do it one level at a time
      unsigned int size;
      ar &         size;
      this->levels_hp.resize(size);
      for (unsigned int i = 0; i < size; ++i)
        {
          std::unique_ptr<dealii::internal::hp::DoFLevel> level;
          ar &                                            level;
          this->levels_hp[i] = std::move(level);
        }

      // Workaround for nullptr, see in save().
      bool faces_is_nullptr = true;
      ar & faces_is_nullptr;
      if (!faces_is_nullptr)
        ar & this->faces_hp;

      // these are the checks that correspond to the last block in the save()
      // function
      unsigned int n_cells;
      std::string  policy_name;

      ar &n_cells &policy_name;

      AssertThrow(
        n_cells == this->tria->n_cells(),
        ExcMessage(
          "The object being loaded into does not match the triangulation "
          "that has been stored previously."));
      AssertThrow(
        policy_name == dealii::internal::policy_to_string(*this->policy),
        ExcMessage("The policy currently associated with this DoFHandler (" +
                   dealii::internal::policy_to_string(*this->policy) +
                   ") does not match the one that was associated with the "
                   "DoFHandler previously stored (" +
                   policy_name + ")."));
    }
  else
    {
      ar & this->block_info_object;
      ar & this->vertex_dofs;
      ar & this->number_cache;

      // boost::serialization can restore pointers just fine, but if the
      // pointer object still points to something useful, that object is not
      // destroyed and we end up with a memory leak. consequently, first delete
      // previous content before re-loading stuff
      this->levels.clear();
      this->faces.reset();

      // some versions of gcc have trouble with loading vectors of
      // std::unique_ptr objects because std::unique_ptr does not
      // have a copy constructor. do it one level at a time
      unsigned int size;
      ar &         size;
      this->levels.resize(size);
      for (unsigned int i = 0; i < this->levels.size(); ++i)
        {
          std::unique_ptr<internal::DoFHandlerImplementation::DoFLevel<dim>>
              level;
          ar &level;
          this->levels[i] = std::move(level);
        }

      // Workaround for nullptr, see in save().
      bool faces_is_nullptr = true;
      ar & faces_is_nullptr;
      if (!faces_is_nullptr)
        ar & this->faces;

      // these are the checks that correspond to the last block in the save()
      // function
      unsigned int n_cells;
      std::string  fe_name;
      std::string  policy_name;

      ar &n_cells &fe_name &policy_name;

      AssertThrow(
        n_cells == this->tria->n_cells(),
        ExcMessage(
          "The object being loaded into does not match the triangulation "
          "that has been stored previously."));
      AssertThrow(
        fe_name == this->get_fe(0).get_name(),
        ExcMessage(
          "The finite element associated with this DoFHandler does not match "
          "the one that was associated with the DoFHandler previously stored."));
      AssertThrow(policy_name == internal::policy_to_string(*this->policy),
                  ExcMessage(
                    "The policy currently associated with this DoFHandler (" +
                    internal::policy_to_string(*this->policy) +
                    ") does not match the one that was associated with the "
                    "DoFHandler previously stored (" +
                    policy_name + ")."));
    }
}

template <int dim, int spacedim, typename T>
unsigned int
DoFHandlerBase<dim, spacedim, T>::max_couplings_between_dofs() const
{
  Assert(this->fe_collection.size() > 0, ExcNoFESelected());
  return internal::DoFHandlerImplementation::Implementation::
    max_couplings_between_dofs(*this);
}

template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::clear_space()
{
  if (is_hp_dof_handler)
    {
      this->levels_hp.clear();
      this->faces_hp.reset();

      this->vertex_dofs        = std::vector<types::global_dof_index>();
      this->vertex_dof_offsets = std::vector<unsigned int>();
    }
  else
    {
      this->levels.clear();
      this->faces.reset();

      std::vector<types::global_dof_index> tmp;
      std::swap(this->vertex_dofs, tmp);

      this->number_cache.clear();
    }
}


template <int dim, int spacedim, typename T>
void
DoFHandlerBase<dim, spacedim, T>::clear()
{
  if (is_hp_dof_handler)
    {
      // release memory
      this->clear_space();
    }
  else
    {
      // release memory
      this->clear_space();
      this->clear_mg_space();
    }
}



DEAL_II_NAMESPACE_CLOSE

#endif
