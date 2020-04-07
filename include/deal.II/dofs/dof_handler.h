// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2020 by the deal.II authors
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

#ifndef dealii_dof_handler_h
#define dealii_dof_handler_h



#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/distributed/tria_base.h>

#include <deal.II/dofs/block_info.h>
#include <deal.II/dofs/deprecated_function_map.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_faces.h>
#include <deal.II/dofs/dof_handler_base.h>
#include <deal.II/dofs/dof_iterator_selector.h>
#include <deal.II/dofs/dof_levels.h>
#include <deal.II/dofs/number_cache.h>

#include <deal.II/hp/fe_collection.h>

#include <boost/serialization/split_member.hpp>

#include <map>
#include <memory>
#include <set>
#include <vector>

DEAL_II_NAMESPACE_OPEN

/**
 * Given a triangulation and a description of a finite element, this
 * class enumerates degrees of freedom on all vertices, edges, faces,
 * and cells of the triangulation. As a result, it also provides a
 * <i>basis</i> for a discrete space $V_h$ whose elements are finite
 * element functions defined on each cell by a FiniteElement object.
 * This class satisfies the
 * @ref ConceptMeshType "MeshType concept"
 * requirements.
 *
 * It is first used in the step-2 tutorial program.
 *
 * For each vertex, line, quad, etc, this class stores a list of the indices
 * of degrees of freedom living on this object. These indices refer to the
 * unconstrained degrees of freedom, i.e. constrained degrees of freedom are
 * numbered in the same way as unconstrained ones, and are only later
 * eliminated.  This leads to the fact that indices in global vectors and
 * matrices also refer to all degrees of freedom and some kind of condensation
 * is needed to restrict the systems of equations to the unconstrained degrees
 * of freedom only. The actual layout of storage of the indices is described
 * in the dealii::internal::DoFHandlerImplementation::DoFLevel class
 * documentation.
 *
 * The class offers iterators to traverse all cells, in much the same way as
 * the Triangulation class does. Using the begin() and end() functions (and
 * companions, like begin_active()), one can obtain iterators to walk over
 * cells, and query the degree of freedom structures as well as the
 * triangulation data. These iterators are built on top of those of the
 * Triangulation class, but offer the additional information on degrees of
 * freedom functionality compared to pure triangulation iterators. The order
 * in which dof iterators are presented by the <tt>++</tt> and <tt>\--</tt>
 * operators is the same as that for the corresponding iterators traversing
 * the triangulation on which this DoFHandler is constructed.
 *
 * The <tt>spacedim</tt> parameter has to be used if one wants to solve
 * problems on surfaces. If not specified, this parameter takes the default
 * value <tt>=dim</tt> implying that we want to solve problems in a domain
 * whose dimension equals the dimension of the space in which it is embedded.
 *
 *
 * <h3>Distribution of indices for degrees of freedom</h3>
 *
 * The degrees of freedom (`dofs') are distributed on the given triangulation
 * by the function distribute_dofs(). It gets passed a finite element object
 * describing how many degrees of freedom are located on vertices, lines, etc.
 * It traverses the triangulation cell by cell and numbers the dofs of that
 * cell if not yet numbered. For non-multigrid algorithms, only active cells
 * are considered. Active cells are defined to be those cells which have no
 * children, i.e. they are the most refined ones.
 *
 * Since the triangulation is traversed starting with the cells of the
 * coarsest active level and going to more refined levels, the lowest numbers
 * for dofs are given to the largest cells as well as their bounding lines and
 * vertices, with the dofs of more refined cells getting higher numbers.
 *
 * This numbering implies very large bandwidths of the resulting matrices and
 * is thus vastly suboptimal for some solution algorithms. For this reason,
 * the DoFRenumbering class offers several algorithms to reorder the dof
 * numbering according. See there for a discussion of the implemented
 * algorithms.
 *
 *
 * <h3>Interaction with distributed meshes</h3>
 *
 * Upon construction, this class takes a reference to a triangulation object.
 * In most cases, this will be a reference to an object of type Triangulation,
 * i.e. the class that represents triangulations that entirely reside on a
 * single processor. However, it can also be of type
 * parallel::distributed::Triangulation (see, for example, step-32, step-40
 * and in particular the
 * @ref distributed
 * module) in which case the DoFHandler object will proceed to only manage
 * degrees of freedom on locally owned and ghost cells. This process is
 * entirely transparent to the used.
 *
 *
 * <h3>User defined renumbering schemes</h3>
 *
 * The DoFRenumbering class offers a number of renumbering schemes like the
 * Cuthill-McKee scheme. Basically, the function sets up an array in which for
 * each degree of freedom we store the new index this DoF should have after
 * renumbering. Using this array, the renumber_dofs() function of the present
 * class is called, which actually performs the change from old DoF indices to
 * the ones given in the array. In some cases, however, a user may want to
 * compute their own renumbering order; in this case, one can allocate an array
 * with one element per degree of freedom and fill it with the number that the
 * respective degree of freedom shall be assigned. This number may, for
 * example, be obtained by sorting the support points of the degrees of
 * freedom in downwind direction.  Then call the
 * <tt>renumber_dofs(vector<types::global_dof_index>)</tt> function with the
 * array, which converts old into new degree of freedom indices.
 *
 *
 * <h3>Serializing (loading or storing) DoFHandler objects</h3>
 *
 * Like many other classes in deal.II, the DoFHandler class can stream its
 * contents to an archive using BOOST's serialization facilities. The data so
 * stored can later be retrieved again from the archive to restore the
 * contents of this object. This facility is frequently used to save the state
 * of a program to disk for possible later resurrection, often in the context
 * of checkpoint/restart strategies for long running computations or on
 * computers that aren't very reliable (e.g. on very large clusters where
 * individual nodes occasionally fail and then bring down an entire MPI job).
 *
 * The model for doing so is similar for the DoFHandler class as it is for the
 * Triangulation class (see the section in the general documentation of that
 * class). In particular, the load() function does not exactly restore the
 * same state as was stored previously using the save() function. Rather, the
 * function assumes that you load data into a DoFHandler object that is
 * already associated with a triangulation that has a content that matches the
 * one that was used when the data was saved. Likewise, the load() function
 * assumes that the current object is already associated with a finite element
 * object that matches the one that was associated with it when data was
 * saved; the latter can be achieved by calling DoFHandler::distribute_dofs()
 * using the same kind of finite element before re-loading data from the
 * serialization archive.
 *
 * @ingroup dofs
 * @author Wolfgang Bangerth, Markus Buerg, Timo Heister, Guido Kanschat
 * @date 1998, 1999, 2000, 2012
 *
 * @note Task is delegated to the base class DoFHandlerBase.
 */
template <int dim, int spacedim = dim>
class DoFHandler
  : public DoFHandlerBase<dim, spacedim, DoFHandler<dim, spacedim>>
{
public:
  /**
   * Standard constructor, not initializing any data. After constructing an
   * object with this constructor, use initialize() to make a valid
   * DoFHandler.
   */
  DoFHandler(const bool is_hp_dof_handler = false);

  /**
   * Constructor. Take @p tria as the triangulation to work on.
   */
  DoFHandler(const Triangulation<dim, spacedim> &tria,
             const bool                          is_hp_dof_handler = false);

  /**
   * Copy constructor. DoFHandler objects are large and expensive.
   * They should not be copied, in particular not by accident, but
   * rather deliberately constructed. As a consequence, this constructor
   * is explicitly removed from the interface of this class.
   */
  DoFHandler(const DoFHandler &) = delete;

  /**
   * Copy operator. DoFHandler objects are large and expensive.
   * They should not be copied, in particular not by accident, but
   * rather deliberately constructed. As a consequence, this operator
   * is explicitly removed from the interface of this class.
   */
  DoFHandler &
  operator=(const DoFHandler &) = delete;
};

namespace hp
{
  /**
   * Manage the distribution and numbering of the degrees of freedom for hp-
   * FEM algorithms. This class satisfies the
   * @ref ConceptMeshType "MeshType concept"
   * requirements.
   *
   * The purpose of this class is to allow for an enumeration of degrees of
   * freedom in the same way as the ::DoFHandler class, but it allows to use a
   * different finite element on every cell. To this end, one assigns an
   * <code>active_fe_index</code> to every cell that indicates which element
   * within a collection of finite elements (represented by an object of type
   * hp::FECollection) is the one that lives on this cell. The class then
   * enumerates the degree of freedom associated with these finite elements on
   * each cell of a triangulation and, if possible, identifies degrees of
   * freedom at the interfaces of cells if they match. If neighboring cells
   * have degrees of freedom along the common interface that do not immediate
   * match (for example, if you have $Q_2$ and $Q_3$ elements meeting at a
   * common face), then one needs to compute constraints to ensure that the
   * resulting finite element space on the mesh remains conforming.
   *
   * The whole process of working with objects of this type is explained in
   * step-27. Many of the algorithms this class implements are described in
   * the
   * @ref hp_paper "hp paper".
   *
   *
   * <h3>Active FE indices and their behavior under mesh refinement</h3>
   *
   * The typical workflow for using this class is to create a mesh, assign an
   * active FE index to every active cell, calls
   * hp__DoFHandler::distribute_dofs(), and then assemble a linear system and
   * solve a problem on this finite element space. However, one can skip
   * assigning active FE indices upon mesh refinement in certain
   * circumstances. In particular, the following rules apply:
   * - Upon mesh refinement, child cells inherit the active FE index of
   *   the parent.
   * - When coarsening cells, the (now active) parent cell will be assigned
   *   an active FE index that is determined from its (no longer active)
   *   children, following the FiniteElementDomination logic: Out of the set of
   *   elements previously assigned to the former children, we choose the one
   *   dominated by all children for the parent cell. If none was found, we pick
   *   the most dominant element in the whole collection that is dominated by
   *   all former children. See hp::FECollection::find_dominated_fe_extended()
   *   for further information on this topic.
   *
   * @note Finite elements need to be assigned to each cell by either calling
   * set_fe() or distribute_dofs() first to make this functionality available.
   *
   *
   * <h3>Active FE indices and parallel meshes</h3>
   *
   * When this class is used with either a parallel::shared::Triangulation
   * or a parallel::distributed::Triangulation, you can only set active
   * FE indices on cells that are locally owned,
   * using a call such as <code>cell-@>set_active_fe_index(...)</code>.
   * On the other hand, setting the active FE index on ghost
   * or artificial cells is not allowed.
   *
   * Ghost cells do acquire the information what element
   * is active on them, however: whenever
   * you call hp__DoFHandler::distribute_dofs(), all processors that
   * participate in the parallel mesh exchange information in such a way
   * that the active FE index on ghost cells equals the active FE index
   * that was set on that processor that owned that particular ghost cell.
   * Consequently, one can <i>query</i> the @p active_fe_index on ghost
   * cells, just not set it by hand.
   *
   * On artificial cells, no information is available about the
   * @p active_fe_index used there. That's because we don't even know
   * whether these cells exist at all, and even if they did, the
   * current processor does not know anything specific about them.
   * See
   * @ref GlossArtificialCell "the glossary entry on artificial cells"
   * for more information.
   *
   * During refinement and coarsening, information about the @p active_fe_index
   * of each cell will be automatically transferred.
   *
   * However, using a parallel::distributed::Triangulation with an
   * hp__DoFHandler requires additional attention during serialization, since no
   * information on active FE indices will be automatically transferred. This
   * has to be done manually using the
   * prepare_for_serialization_of_active_fe_indices() and
   * deserialize_active_fe_indices() functions. The former has to be called
   * before parallel::distributed::Triangulation::save() is invoked, and the
   * latter needs to be run after parallel::distributed::Triangulation::load().
   * If further data will be attached to the triangulation via the
   * parallel::distributed::CellDataTransfer,
   * parallel::distributed::SolutionTransfer, or Particles::ParticleHandler
   * classes, all corresponding preparation and deserialization function calls
   * need to happen in the same order. Consult the documentation of
   * parallel::distributed::SolutionTransfer for more information.
   *
   *
   * @ingroup dofs
   * @ingroup hp
   *
   * @author Wolfgang Bangerth, 2003, 2004, 2017, 2018
   * @author Oliver Kayser-Herold, 2003, 2004
   * @author Marc Fehling, 2018
   *
   * @note Task is delegated to the base class DoFHandlerBase.
   */
  template <int dim, int spacedim = dim>
  class DoFHandler : public ::dealii::DoFHandler<dim, spacedim>
  {
  public:
    /**
     * Default Constructor.
     */
    DoFHandler();

    /**
     * Constructor. Take @p tria as the triangulation to work on.
     */
    DoFHandler(const Triangulation<dim, spacedim> &tria);

    /**
     * Copy constructor. DoFHandler objects are large and expensive.
     * They should not be copied, in particular not by accident, but
     * rather deliberately constructed. As a consequence, this constructor
     * is explicitly removed from the interface of this class.
     */
    DoFHandler(const DoFHandler &) = delete;

    /**
     * Copy operator. DoFHandler objects are large and expensive.
     * They should not be copied, in particular not by accident, but
     * rather deliberately constructed. As a consequence, this operator
     * is explicitly removed from the interface of this class.
     */
    DoFHandler &
    operator=(const DoFHandler &) = delete;
  };
} // namespace hp


DEAL_II_NAMESPACE_CLOSE

#endif
