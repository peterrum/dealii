// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2021 by the deal.II authors
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

#ifndef dealii_tensor_product_matrix_h
#define dealii_tensor_product_matrix_h


#include <deal.II/base/config.h>

#include <deal.II/base/array_view.h>
#include <deal.II/base/mutex.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/matrix_free/tensor_product_kernels.h>

DEAL_II_NAMESPACE_OPEN

// Forward declarations
#ifndef DOXYGEN
template <typename>
class Vector;
template <typename>
class FullMatrix;
#endif

/**
 * This is a special matrix class defined as the tensor product (or Kronecker
 * product) of 1D matrices of the type
 * @f{align*}{
 * L &= A_1 \otimes M_0 + M_1 \otimes A_0
 * @f}
 * in 2D and
 * @f{align*}{
 * L &= A_2 \otimes M_1 \otimes M_0 + M_2 \otimes A_1 \otimes M_0 + M_2 \otimes
 * M_1 \otimes A_0
 * @f}
 * in 3D. The typical application setting is a discretization of the Laplacian
 * $L$ on a Cartesian (axis-aligned) geometry, where it can be exactly
 * represented by the Kronecker or tensor product of a 1D mass matrix $M$ and
 * a 1D Laplace matrix $A$ in each tensor direction (due to symmetry $M$ and $A$
 * are the same in each dimension). The dimension of the resulting class is the
 * product of the one-dimensional matrices.
 *
 * This class implements two basic operations, namely the usual multiplication
 * by a vector and the inverse. For both operations, fast tensorial techniques
 * can be applied that implement the operator evaluation in
 * $\text{size}(M)^{d+1}$ arithmetic operations, considerably less than
 * $\text{size}(M)^{2d}$ for the naive forward transformation and
 * $\text{size}(M)^{3d}$ for setting up the inverse of $L$.
 *
 * Interestingly, the exact inverse of the matrix $L$ can be found through
 * tensor products due to an article by <a
 * href="http://dl.acm.org/citation.cfm?id=2716130">R. E. Lynch, J. R. Rice,
 * D. H. Thomas, Direct solution of partial difference equations by tensor
 * product methods, Numerische Mathematik 6, 185-199</a> from 1964,
 * @f{align*}{
 * L^{-1} &= S_1 \otimes S_0 (\Lambda_1 \otimes I + I \otimes \Lambda_0)^{-1}
 * S_1^\mathrm T \otimes S_0^\mathrm T,
 * @f}
 * where $S_d$ is the matrix of eigenvectors to the generalized eigenvalue
 * problem in the given tensor direction $d$:
 * @f{align*}{
 * A_d s  &= \lambda M_d s, d = 0, \quad \ldots,\mathrm{dim},
 * @f}
 * and $\Lambda_d$ is the diagonal matrix representing the generalized
 * eigenvalues $\lambda$. Note that the vectors $s$ are such that they
 * simultaneously diagonalize $A_d$ and $M_d$, i.e. $S_d^{\mathrm T} A_d S_d =
 * \Lambda_d$ and $S_d^{\mathrm T} M_d S_d = I$. This method of matrix inversion
 * is called fast diagonalization method.
 *
 * This class requires LAPACK support.
 *
 * @note This class allows for two modes of usage. The first is a use case
 * with run time constants for the matrix dimensions that is achieved by
 * setting the optional template parameter <tt>n_rows_1d</tt> to -1. The second
 * mode of usage that is faster allows to set the template parameter as a
 * compile time constant, giving significantly faster code in particular for
 * small sizes of the matrix.
 *
 * @note This class can work with scalar types (float, double) and
 * VectorizedArray types.
 *
 * @tparam dim Dimension of the problem. Currently, 1D, 2D, and 3D codes are
 * implemented.
 *
 * @tparam Number Arithmetic type of the underlying array elements. Note that the
 * underlying LAPACK implementation supports only float and double numbers, so
 * only these two types are currently supported by the generic class.
 * Nevertheless, a template specialization for the vectorized types
 * VectorizedArray<float> and VectorizedArray<double> exists. This is necessary
 * to perform LAPACK calculations for each vectorization lane, i.e. for the
 * supported float and double numbers.
 *
 * @tparam n_rows_1d Compile-time number of rows of 1D matrices (only
 * valid if the number of rows and columns coincide for each
 * dimension). By default at -1, which means that the number of rows
 * is determined at run-time by means of the matrices passed to the
 * reinit() function.
 */
template <int dim, typename Number, int n_rows_1d = -1>
class TensorProductMatrixSymmetricSum
{
public:
  /**
   * Type of matrix entries. This alias is analogous to <tt>value_type</tt>
   * in the standard library containers.
   */
  using value_type = Number;

  /**
   * The static number of rows of the 1D matrices. For more details,
   * see the description of the template parameter <tt>n_rows_1d</tt>.
   */
  static constexpr int n_rows_1d_static = n_rows_1d;

  /**
   * Default constructor.
   */
  TensorProductMatrixSymmetricSum() = default;

  /**
   * Constructor that is equivalent to the empty constructor and
   * immediately calling reinit(mass_matrix, derivative_matrix).
   */
  template <typename T>
  TensorProductMatrixSymmetricSum(const T &mass_matrix,
                                  const T &derivative_matrix);

  /**
   * Initializes the tensor product matrix by copying the arrays of 1D mass
   * matrices @p mass_matrix and 1D derivative matrices @p derivative_matrix into its
   * base class counterparts, respectively, and by assembling the regarding
   * generalized eigenvalues and eigenvectors in eigenvalues
   * and eigenvectors, respectively.
   * Note that the current implementation requires each $M_{d}$ to be symmetric
   * and positive definite and every $A_{d}$ to be symmetric and invertible but
   * not necessarily positive definite. Columns and rows filled with zero are
   * ignored.
   *
   * @warning This class accepts the following types:
   * "std::array<Table<2, Number>, dim>", "std::array<FullMatrix<Number>, dim>",
   * and "Table<2, Number>". In the latter case, we consider the same 1D
   * mass matrix @p mass_matrix and the same 1D derivative matrix
   * @p derivative_matrix for each tensor direction.
   */
  template <typename T>
  void
  reinit(const T &mass_matrix, const T &derivative_matrix);

  /**
   * Return the number of rows of the tensor product matrix
   * resulting from the Kronecker product of 1D matrices, which is described
   * in the main documentation of TensorProductMatrixSymmetricSum.
   */
  unsigned int
  m() const;

  /**
   * Return the number of columns of the tensor product matrix
   * resulting from the Kronecker product of 1D matrices, which is described
   * in the main documentation of TensorProductMatrixSymmetricSum.
   */
  unsigned int
  n() const;

  /**
   * Implements a matrix-vector product with the underlying matrix as
   * described in the main documentation of TensorProductMatrixSymmetricSum.
   * This function is operating on ArrayView to allow checks of
   * array bounds with respect to @p dst and @p src.
   *
   * @warning This function works on an internal temporal array, leading to
   * increased memory consumption if many instances of this class are created,
   * e.g., a different object on every cell with different underlying
   * coefficients each. Furthermore, only one thread runs this function at a
   * time (ensured internally with a mutex). If these two limitations are an
   * issue, please consider the other version of this function.
   */
  void
  vmult(const ArrayView<Number> &dst, const ArrayView<const Number> &src) const;

  /**
   * Same as above but letting the user provide a user-owned temporary array,
   * resolving the two issues described above. This array is resized
   * internally to the needed size.
   */
  void
  vmult(const ArrayView<Number> &      dst,
        const ArrayView<const Number> &src,
        AlignedVector<Number> &        tmp) const;

  /**
   * Implements a matrix-vector product with the underlying matrix as
   * described in the main documentation of TensorProductMatrixSymmetricSum.
   * This function is operating on ArrayView to allow checks of
   * array bounds with respect to @p dst and @p src.
   *
   * @warning This function works on an internal temporal array, leading to
   * increased memory consumption if many instances of this class are created,
   * e.g., a different object on every cell with different underlying
   * coefficients each. Furthermore, only one thread runs this function at a
   * time (ensured internally with a mutex). If these two limitations are an
   * issue, please consider the other version of this function.
   */
  void
  apply_inverse(const ArrayView<Number> &      dst,
                const ArrayView<const Number> &src) const;

  /**
   * Same as above but letting the user provide a user-owned temporary array,
   * resolving the two issues described above. This array is resized
   * internally to the needed size.
   */
  void
  apply_inverse(const ArrayView<Number> &      dst,
                const ArrayView<const Number> &src,
                AlignedVector<Number> &        tmp) const;

  /**
   * Return the memory consumption of the allocated memory in this class.
   */
  std::size_t
  memory_consumption() const;

protected:
  /**
   * An array containing a mass matrix for each tensor direction.
   */
  std::array<Table<2, Number>, dim> mass_matrix;

  /**
   * An array containing a derivative matrix for each tensor direction.
   */
  std::array<Table<2, Number>, dim> derivative_matrix;

  /**
   * An array storing the generalized eigenvalues
   * for each tensor direction.
   */
  std::array<AlignedVector<Number>, dim> eigenvalues;

  /**
   * An array storing the generalized eigenvectors
   * for each tensor direction.
   */
  std::array<Table<2, Number>, dim> eigenvectors;

private:
  /**
   * An array for temporary data.
   */
  mutable AlignedVector<Number> tmp_array;

  /**
   * A mutex that guards access to the array @p tmp_array.
   */
  mutable Threads::Mutex mutex;
};



namespace internal
{
  namespace TensorProductMatrixSymmetricSum
  {
    template <typename Number>
    struct MatrixPairComparator
    {
      using MatrixPairType = std::pair<Table<2, Number>, Table<2, Number>>;

      bool
      operator()(const MatrixPairType &left, const MatrixPairType &right) const
      {
        const auto &M_0 = left.first;
        const auto &K_0 = left.second;
        const auto &M_1 = right.first;
        const auto &K_1 = right.second;

        const unsigned int n_lanes = Number::size();

        std::array<bool, n_lanes> mask;

        using NumberScalar = typename Number::value_type;

        for (unsigned int v = 0; v < n_lanes; ++v)
          {
            NumberScalar a = 0.0;
            NumberScalar b = 0.0;

            for (unsigned int i = 0; i < M_0.size(0); ++i)
              for (unsigned int j = 0; j < M_0.size(1); ++j)
                {
                  a += std::abs(M_0[i][j][v]);
                  b += std::abs(M_1[i][j][v]);
                }

            mask[v] = (a != 0.0) && (b != 0.0);
          }

        const auto eps = std::pow<NumberScalar>(
          static_cast<NumberScalar>(10.0),
          static_cast<NumberScalar>(
            std::log10(std::numeric_limits<NumberScalar>::epsilon()) / 2.0));

        const auto less = [eps](const auto a, const auto b) -> bool {
          return (b - a) > std::max(eps, std::abs(a + b) * eps);
        };

        const auto greater = [eps](const auto a, const auto b) -> bool {
          return (a - b) > std::max(eps, std::abs(a + b) * eps);
        };

        for (unsigned int v = 0; v < n_lanes; ++v)
          if (mask[v])
            for (unsigned int i = 0; i < M_0.size(0); ++i)
              for (unsigned int j = 0; j < M_0.size(1); ++j)
                if (less(M_0[i][j][v], M_1[i][j][v]))
                  return true;
                else if (greater(M_0[i][j][v], M_1[i][j][v]))
                  return false;

        for (unsigned int v = 0; v < n_lanes; ++v)
          if (mask[v])
            for (unsigned int i = 0; i < K_0.size(0); ++i)
              for (unsigned int j = 0; j < K_0.size(1); ++j)
                if (less(K_0[i][j][v], K_1[i][j][v]))
                  return true;
                else if (greater(K_0[i][j][v], K_1[i][j][v]))
                  return false;

        return false;
      }
    };
  } // namespace TensorProductMatrixSymmetricSum
} // namespace internal



/**
 * A class similar to TensorProductMatrixSymmetricSum.
 *
 * The class TensorProductMatrixSymmetricSum stores a
 * 1D mass matrix, 1D stiffness matrix, eigenvalues and eigenvectors
 * for each direction. If one uses one TensorProductMatrixSymmetricSum
 * instance for, e.g., each cell, these quantities are stored
 * for each cell. There is no possibility to reusage quanties between
 * TensorProductMatrixSymmetricSum instances even if the values of the
 * internal data structures might be the same. This class targets the case
 * that one needs many TensorProductMatrixSymmetricSum instances and
 * there might be the possibility to reuse the internal quanties.
 *
 * This class is flexible and allows to interpret the parameter
 * @p index arbitraryly. In the case of an element-centric patch
 * smoother, the index might correspond to the cell index and,
 * in the case of a vertex patch smoother, the index might
 * correspond to the vertex index defining the vertex patch. If
 * @p n_rows_1d is set to -1, the sizes of the mass matrices and
 * the stiffness matrices can differ between cells, which might be
 * useful if the class is used in the context of hp-refinement to
 * construct a smoother.
 */
template <int dim, typename Number, int n_rows_1d = -1>
class TensorProductMatrixSymmetricSumCache
{
  using MatrixPairType = std::pair<Table<2, Number>, Table<2, Number>>;

public:
  /**
   * Allocate memory. The parameter @p specifies the maximum value
   * of the index used in invert() and apply_inverse().
   */
  void
  reserve(const unsigned int size);

  /**
   * Attach for a given @p index, the mass matrices @p Ms and
   * stiffness matrices @p Ks.
   */
  void
  insert(const unsigned int                       index,
         const std::array<Table<2, Number>, dim> &Ms,
         const std::array<Table<2, Number>, dim> &Ks);

  /**
   * Finalize setup. This function computes, e.g., the
   * eigenvalues and the eigenvectors.
   */
  void
  finalize();

  /**
   * Apply the inverse matrix for a given @p index.
   */
  void
  apply_inverse(const unsigned int             index,
                const ArrayView<Number> &      dst_in,
                const ArrayView<const Number> &src_in,
                AlignedVector<Number> &        tmp_array) const;

  /**
   * Return the memory consumption of this class in bytes.
   */
  std::size_t
  memory_consumption() const;

  /**
   * Return the number of 1D matrices of each type stored internally.
   * In the case that no compression could be performed, its value
   * is the parameter passed to the function reserve() times the
   * number of dimension. If compression could be performed, the
   * value returned is less. In the optimal case (uniform Cartesian
   * mesh), the value is one.
   */
  std::size_t
  size() const;

private:
  /**
   * Container used during setup to determine the unique 1D
   * matrices. The memory is freed during finalize().
   */
  std::map<
    MatrixPairType,
    unsigned int,
    internal::TensorProductMatrixSymmetricSum::MatrixPairComparator<Number>>
    cache;

  /**
   * Map from index to the right index within vector_mass_matrix,
   * vector_derivative_matrix, vector_eigenvectors, and
   * vector_eigenvalues. If compression was not successful, this
   * vector is emtyp, since the vectors can be access directly
   * with the given index.
   */
  std::vector<unsigned int> indices;

  /**
   * Vector of 1D mass matrices.
   */
  std::vector<Table<2, Number>> vector_mass_matrix;

  /**
   * Vector of 1D derivative matrices.
   */
  std::vector<Table<2, Number>> vector_derivative_matrix;

  /**
   * Vector of eigenvectors.
   */
  std::vector<Table<2, Number>> vector_eigenvectors;

  /**
   * Vector of eigenvalues.
   */
  std::vector<AlignedVector<Number>> vector_eigenvalues;
};



/*----------------------- Inline functions ----------------------------------*/

#ifndef DOXYGEN

namespace internal
{
  namespace TensorProductMatrixSymmetricSum
  {
    /**
     * Compute generalized eigenvalues and eigenvectors of the real
     * generalized symmetric eigenproblem $A v = \lambda M v$. Since we are
     * operating on plain pointers we require the size of the matrices
     * beforehand. Note that the data arrays for the eigenvalues and
     * eigenvectors have to be initialized to a proper size, too. (no check of
     * array bounds possible)
     */
    template <typename Number>
    void
    spectral_assembly(const Number *     mass_matrix,
                      const Number *     derivative_matrix,
                      const unsigned int n_rows,
                      const unsigned int n_cols,
                      Number *           eigenvalues,
                      Number *           eigenvectors)
    {
      Assert(n_rows == n_cols, ExcNotImplemented());

      std::vector<bool> constrained_dofs(n_rows, false);

      for (unsigned int i = 0; i < n_rows; ++i)
        {
          if (mass_matrix[i + i * n_rows] == 0.0)
            {
              Assert(derivative_matrix[i + i * n_rows] == 0.0,
                     ExcInternalError());

              for (unsigned int j = 0; j < n_rows; ++j)
                {
                  Assert(derivative_matrix[i + j * n_rows] == 0,
                         ExcInternalError());
                  Assert(derivative_matrix[j + i * n_rows] == 0,
                         ExcInternalError());
                }

              constrained_dofs[i] = true;
            }
        }

      const auto transpose_fill_nm = [&constrained_dofs](Number *           out,
                                                         const Number *     in,
                                                         const unsigned int n,
                                                         const unsigned int m) {
        for (unsigned int mm = 0, c = 0; mm < m; ++mm)
          for (unsigned int nn = 0; nn < n; ++nn, ++c)
            out[mm + nn * m] =
              (mm == nn && constrained_dofs[mm]) ? Number(1.0) : in[c];
      };

      std::vector<dealii::Vector<Number>> eigenvecs(n_rows);
      LAPACKFullMatrix<Number>            mass_copy(n_rows, n_cols);
      LAPACKFullMatrix<Number>            deriv_copy(n_rows, n_cols);

      transpose_fill_nm(&(mass_copy(0, 0)), mass_matrix, n_rows, n_cols);
      transpose_fill_nm(&(deriv_copy(0, 0)), derivative_matrix, n_rows, n_cols);

      deriv_copy.compute_generalized_eigenvalues_symmetric(mass_copy,
                                                           eigenvecs);
      AssertDimension(eigenvecs.size(), n_rows);
      for (unsigned int i = 0, c = 0; i < n_rows; ++i)
        for (unsigned int j = 0; j < n_cols; ++j, ++c)
          if (constrained_dofs[i] == false)
            eigenvectors[c] = eigenvecs[j][i];

      for (unsigned int i = 0; i < n_rows; ++i, ++eigenvalues)
        *eigenvalues = deriv_copy.eigenvalue(i).real();
    }



    template <std::size_t dim, typename Number>
    inline void
    setup(const std::array<Table<2, Number>, dim> &mass_matrices,
          const std::array<Table<2, Number>, dim> &derivative_matrices,
          std::array<Table<2, Number>, dim> &      eigenvectors,
          std::array<AlignedVector<Number>, dim> & eigenvalues)
    {
      const unsigned int n_rows_1d = mass_matrices[0].n_cols();
      (void)n_rows_1d;

      for (unsigned int dir = 0; dir < dim; ++dir)
        {
          AssertDimension(n_rows_1d, mass_matrices[dir].n_cols());
          AssertDimension(mass_matrices[dir].n_rows(),
                          mass_matrices[dir].n_cols());
          AssertDimension(mass_matrices[dir].n_rows(),
                          derivative_matrices[dir].n_rows());
          AssertDimension(mass_matrices[dir].n_rows(),
                          derivative_matrices[dir].n_cols());

          eigenvectors[dir].reinit(mass_matrices[dir].n_cols(),
                                   mass_matrices[dir].n_rows());
          eigenvalues[dir].resize(mass_matrices[dir].n_cols());
          internal::TensorProductMatrixSymmetricSum::spectral_assembly<Number>(
            &(mass_matrices[dir](0, 0)),
            &(derivative_matrices[dir](0, 0)),
            mass_matrices[dir].n_rows(),
            mass_matrices[dir].n_cols(),
            eigenvalues[dir].begin(),
            &(eigenvectors[dir](0, 0)));
        }
    }



    template <std::size_t dim, typename Number>
    inline void
    setup(const std::array<Table<2, VectorizedArray<Number>>, dim> &mass_matrix,
          const std::array<Table<2, VectorizedArray<Number>>, dim>
            &                                                 derivative_matrix,
          std::array<Table<2, VectorizedArray<Number>>, dim> &eigenvectors,
          std::array<AlignedVector<VectorizedArray<Number>>, dim> &eigenvalues)
    {
      const unsigned int     n_rows_1d   = mass_matrix[0].n_cols();
      constexpr unsigned int macro_size  = VectorizedArray<Number>::size();
      const std::size_t nm_flat_size_max = n_rows_1d * n_rows_1d * macro_size;
      const std::size_t n_flat_size_max  = n_rows_1d * macro_size;

      std::vector<Number> mass_matrix_flat;
      std::vector<Number> deriv_matrix_flat;
      std::vector<Number> eigenvalues_flat;
      std::vector<Number> eigenvectors_flat;
      mass_matrix_flat.resize(nm_flat_size_max);
      deriv_matrix_flat.resize(nm_flat_size_max);
      eigenvalues_flat.resize(n_flat_size_max);
      eigenvectors_flat.resize(nm_flat_size_max);
      std::array<unsigned int, macro_size> offsets_nm;
      std::array<unsigned int, macro_size> offsets_n;
      for (unsigned int dir = 0; dir < dim; ++dir)
        {
          AssertDimension(n_rows_1d, mass_matrix[dir].n_cols());
          AssertDimension(mass_matrix[dir].n_rows(), mass_matrix[dir].n_cols());
          AssertDimension(mass_matrix[dir].n_rows(),
                          derivative_matrix[dir].n_rows());
          AssertDimension(mass_matrix[dir].n_rows(),
                          derivative_matrix[dir].n_cols());

          const unsigned int n_rows = mass_matrix[dir].n_rows();
          const unsigned int n_cols = mass_matrix[dir].n_cols();
          const unsigned int nm     = n_rows * n_cols;
          for (unsigned int vv = 0; vv < macro_size; ++vv)
            offsets_nm[vv] = nm * vv;

          vectorized_transpose_and_store(false,
                                         nm,
                                         &(mass_matrix[dir](0, 0)),
                                         offsets_nm.cbegin(),
                                         mass_matrix_flat.data());
          vectorized_transpose_and_store(false,
                                         nm,
                                         &(derivative_matrix[dir](0, 0)),
                                         offsets_nm.cbegin(),
                                         deriv_matrix_flat.data());

          const Number *mass_cbegin    = mass_matrix_flat.data();
          const Number *deriv_cbegin   = deriv_matrix_flat.data();
          Number *      eigenvec_begin = eigenvectors_flat.data();
          Number *      eigenval_begin = eigenvalues_flat.data();
          for (unsigned int lane = 0; lane < macro_size; ++lane)
            internal::TensorProductMatrixSymmetricSum::spectral_assembly<
              Number>(mass_cbegin + nm * lane,
                      deriv_cbegin + nm * lane,
                      n_rows,
                      n_cols,
                      eigenval_begin + n_rows * lane,
                      eigenvec_begin + nm * lane);

          eigenvalues[dir].resize(n_rows);
          eigenvectors[dir].reinit(n_rows, n_cols);
          for (unsigned int vv = 0; vv < macro_size; ++vv)
            offsets_n[vv] = n_rows * vv;
          vectorized_load_and_transpose(n_rows,
                                        eigenvalues_flat.data(),
                                        offsets_n.cbegin(),
                                        eigenvalues[dir].begin());
          vectorized_load_and_transpose(nm,
                                        eigenvectors_flat.data(),
                                        offsets_nm.cbegin(),
                                        &(eigenvectors[dir](0, 0)));
        }
    }



    template <std::size_t dim, typename Number>
    inline std::array<Table<2, Number>, dim>
    convert(const std::array<Table<2, Number>, dim> &mass_matrix)
    {
      return mass_matrix;
    }



    template <std::size_t dim, typename Number>
    inline std::array<Table<2, Number>, dim>
    convert(const std::array<FullMatrix<Number>, dim> &mass_matrix)
    {
      std::array<Table<2, Number>, dim> mass_copy;

      std::transform(mass_matrix.cbegin(),
                     mass_matrix.cend(),
                     mass_copy.begin(),
                     [](const FullMatrix<Number> &m) -> Table<2, Number> {
                       return m;
                     });

      return mass_copy;
    }



    template <std::size_t dim, typename Number>
    inline std::array<Table<2, Number>, dim>
    convert(const Table<2, Number> &mass_matrix)
    {
      std::array<Table<2, Number>, dim> mass_matrices;

      std::fill(mass_matrices.begin(), mass_matrices.end(), mass_matrix);

      return mass_matrices;
    }



    template <int n_rows_1d_templated, std::size_t dim, typename Number>
    void
    vmult(Number *                               dst,
          const Number *                         src,
          AlignedVector<Number> &                tmp,
          const unsigned int                     n_rows_1d_non_templated,
          const std::array<const Number *, dim> &mass_matrix,
          const std::array<const Number *, dim> &derivative_matrix)
    {
      const unsigned int n_rows_1d = n_rows_1d_templated == 0 ?
                                       n_rows_1d_non_templated :
                                       n_rows_1d_templated;
      const unsigned int n         = Utilities::fixed_power<dim>(n_rows_1d);

      tmp.resize_fast(n * 2);
      Number *t = tmp.begin();

      internal::EvaluatorTensorProduct<internal::evaluate_general,
                                       dim,
                                       n_rows_1d_templated,
                                       n_rows_1d_templated,
                                       Number>
        eval({}, {}, {}, n_rows_1d, n_rows_1d);

      if (dim == 1)
        {
          const Number *A = derivative_matrix[0];
          eval.template apply<0, false, false>(A, src, dst);
        }

      else if (dim == 2)
        {
          const Number *A0 = derivative_matrix[0];
          const Number *M0 = mass_matrix[0];
          const Number *A1 = derivative_matrix[1];
          const Number *M1 = mass_matrix[1];
          eval.template apply<0, false, false>(M0, src, t);
          eval.template apply<1, false, false>(A1, t, dst);
          eval.template apply<0, false, false>(A0, src, t);
          eval.template apply<1, false, true>(M1, t, dst);
        }

      else if (dim == 3)
        {
          const Number *A0 = derivative_matrix[0];
          const Number *M0 = mass_matrix[0];
          const Number *A1 = derivative_matrix[1];
          const Number *M1 = mass_matrix[1];
          const Number *A2 = derivative_matrix[2];
          const Number *M2 = mass_matrix[2];
          eval.template apply<0, false, false>(M0, src, t + n);
          eval.template apply<1, false, false>(M1, t + n, t);
          eval.template apply<2, false, false>(A2, t, dst);
          eval.template apply<1, false, false>(A1, t + n, t);
          eval.template apply<0, false, false>(A0, src, t + n);
          eval.template apply<1, false, true>(M1, t + n, t);
          eval.template apply<2, false, true>(M2, t, dst);
        }

      else
        AssertThrow(false, ExcNotImplemented());
    }



    template <int n_rows_1d_templated, std::size_t dim, typename Number>
    void
    apply_inverse(Number *               dst,
                  const Number *         src,
                  AlignedVector<Number> &tmp,
                  const unsigned int     n_rows_1d_non_templated,
                  const std::array<const Number *, dim> &eigenvectors,
                  const std::array<const Number *, dim> &eigenvalues)
    {
      const unsigned int n_rows_1d = n_rows_1d_templated == 0 ?
                                       n_rows_1d_non_templated :
                                       n_rows_1d_templated;
      const unsigned int n         = Utilities::fixed_power<dim>(n_rows_1d);

      tmp.resize_fast(n);
      Number *t = tmp.begin();

      internal::EvaluatorTensorProduct<internal::evaluate_general,
                                       dim,
                                       n_rows_1d_templated,
                                       n_rows_1d_templated,
                                       Number>
        eval({}, {}, {}, n_rows_1d, n_rows_1d);

      // NOTE: dof_to_quad has to be interpreted as 'dof to eigenvalue index'
      //       --> apply<.,true,.> (S,src,dst) calculates dst = S^T * src,
      //       --> apply<.,false,.> (S,src,dst) calculates dst = S * src,
      //       while the eigenvectors are stored column-wise in S, i.e.
      //       rows correspond to dofs whereas columns to eigenvalue indices!
      if (dim == 1)
        {
          const Number *S = eigenvectors[0];
          eval.template apply<0, true, false>(S, src, t);
          for (unsigned int i = 0; i < n_rows_1d; ++i)
            t[i] /= eigenvalues[0][i];
          eval.template apply<0, false, false>(S, t, dst);
        }

      else if (dim == 2)
        {
          const Number *S0 = eigenvectors[0];
          const Number *S1 = eigenvectors[1];
          eval.template apply<0, true, false>(S0, src, t);
          eval.template apply<1, true, false>(S1, t, dst);
          for (unsigned int i1 = 0, c = 0; i1 < n_rows_1d; ++i1)
            for (unsigned int i0 = 0; i0 < n_rows_1d; ++i0, ++c)
              dst[c] /= (eigenvalues[1][i1] + eigenvalues[0][i0]);
          eval.template apply<0, false, false>(S0, dst, t);
          eval.template apply<1, false, false>(S1, t, dst);
        }

      else if (dim == 3)
        {
          const Number *S0 = eigenvectors[0];
          const Number *S1 = eigenvectors[1];
          const Number *S2 = eigenvectors[2];
          eval.template apply<0, true, false>(S0, src, t);
          eval.template apply<1, true, false>(S1, t, dst);
          eval.template apply<2, true, false>(S2, dst, t);
          for (unsigned int i2 = 0, c = 0; i2 < n_rows_1d; ++i2)
            for (unsigned int i1 = 0; i1 < n_rows_1d; ++i1)
              for (unsigned int i0 = 0; i0 < n_rows_1d; ++i0, ++c)
                t[c] /= (eigenvalues[2][i2] + eigenvalues[1][i1] +
                         eigenvalues[0][i0]);
          eval.template apply<0, false, false>(S0, t, dst);
          eval.template apply<1, false, false>(S1, dst, t);
          eval.template apply<2, false, false>(S2, t, dst);
        }

      else
        Assert(false, ExcNotImplemented());
    }



    template <int n_rows_1d_templated, std::size_t dim, typename Number>
    void
    select_vmult(Number *                               dst,
                 const Number *                         src,
                 AlignedVector<Number> &                tmp,
                 const unsigned int                     n_rows_1d,
                 const std::array<const Number *, dim> &mass_matrix,
                 const std::array<const Number *, dim> &derivative_matrix);



    template <int n_rows_1d_templated, std::size_t dim, typename Number>
    void
    select_apply_inverse(Number *                               dst,
                         const Number *                         src,
                         AlignedVector<Number> &                tmp,
                         const unsigned int                     n_rows_1d,
                         const std::array<const Number *, dim> &eigenvectors,
                         const std::array<const Number *, dim> &eigenvalues);
  } // namespace TensorProductMatrixSymmetricSum
} // namespace internal


template <int dim, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>::m() const
{
  unsigned int m = mass_matrix[0].n_rows();
  for (unsigned int d = 1; d < dim; ++d)
    m *= mass_matrix[d].n_rows();
  return m;
}



template <int dim, typename Number, int n_rows_1d>
inline unsigned int
TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>::n() const
{
  unsigned int n = mass_matrix[0].n_cols();
  for (unsigned int d = 1; d < dim; ++d)
    n *= mass_matrix[d].n_cols();
  return n;
}



template <int dim, typename Number, int n_rows_1d>
inline void
TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>::vmult(
  const ArrayView<Number> &      dst_view,
  const ArrayView<const Number> &src_view) const
{
  std::lock_guard<std::mutex> lock(this->mutex);
  this->vmult(dst_view, src_view, this->tmp_array);
}



template <int dim, typename Number, int n_rows_1d>
inline void
TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>::vmult(
  const ArrayView<Number> &      dst_view,
  const ArrayView<const Number> &src_view,
  AlignedVector<Number> &        tmp_array) const
{
  AssertDimension(dst_view.size(), this->m());
  AssertDimension(src_view.size(), this->n());

  Number *      dst = dst_view.begin();
  const Number *src = src_view.begin();

  std::array<const Number *, dim> mass_matrix, derivative_matrix;

  for (unsigned int d = 0; d < dim; ++d)
    {
      mass_matrix[d]       = &this->mass_matrix[d](0, 0);
      derivative_matrix[d] = &this->derivative_matrix[d](0, 0);
    }

  const unsigned int n_rows_1d_non_templated = this->mass_matrix[0].n_rows();

  if (n_rows_1d != -1)
    internal::TensorProductMatrixSymmetricSum::vmult<
      n_rows_1d == -1 ? 0 : n_rows_1d>(dst,
                                       src,
                                       tmp_array,
                                       n_rows_1d_non_templated,
                                       mass_matrix,
                                       derivative_matrix);
  else
    internal::TensorProductMatrixSymmetricSum::select_vmult<1>(
      dst,
      src,
      tmp_array,
      n_rows_1d_non_templated,
      mass_matrix,
      derivative_matrix);
}



template <int dim, typename Number, int n_rows_1d>
inline void
TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>::apply_inverse(
  const ArrayView<Number> &      dst_view,
  const ArrayView<const Number> &src_view) const
{
  std::lock_guard<std::mutex> lock(this->mutex);
  this->apply_inverse(dst_view, src_view, this->tmp_array);
}



template <int dim, typename Number, int n_rows_1d>
inline void
TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>::apply_inverse(
  const ArrayView<Number> &      dst_view,
  const ArrayView<const Number> &src_view,
  AlignedVector<Number> &        tmp_array) const
{
  AssertDimension(dst_view.size(), this->n());
  AssertDimension(src_view.size(), this->m());

  Number *      dst = dst_view.begin();
  const Number *src = src_view.begin();

  std::array<const Number *, dim> eigenvectors, eigenvalues;

  for (unsigned int d = 0; d < dim; ++d)
    {
      eigenvectors[d] = &this->eigenvectors[d](0, 0);
      eigenvalues[d]  = this->eigenvalues[d].data();
    }

  const unsigned int n_rows_1d_non_templated = this->mass_matrix[0].n_rows();

  if (n_rows_1d != -1)
    internal::TensorProductMatrixSymmetricSum::apply_inverse<
      n_rows_1d == -1 ? 0 : n_rows_1d>(
      dst, src, tmp_array, n_rows_1d_non_templated, eigenvectors, eigenvalues);
  else
    internal::TensorProductMatrixSymmetricSum::select_apply_inverse<1>(
      dst, src, tmp_array, n_rows_1d_non_templated, eigenvectors, eigenvalues);
}



template <int dim, typename Number, int n_rows_1d>
std::size_t
TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>::memory_consumption()
  const
{
  return MemoryConsumption::memory_consumption(mass_matrix) +
         MemoryConsumption::memory_consumption(derivative_matrix) +
         MemoryConsumption::memory_consumption(eigenvalues) +
         MemoryConsumption::memory_consumption(eigenvectors) +
         MemoryConsumption::memory_consumption(tmp_array);
}



template <int dim, typename Number, int n_rows_1d>
template <typename T>
inline TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>::
  TensorProductMatrixSymmetricSum(const T &mass_matrix,
                                  const T &derivative_matrix)
{
  reinit(mass_matrix, derivative_matrix);
}



template <int dim, typename Number, int n_rows_1d>
template <typename T>
inline void
TensorProductMatrixSymmetricSum<dim, Number, n_rows_1d>::reinit(
  const T &mass_matrix,
  const T &derivative_matrix)
{
  this->mass_matrix =
    internal::TensorProductMatrixSymmetricSum::convert<dim>(mass_matrix);
  this->derivative_matrix =
    internal::TensorProductMatrixSymmetricSum::convert<dim>(derivative_matrix);

  internal::TensorProductMatrixSymmetricSum::setup(this->mass_matrix,
                                                   this->derivative_matrix,
                                                   this->eigenvectors,
                                                   this->eigenvalues);
}



template <int dim, typename Number, int n_rows_1d>
void
TensorProductMatrixSymmetricSumCache<dim, Number, n_rows_1d>::reserve(
  const unsigned int size)
{
  indices.assign(size * dim, numbers::invalid_unsigned_int);
}



template <int dim, typename Number, int n_rows_1d>
void
TensorProductMatrixSymmetricSumCache<dim, Number, n_rows_1d>::insert(
  const unsigned int                       index,
  const std::array<Table<2, Number>, dim> &Ms,
  const std::array<Table<2, Number>, dim> &Ks)
{
  for (unsigned int d = 0; d < dim; ++d)
    {
      const MatrixPairType matrix(Ms[d], Ks[d]);

      const auto ptr = cache.find(matrix);

      if (ptr != cache.end())
        indices[index * dim + d] = ptr->second;
      else
        {
          indices[index * dim + d] = cache.size();
          cache[matrix]            = cache.size();
        }
    }
}



template <int dim, typename Number, int n_rows_1d>
void
TensorProductMatrixSymmetricSumCache<dim, Number, n_rows_1d>::finalize()
{
  vector_mass_matrix.resize(cache.size());
  vector_derivative_matrix.resize(cache.size());
  vector_eigenvectors.resize(cache.size());
  vector_eigenvalues.resize(cache.size());

  const auto store = [&](const unsigned int    index,
                         const MatrixPairType &M_and_K) {
    std::array<Table<2, Number>, 1> mass_matrices;
    mass_matrices[0] = M_and_K.first;

    std::array<Table<2, Number>, 1> derivative_matrices;
    derivative_matrices[0] = M_and_K.second;

    std::array<Table<2, Number>, 1>      eigenvectors;
    std::array<AlignedVector<Number>, 1> eigenvalues;

    internal::TensorProductMatrixSymmetricSum::setup(mass_matrices,
                                                     derivative_matrices,
                                                     eigenvectors,
                                                     eigenvalues);

    vector_mass_matrix[index]       = M_and_K.first;
    vector_derivative_matrix[index] = M_and_K.second;
    vector_eigenvectors[index]      = eigenvectors[0];
    vector_eigenvalues[index]       = eigenvalues[0];
  };

  if (cache.size() == indices.size())
    {
      std::map<unsigned int, MatrixPairType> inverted_cache;

      for (const auto &i : cache)
        inverted_cache[i.second] = i.first;

      for (unsigned int i = 0; i < indices.size(); ++i)
        store(i, inverted_cache[indices[i]]);

      indices.clear();
    }
  else
    {
      for (const auto &i : cache)
        store(i.second, i.first);
    }

  cache.clear();
}



template <int dim, typename Number, int n_rows_1d>
void
TensorProductMatrixSymmetricSumCache<dim, Number, n_rows_1d>::apply_inverse(
  const unsigned int             index,
  const ArrayView<Number> &      dst_in,
  const ArrayView<const Number> &src_in,
  AlignedVector<Number> &        tmp_array) const
{
  Number *      dst = dst_in.begin();
  const Number *src = src_in.begin();

  std::array<const Number *, dim> eigenvectors, eigenvalues;
  unsigned int                    n_rows_1d_non_templated = 0;

  for (unsigned int d = 0; d < dim; ++d)
    {
      const unsigned int translated_index =
        (indices.size() > 0) ? indices[dim * index + d] : (dim * index + d);

      eigenvectors[d]         = &vector_eigenvectors[translated_index](0, 0);
      eigenvalues[d]          = vector_eigenvalues[translated_index].data();
      n_rows_1d_non_templated = vector_eigenvalues[translated_index].size();
    }

  if (n_rows_1d != -1)
    internal::TensorProductMatrixSymmetricSum::apply_inverse<
      n_rows_1d == -1 ? 0 : n_rows_1d>(
      dst, src, tmp_array, n_rows_1d_non_templated, eigenvectors, eigenvalues);
  else
    internal::TensorProductMatrixSymmetricSum::select_apply_inverse<1>(
      dst, src, tmp_array, n_rows_1d_non_templated, eigenvectors, eigenvalues);
}



template <int dim, typename Number, int n_rows_1d>
std::size_t
TensorProductMatrixSymmetricSumCache<dim, Number, n_rows_1d>::
  memory_consumption() const
{
  return MemoryConsumption::memory_consumption(indices) +
         MemoryConsumption::memory_consumption(vector_mass_matrix) +
         MemoryConsumption::memory_consumption(vector_derivative_matrix) +
         MemoryConsumption::memory_consumption(vector_eigenvectors) +
         MemoryConsumption::memory_consumption(vector_eigenvalues);
}



template <int dim, typename Number, int n_rows_1d>
std::size_t
TensorProductMatrixSymmetricSumCache<dim, Number, n_rows_1d>::size() const
{
  return vector_mass_matrix.size();
}



#endif

DEAL_II_NAMESPACE_CLOSE

#endif
