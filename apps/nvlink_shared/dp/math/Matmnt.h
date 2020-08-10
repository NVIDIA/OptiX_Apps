// Copyright (c) 2009-2015, NVIDIA CORPORATION. All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
/** @file */

#include <dp/math/Config.h>
#include <dp/math/Quatt.h>
#include <dp/math/Vecnt.h>
#include <array>

namespace dp
{
  namespace math
  {

    template<typename T> class Quatt;

    /*! \brief Matrix class of fixed size and type.
     *  \remarks This class is templated by size and type. It holds \a m rows times \a n columns values of type \a
     *  T. There are typedefs for the most common usage with 3x3 and 4x4 values of type \c float and \c
     *
     *  The layout in memory is is row-major.
     *  Vectors have to be multiplied from the left (result = v*M).
     *  The last row [12-14] contains the translation.
     *
     *  double: Mat33f, Mat33d, Mat44f, Mat44d. */
    template<unsigned int m, unsigned int n, typename T> class Matmnt
    {
      public:
        /*! \brief Default constructor.
         *  \remarks For performance reasons, no initialization is performed. */
        Matmnt();

        /*! \brief Copy constructor from a matrix of possibly different size and type.
         *  \param rhs A matrix with \a m times \a m values of type \a S.
         *  \remarks The minimum \a x of \a m and \a k, and the minimum \a y of \a n and \a l is determined. The
         *  first \a y values of type \a S in the first \a x rows from \a rhs are converted to type \a T and
         *  assigned as the first \a y values in the first \a x rows of \c this. If \a x is less than \a m, the
         *  last rows of \a this are not initialized. If \a y is less than \a n, the last values of the first \a
         *  x rows are not initialized. */
        template<unsigned int k, unsigned int l, typename S>
          explicit Matmnt( const Matmnt<k,l,S> & rhs );

        /*! \brief Constructor for a matrix by an array of m rows of type Vecnt<n,T>.
         *  \param rows An array of m rows of type Vecnt<n,T>
         *  \remarks This constructor can easily be called, using an initializer-list
         *  \code
         *    Mat33f m33f( { xAxis, yAxis, zAxis } );
         *  \endcode */
        explicit Matmnt( const std::array<Vecnt<n,T>,m> & rows );

        /*! \brief Constructor for a matrix by an array of 2 rows of type Vecnt<n,T>.
         *  \param v1,v2 The vectors for the rows
         */
        explicit Matmnt( Vecnt<n,T> const& v1, Vecnt<n,T> const& v2 );

        /*! \brief Constructor for a matrix by an array of 3 rows of type Vecnt<n,T>.
         *  \param v1,v2,v3 The vectors for the rows
         */
        explicit Matmnt( Vecnt<n,T> const& v1, Vecnt<n,T> const& v2, Vecnt<n,T> const& v3 );

        /*! \brief Constructor for a matrix by an array of 4 rows of type Vecnt<n,T>.
         *  \param v1,v2,v3,v4 The vectors for the rows
         */
        explicit Matmnt( Vecnt<n,T> const& v1, Vecnt<n,T> const& v2, Vecnt<n,T> const& v3, Vecnt<n,T> const& v4 );


        /*! \brief Constructor for a matrix by an array of m*n scalars of type T.
         *  \param scalars An array of m*n scalars of type T
         *  \remarks This constructor can easily be called, using an initializer-list
         *  \code
         *    Mat33f m33f( { m00, m01, m02, m10, m11, m12, m20, m21, m22 } );
         *  \endcode */
        explicit Matmnt( const std::array<T,m*n> & scalars );

        /*! \brief Constructor for a 3 by 3 rotation matrix out of an axis and an angle.
         *  \param axis A reference to the constant axis to rotate about.
         *  \param angle The angle, in radians, to rotate.
         *  \remarks The resulting 3 by 3 matrix is a pure rotation.
         *  \note The behavior is undefined, if \a axis is not normalized.
         *  \par Example:
         *  \code
         *    Mat33f rotZAxisBy45Degrees( Vec3f( 0.0f, 0.0f, 1.0f ), PI/4 );
         *  \endcode */
        Matmnt( const Vecnt<3,T> & axis, T angle );

        /*! \brief Constructor for a 3 by 3 rotation matrix out of a normalized quaternion.
         *  \param ori A reference to the normalized quaternion representing the rotation.
         *  \remarks The resulting 3 by 3 matrix is a pure rotation.
         *  \note The behavior is undefined, if \a ori is not normalized. */
        explicit Matmnt( const Quatt<T> & ori );

        /*! \brief Constructor for a 4 by 4 transformation matrix out of a quaternion and a translation.
         *  \param ori A reference to the normalized quaternion representing the rotational part.
         *  \param trans A reference to the vector representing the translational part.
         *  \note The behavior is undefined, if \ ori is not normalized. */
        Matmnt( const Quatt<T> & ori, const Vecnt<3,T> & trans );

        /*! \brief Constructor for a 4 by 4 transformation matrix out of a quaternion, a translation,
         *  and a scaling.
         *  \param ori A reference to the normalized quaternion representing the rotational part.
         *  \param trans A reference to the vector representing the translational part.
         *  \param scale A reference to the vector representing the scaling along the three axis directions.
         *  \note The behavior is undefined, if \ ori is not normalized. */
        Matmnt( const Quatt<T> & ori, const Vecnt<3,T> & trans, const Vecnt<3,T> & scale );

      public:
        /*! \brief Get a constant pointer to the n times n values of the matrix.
         *  \return A constant pointer to the matrix elements.
         *  \remarks The matrix elements are stored in row-major order. This function returns the
         *  address of the first element of the first row. It is assured, that the other elements of
         *  the matrix follow linearly.
         *  \par Example:
         *  If \c m is a 3 by 3 matrix, m.getPtr() gives a pointer to the 9 elements m00, m01, m02, m10,
         *  m11, m12, m20, m21, m22, in that order. */
        const T * getPtr() const;

        /*! \brief Invert the matrix.
         *  \return \c true, if the matrix was successfully inverted, otherwise \c false. */
        bool invert();

        /*! \brief Non-constant subscript operator.
         *  \param i Index of the row to address.
         *  \return A reference to the \a i th row of the matrix. */
          Vecnt<n,T> & operator[]( unsigned int i );

        /*! \brief Constant subscript operator.
         *  \param i Index of the row to address.
         *  \return A constant reference to the \a i th row of the matrix. */
          const Vecnt<n,T> & operator[]( unsigned int i ) const;

        /*! \brief Matrix addition and assignment operator.
         *  \param mat A constant reference to the matrix to add.
         *  \return A reference to \c this.
         *  \remarks The matrix \a mat has to be of the same size as \c this, but may hold values of a
         *  different type. The matrix elements of type \a S of \a mat are converted to type \a T and
         *  added to the corresponding matrix elements of \c this. */
        template<typename S>
          Matmnt<m,n,T> & operator+=( const Matmnt<m,n,S> & mat );

        /*! \brief Matrix subtraction and assignment operator.
         *  \param mat A constant reference to the matrix to subtract.
         *  \return A reference to \c this.
         *  \remarks The matrix \a mat has to be of the same size as \c this, but may hold values of a
         *  different type. The matrix elements of type \a S of \a mat are converted to type \a T and
         *  subtracted from the corresponding matrix elements of \c this. */
        template<typename S>
          Matmnt<m,n,T> & operator-=( const Matmnt<m,n,S> & mat );

        /*! \brief Scalar multiplication and assignment operator.
         *  \param s A scalar value to multiply with.
         *  \return A reference to \c this.
         *  \remarks The type of \a s may be of different type as the elements of the \c this. \a s is
         *  converted to type \a T and each element of \c this is multiplied with it. */
        template<typename S>
          Matmnt<m,n,T> & operator*=( S s );

        /*! \brief Matrix multiplication and assignment operator.
         *  \param mat A constant reference to the matrix to multiply with.
         *  \return A reference to \c this.
         *  \remarks The matrix multiplication \code *this * mat \endcode is calculated and assigned to
         *  \c this. */
        Matmnt<m,n,T> & operator*=( const Matmnt<m,n,T> & mat );

        /*! \brief Scalar division and assignment operator.
         *  \param s A scalar value to divide by.
         *  \return A reference to \c this.
         *  \remarks The type of \a s may be of different type as the elements of the \c this. \a s is
         *  converted to type \a T and each element of \c this is divided by it.
         *  \note The behavior is undefined if \a s is very close to zero. */
        template<typename S>
          Matmnt<m,n,T> & operator/=( S s );

      private:
        Vecnt<n,T>  m_mat[m];
    };


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // non-member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    /*! \brief Determine the determinant of a matrix.
     *  \param mat A constant reference to the matrix to determine the determinant from.
     *  \return The determinant of \a mat. */
    template<unsigned int n, typename T>
      T determinant( const Matmnt<n,n,T> & mat );

    /*! \brief Invert a matrix.
     *  \param mIn A constant reference to the matrix to invert.
     *  \param mOut A reference to the matrix to hold the inverse.
     *  \return \c true, if the matrix \a mIn was successfully inverted, otherwise \c false.
     *  \note If the mIn was not successfully inverted, the values in mOut are undefined. */
    template<unsigned int n, typename T>
      bool invert( const Matmnt<n,n,T> & mIn, Matmnt<n,n,T> & mOut );

    /*! \brief Test if a matrix is the identity.
     *  \param mat A constant reference to the matrix to test for identity.
     *  \param eps An optional value giving the allowed epsilon. The default is a type dependent value.
     *  \return \c true, if the matrix is the identity, otherwise \c false.
     *  \remarks A matrix is considered to be the identity, if each of the diagonal elements differ
     *  less than \a eps from one, and each of the other matrix elements differ less than \a eps from
     *  zero.
     *  \sa isNormalized, isNull, isOrthogonal, isSingular */
    template<unsigned int n, typename T>
      bool isIdentity( const Matmnt<n,n,T> & mat, T eps = std::numeric_limits<T>::epsilon() )
    {
      bool identity = true;
      for ( unsigned int i=0 ; identity && i<n ; ++i )
      {
        for ( unsigned int j=0 ; identity && j<i ; ++j )
        {
          identity = ( std::abs( mat[i][j] ) <= eps );
        }
        if ( identity )
        {
          identity = ( std::abs( mat[i][i] - 1 ) <= eps );
        }
        for ( unsigned int j=i+1 ; identity && j<n ; ++j )
        {
          identity = ( std::abs( mat[i][j] ) <= eps );
        }
      }
      return( identity );
    }

    /*! \brief Test if a matrix is normalized.
     *  \param mat A constant reference to the matrix to test.
     *  \param eps An optional value giving the allowed epsilon. The default is a type dependent value.
     *  \return \c true if the matrix is normalized, otherwise \c false.
     *  \remarks A matrix is considered to be normalized, if each row and each column is normalized.
     *  \sa isIdentity, isNull, isOrthogonal, isSingular */
    template<unsigned int n, typename T>
      bool isNormalized( const Matmnt<n,n,T> & mat, T eps = std::numeric_limits<T>::epsilon() )
    {
      bool normalized = true;
      for ( unsigned int i=0 ; normalized && i<n ; ++i )
      {
        normalized = isNormalized( mat[i], eps );
      }
      for ( unsigned int i=0 ; normalized && i<n ; ++i )
      {
        Vecnt<n,T> v;
        for ( unsigned int j=0 ; j<n ; j++ )
        {
          v[j] = mat[j][i];
        }
        normalized = isNormalized( v, eps );
      }
      return( normalized );
    }

    /*! \brief Test if a matrix is null.
     *  \param mat A constant reference to the matrix to test.
     *  \param eps An optional value giving the allowed epsilon. The default is a type dependent value.
     *  \return \c true if the matrix is null, otherwise \c false.
     *  \remarks A matrix is considered to be null, if each row is null.
     *  \sa isIdentity, isNormalized, isOrthogonal, isSingular */
    template<unsigned int m, unsigned int n, typename T>
      bool isNull( const Matmnt<m,n,T> & mat, T eps = std::numeric_limits<T>::epsilon() )
    {
      bool null = true;
      for ( unsigned int i=0 ; null && i<m ; ++i )
      {
        null = isNull( mat[i], eps );
      }
      return( null );
    }

    /*! \brief Test if a matrix is orthogonal.
     *  \param mat A constant reference to the matrix to test.
     *  \param eps An optional value giving the allowed epsilon. The default is a type dependent value.
     *  \return \c true, if the matrix is orthogonal, otherwise \c false.
     *  \remarks A matrix is considered to be orthogonal, if each pair of rows and each pair of
     *  columns are orthogonal to each other.
     *  \sa isIdentity, isNormalized, isNull, isSingular */
    template<unsigned int n, typename T>
      bool isOrthogonal( const Matmnt<n,n,T> & mat, T eps = std::numeric_limits<T>::epsilon() )
    {
      bool orthogonal = true;
      for ( unsigned int i=0 ; orthogonal && i+1<n ; ++i )
      {
        for ( unsigned int j=i+1 ; orthogonal && j<n ; ++j )
        {
          orthogonal = areOrthogonal( mat[i], mat[j], eps );
        }
      }
      if ( orthogonal )
      {
        Matmnt<n,n,T> tm = ~mat;
        for ( unsigned int i=0 ; orthogonal && i+1<n ; ++i )
        {
          for ( unsigned int j=i+1 ; orthogonal && j<n ; ++j )
          {
            orthogonal = areOrthogonal( tm[i], tm[j], eps );
          }
        }
      }
      return( orthogonal );
    }

    /*! \brief Test if a matrix is singular.
     *  \param m A constant reference to the matrix to test.
     *  \param eps An optional value giving the allowed epsilon. The default is a type dependent value.
     *  \return \c true, if the matrix is singular, otherwise \c false.
     *  \remarks A matrix is considered to be singular, if its determinant is zero.
     *  \sa isIdentity, isNormalized, isNull, isOrthogonal */
    template<unsigned int n, typename T>
      bool isSingular( const Matmnt<n,n,T> & mat, T eps = std::numeric_limits<T>::epsilon() )
    {
      return( abs( determinant( mat ) ) <= eps );
    }

    /*! \brief Get the value of the maximal absolute element of a matrix.
     *  \param mat A constant reference to a matrix to get the maximal element from.
     *  \return The value of the maximal absolute element of \a mat.
     *  \sa minElement */
    template<unsigned int m, unsigned int n, typename T>
      T maxElement( const Matmnt<m,n,T> & mat );

    /*! \brief Get the value of the minimal absolute element of a matrix.
     *  \param mat A constant reference to a matrix to get the minimal element from.
     *  \return The value of the minimal absolute element of \a mat.
     *  \sa maxElement */
    template<unsigned int m, unsigned int n, typename T>
      T minElement( const Matmnt<m,n,T> & mat );

    /*! \brief Matrix equality operator.
     *  \param m0 A constant reference to the first matrix to compare.
     *  \param m1 A constant reference to the second matrix to compare.
     *  \return \c true, if \a m0 and \a m1 are equal, otherwise \c false.
     *  \remarks Two matrices are considered to be equal, if each element of \a m0 differs less than
     *  the type dependent epsilon from the the corresponding element of \a m1. */
    template<unsigned int m, unsigned int n, typename T>
      bool operator==( const Matmnt<m,n,T> & m0, const Matmnt<m,n,T> & m1 );

    /*! \brief Matrix inequality operator.
     *  \param m0 A constant reference to the first matrix to compare.
     *  \param m1 A constant reference to the second matrix to compare.
     *  \return \c true, if \a m0 and \a m1 are not equal, otherwise \c false.
     *  \remarks Two matrices are considered to be not equal, if at least one element of \a m0 differs
     *  more than the type dependent epsilon from the the corresponding element of \a m1. */
    template<unsigned int m, unsigned int n, typename T>
      bool operator!=( const Matmnt<m,n,T> & m0, const Matmnt<m,n,T> & m1 );

    /*! \brief Matrix transpose operator.
     *  \param mat A constant reference to the matrix to transpose.
     *  \return The transposed version of \a m. */
    template<unsigned int m, unsigned int n, typename T>
      Matmnt<n,m,T> operator~( const Matmnt<m,n,T> & mat );

    /*! \brief Matrix addition operator.
     *  \param m0 A constant reference to the first matrix to add.
     *  \param m1 A constant reference to the second matrix to add.
     *  \return A matrix representing the sum of \code m0 + m1 \endcode */
    template<unsigned int m, unsigned int n, typename T>
      Matmnt<m,n,T> operator+( const Matmnt<m,n,T> & m0, const Matmnt<m,n,T> & m1 );

    /*! \brief Matrix negation operator.
     *  \param mat A constant reference to the matrix to negate.
     *  \return A matrix representing the negation of \a mat. */
    template<unsigned int m, unsigned int n, typename T>
      Matmnt<m,n,T> operator-( const Matmnt<m,n,T> & mat );

    /*! \brief Matrix subtraction operator.
     *  \param m0 A constant reference to the first argument of the subtraction.
     *  \param m1 A constant reference to the second argument of the subtraction.
     *  \return A matrix representing the difference \code m0 - m1 \endcode */
    template<unsigned int m, unsigned int n, typename T>
      Matmnt<m,n,T> operator-( const Matmnt<m,n,T> & m0, const Matmnt<m,n,T> & m1 );

    /*! \brief Scalar multiplication operator.
     *  \param mat A constant reference to the matrix to multiply.
     *  \param s The scalar value to multiply with.
     *  \return A matrix representing the product \code mat * s \endcode */
    template<unsigned int m, unsigned int n, typename T>
      Matmnt<m,n,T> operator*( const Matmnt<m,n,T> & mat, T s );

    /*! \brief Scalar multiplication operator.
     *  \param s The scalar value to multiply with.
     *  \param mat A constant reference to the matrix to multiply.
     *  \return A matrix representing the product \code s * mat \endcode */
    template<unsigned int m, unsigned int n, typename T>
      Matmnt<m,n,T> operator*( T s, const Matmnt<m,n,T> & mat );

    /*! \brief Vector multiplication operator.
     *  \param mat A constant reference to the matrix to multiply.
     *  \param v A constant reference to the vector to multiply with.
     *  \return A vector representing the product \code mat * v \endcode */
    template<unsigned int m, unsigned int n, typename T>
      Vecnt<m,T> operator*( const Matmnt<m,n,T> & mat, const Vecnt<n,T> & v );

    /*! \brief Vector multiplication operator.
     *  \param v A constant reference to the vector to multiply with.
     *  \param mat A constant reference to the matrix to multiply.
     *  \return A vector representing the product \code v * mat \endcode */
    template<unsigned int m, unsigned int n, typename T>
      Vecnt<n,T> operator*( const Vecnt<m,T> & v, const Matmnt<m,n,T> & mat );

    /*! \brief Matrix multiplication operator.
     *  \param m0 A constant reference to the first operand of the multiplication.
     *  \param m1 A constant reference to the second operand of the multiplication.
     *  \return A matrix representing the product \code m0 * m1 \endcode */
    template<unsigned int m, unsigned int n, unsigned int k, typename T>
      Matmnt<m,k,T> operator*( const Matmnt<m,n,T> & m0, const Matmnt<n,k,T> & m1 );

    /*! \brief Scalar division operator.
     *  \param mat A constant reference to the matrix to divide.
     *  \param s The scalar value to divide by.
     *  \return A matrix representing the matrix \a mat divided by \a s. */
    template<unsigned int m, unsigned int n, typename T>
      Matmnt<m,n,T> operator/( const Matmnt<m,n,T> & mat, T s );

    /*! \brief Set a matrix to be the identity.
     *  \param mat The matrix to set to identity.
     *  \remarks Each diagonal element of \a mat is set to one, each other element is set to zero. */
    template<unsigned int n, typename T>
      void setIdentity( Matmnt<n,n,T> & mat );


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // non-member functions, specialized for m,n == 3
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    /*! \brief Test if a 3 by 3 matrix represents a rotation.
     *  \param mat A constant reference to the matrix to test.
     *  \param eps An optional value giving the allowed epsilon. The default is a type dependent value.
     *  \return \c true, if the matrix represents a rotation, otherwise \c false.
     *  \remarks A 3 by 3 matrix is considered to be a rotation, if it is normalized, orthogonal, and its
     *  determinant is one.
     *  \sa isIdentity, isNull, isNormalized, isOrthogonal */
    template<typename T>
      bool isRotation( const Matmnt<3,3,T> & mat, T eps = 9 * std::numeric_limits<T>::epsilon() )
    {
      return(   isNormalized( mat, eps )
            &&  isOrthogonal( mat, eps )
            && ( std::abs( determinant( mat ) - 1 ) <= eps ) );
    }

    /*! \brief Set the values of a 3 by 3 matrix using a normalized quaternion.
     *  \param mat A reference to the matrix to set.
     *  \param q A constant reference to the normalized quaternion to use.
     *  \return A reference to \a mat.
     *  \remarks The matrix is set to represent the same rotation as the normalized quaternion \a q.
     *  \note The behavior is undefined if \a q is not normalized. */
    template<typename T>
      Matmnt<3,3,T> & setMat( Matmnt<3,3,T> & mat, const Quatt<T> & q );

    /*! \brief Set the values of a 3 by 3 matrix using a normalized rotation axis and an angle.
     *  \param mat A reference to the matrix to set.
     *  \param axis A constant reference to the normalized rotation axis.
     *  \param angle The angle in radians to rotate around \a axis.
     *  \return A reference to \a mat.
     *  \remarks The matrix is set to represent the rotation by \a angle around \a axis.
     *  \note The behavior is undefined if \a axis is not normalized. */
    template<typename T>
      Matmnt<3,3,T> & setMat( Matmnt<3,3,T> & mat, const Vecnt<3,T> & axis, T angle );


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // non-member functions, specialized for m,n == 3, T == float
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    /*! \brief Decompose a 3 by 3 matrix.
     *  \param mat A constant reference to the matrix to decompose.
     *  \param orientation A reference to the quaternion getting the rotational part of the matrix.
     *  \param scaling A reference to the vector getting the scaling part of the matrix.
     *  \param scaleOrientation A reference to the quaternion getting the orientation of the scaling.
     *  \note The behavior is undefined, if the determinant of \a mat is too small, or the rank of \a mat
     *  is less than three. */
    DP_MATH_API void decompose( const Matmnt<3,3,float> &mat, Quatt<float> &orientation
                           , Vecnt<3,float> &scaling, Quatt<float> &scaleOrientation );


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // non-member functions, specialized for m,n == 4
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    /*!\brief Test if a 4 by 4 matrix represents a mirror transform
     * \param mat A const reference to the matrix to test.
     * \return \c true if the given matrix is a mirror transform, otherwise \c false. */
    template<typename T>
    bool isMirrorMatrix( const Matmnt<4,4,T>& mat )
    {
      const T* ptr = mat.getPtr();

      const Vecnt<3,T> &v0 = reinterpret_cast<const Vecnt<3,T>&>(ptr[0]);
      const Vecnt<3,T> &v1 = reinterpret_cast<const Vecnt<3,T>&>(ptr[4]);
      const Vecnt<3,T> &v2 = reinterpret_cast<const Vecnt<3,T>&>(ptr[8]);

      return (scalarTripleProduct(v0, v1, v2)) < 0;
    }

    /*! \brief Set the values of a 4 by 4 matrix by the constituents of a transformation.
     *  \param mat A reference to the matrix to set.
     *  \param ori A constant reference of the rotation part of the transformation.
     *  \param trans An optional constant reference to the translational part of the transformation. The
     *  default is a null vector.
     *  \param scale An optional constant reference to the scaling part of the transformation. The default
     *  is the identity scaling.
     *  \return A reference to \a mat. */
    template<typename T>
      Matmnt<4,4,T> & setMat( Matmnt<4,4,T> & mat, const Quatt<T> & ori
                                                 , const Vecnt<3,T> & trans = Vecnt<3,T>(0,0,0)
                                                 , const Vecnt<3,T> & scale = Vecnt<3,T>(1,1,1) )
    {
      Matmnt<3,3,T> m3( ori );
      mat[0] = Vecnt<4,T>( scale[0] * m3[0], 0 );
      mat[1] = Vecnt<4,T>( scale[1] * m3[1], 0 );
      mat[2] = Vecnt<4,T>( scale[2] * m3[2], 0 );
      mat[3] = Vecnt<4,T>( trans, 1 );
      return( mat );
    }


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // non-member functions, specialized for m,n == 4, T == float
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    /*! \brief Decompose a 4 by 4 matrix of floats.
     *  \param mat A constant reference to the matrix to decompose.
     *  \param translation A reference to the vector getting the translational part of the matrix.
     *  \param orientation A reference to the quaternion getting the rotational part of the matrix.
     *  \param scaling A reference to the vector getting the scaling part of the matrix.
     *  \param scaleOrientation A reference to the quaternion getting the orientation of the scaling.
     *  \note The behavior is undefined, if the determinant of \a mat is too small.
     *  \note Currently, the behavior is undefined, if the rank of \a mat is less than three. */
    DP_MATH_API void decompose( const Matmnt<4,4,float> &mat, Vecnt<3,float> &translation
                           , Quatt<float> &orientation, Vecnt<3,float> &scaling
                           , Quatt<float> &scaleOrientation );


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // Convenience type definitions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef Matmnt<3,3,float>   Mat33f;
    typedef Matmnt<3,3,double>  Mat33d;
    typedef Matmnt<4,4,float>   Mat44f;
    typedef Matmnt<4,4,double>  Mat44d;


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // inlined member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T>::Matmnt()
    {
    }

    template<unsigned int m, unsigned int n, typename T>
    template<unsigned int k, unsigned int l, typename S>
    inline Matmnt<m,n,T>::Matmnt( const Matmnt<k,l,S> & rhs )
    {
      for ( unsigned int i=0 ; i<std::min( m, k ) ; ++i )
      {
        m_mat[i] = Vecnt<n,T>(rhs[i]);
      }
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T>::Matmnt( const std::array<Vecnt<n,T>,m> & rows )
    {
      for ( unsigned int i=0 ; i<m ; i++ )
      {
        m_mat[i] = rows[i];
      }
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T>::Matmnt( Vecnt<n,T> const& v1, Vecnt<n,T> const& v2 )
    {
      MY_STATIC_ASSERT( m == 2 );
      m_mat[0] = v1;
      m_mat[1] = v2;
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T>::Matmnt( Vecnt<n,T> const& v1, Vecnt<n,T> const& v2, Vecnt<n,T> const& v3 )
    {
      MY_STATIC_ASSERT( m == 3 );
      m_mat[0] = v1;
      m_mat[1] = v2;
      m_mat[2] = v3;
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T>::Matmnt( Vecnt<n,T> const& v1, Vecnt<n,T> const& v2, Vecnt<n,T> const& v3, Vecnt<n,T> const& v4 )
    {
      MY_STATIC_ASSERT( m == 4 );
      m_mat[0] = v1;
      m_mat[1] = v2;
      m_mat[2] = v3;
      m_mat[3] = v4;
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T>::Matmnt( const std::array<T,m*n> & scalars )
    {
      for ( unsigned int i=0, idx=0 ; i<m ; i++ )
      {
        for ( unsigned int j=0 ; j<n ; j++, idx++ )
        {
          m_mat[i][j] = scalars[idx];
        }
      }
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T>::Matmnt( const Vecnt<3,T> & axis, T angle )
    {
      MY_STATIC_ASSERT( ( m == 3 ) && ( n == 3 ) );
      setMat( *this, axis, angle );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T>::Matmnt( const Quatt<T> & ori )
    {
      MY_STATIC_ASSERT( ( m == 3 ) && ( n == 3 ) );
      setMat( *this, ori );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T>::Matmnt( const Quatt<T> & ori, const Vecnt<3,T> & trans )
    {
      MY_STATIC_ASSERT( ( m == 4 ) && ( n == 4 ) );
      setMat( *this, ori, trans );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T>::Matmnt( const Quatt<T> & ori, const Vecnt<3,T> & trans, const Vecnt<3,T> & scale )
    {
      MY_STATIC_ASSERT( ( m == 4 ) && ( n == 4 ) );
      setMat( *this, ori, trans, scale );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline const T * Matmnt<m,n,T>::getPtr() const
    {
      return( m_mat[0].getPtr() );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline bool Matmnt<m,n,T>::invert()
    {
      MY_STATIC_ASSERT( m == n );
      Matmnt<n,n,T> tmp;
      bool ok = dp::math::invert( *this, tmp );
      if ( ok )
      {
        *this = tmp;
      }
      return( ok );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Vecnt<n,T> & Matmnt<m,n,T>::operator[]( unsigned int i )
    {
      MY_ASSERT( i < m );
      return( m_mat[i] );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline const Vecnt<n,T> & Matmnt<m,n,T>::operator[]( unsigned int i ) const
    {
      MY_ASSERT( i < m );
      return( m_mat[i] );
    }

    template<unsigned int m, unsigned int n, typename T>
    template<typename S>
    inline Matmnt<m,n,T> & Matmnt<m,n,T>::operator+=( const Matmnt<m,n,S> & rhs )
    {
      for ( unsigned int i=0 ; i<m ; ++i )
      {
        m_mat[i] += rhs[i];
      }
      return( *this );
    }

    template<unsigned int m, unsigned int n, typename T>
    template<typename S>
    inline Matmnt<m,n,T> & Matmnt<m,n,T>::operator-=( const Matmnt<m,n,S> & rhs )
    {
      for ( unsigned int i=0 ; i<m ; ++i )
      {
        m_mat[i] -= rhs[i];
      }
      return( *this );
    }

    template<unsigned int m, unsigned int n, typename T>
    template<typename S>
    inline Matmnt<m,n,T> & Matmnt<m,n,T>::operator*=( S s )
    {
      for ( unsigned int i=0 ; i<m ; ++i )
      {
        m_mat[i] *= s;
      }
      return( *this );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T> & Matmnt<m,n,T>::operator*=( const Matmnt<m,n,T> & rhs )
    {
      *this = *this * rhs;
      return( *this );
    }

    template<unsigned int m, unsigned int n, typename T>
    template<typename S>
    inline Matmnt<m,n,T> & Matmnt<m,n,T>::operator/=( S s )
    {
      for ( unsigned int i=0 ; i<m ; ++i )
      {
        m_mat[i] /= s;
      }
      return( *this );
    }


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // inlined non-member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    template<unsigned int n, typename T, unsigned int k>
    inline T calculateDeterminant( const Matmnt<n,n,T> & mat, const Vecnt<k,unsigned int> & first, const Vecnt<k,unsigned int> & second )
    {
      Vecnt<k-1,unsigned int> subFirst, subSecond;
      for ( unsigned int i=1 ; i<k ; i++ )
      {
        subFirst[i-1] = first[i];
        subSecond[i-1] = second[i];
      }
      T sum(0);
      T sign(1);
      for ( unsigned int i=0 ; i<k ; i++ )
      {
        sum += sign * mat[first[0]][second[i]] * calculateDeterminant( mat, subFirst, subSecond );
        sign = - sign;
        if ( i < k-1 )
        {
          subSecond[i] = second[i];
        }
      }
      return( sum );
    }

    template<unsigned int n, typename T>
    inline T calculateDeterminant( const Matmnt<n,n,T> & mat, const Vecnt<1,unsigned int> & first, const Vecnt<1,unsigned int> & second )
    {
      return( mat[first[0]][second[0]] );
    }

    template<unsigned n, typename T>
    inline T determinant( const Matmnt<n,n,T> & mat )
    {
      Vecnt<n,unsigned int> first, second;
      for ( unsigned int i=0 ; i<n ; i++)
      {
        first[i] = i;
        second[i] = i;
      }
      return( calculateDeterminant( mat, first, second ) );
    }

    template<unsigned int n, typename T>
    inline bool invert( const Matmnt<n,n,T> & mIn, Matmnt<n,n,T> & mOut )
    {
      mOut = mIn;

      unsigned int p[n];

      bool ok = true;
      for ( unsigned int k=0 ; ok && k<n ; ++k )
      {
        T max(0);
        p[k] = 0;
        for ( unsigned int i=k ; ok && i<n ; ++i )
        {
          T s(0);
          for ( unsigned int j=k ; j<n ; ++j )
          {
            s += std::abs( mOut[i][j] );
          }
          ok = ( std::numeric_limits<T>::epsilon() < std::abs(s) );
          if ( ok )
          {
            T q = std::abs( mOut[i][k] ) / s;
            if ( q > max )
            {
              max = q;
              p[k] = i;
            }
          }
        }

        ok = ( std::numeric_limits<T>::epsilon() < max );
        if ( ok )
        {
          if ( p[k] != k )
          {
            for ( unsigned int j=0 ; j<n ; ++j )
            {
              std::swap( mOut[k][j], mOut[p[k]][j] );
            }
          }

          T pivot = mOut[k][k];
          ok = ( std::numeric_limits<T>::epsilon() < std::abs( pivot ) );
          if ( ok )
          {
            for ( unsigned int j=0 ; j<n ; ++j )
            {
              if ( j != k )
              {
                mOut[k][j] /= - pivot;
                for ( unsigned int i=0 ; i<n ; ++i )
                {
                  if ( i != k )
                  {
                    mOut[i][j] += mOut[i][k] * mOut[k][j];
                  }
                }
              }
            }

            for ( unsigned int i=0 ; i<n ; ++i )
            {
              mOut[i][k] /= pivot;
            }
            mOut[k][k] = 1.0f / pivot;
          }
        }
      }

      if ( ok )
      {
        for ( unsigned int k=n-2 ; k<n ; --k )  //  NOTE: ( unsigned int k < n ) <=> ( int k >= 0 )
        {
          if ( p[k] != k )
          {
            for ( unsigned int i=0 ; i<n ; ++i )
            {
              std::swap( mOut[i][k], mOut[i][p[k]] );
            }
          }
        }
      }

      return( ok );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline T maxElement( const Matmnt<m,n,T> & mat )
    {
      T me = maxElement( mat[0] );
      for ( unsigned int i=1 ; i<m ; ++i )
      {
        T t = maxElement( mat[i] );
        if ( me < t )
        {
          me = t;
        }
      }
      return( me );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline T minElement( const Matmnt<m,n,T> & mat )
    {
      T me = minElement( mat[0] );
      for ( unsigned int i=1 ; i<m ; ++i )
      {
        T t = minElement( mat[i] );
        if ( t < me )
        {
          me = t;
        }
      }
      return( me );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline bool operator==( const Matmnt<m,n,T> & m0, const Matmnt<m,n,T> & m1 )
    {
      bool eq = true;
      for ( unsigned int i=0 ; i<m && eq ; ++i )
      {
        eq = ( m0[i] == m1[i] );
      }
      return( eq );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline bool operator!=( const Matmnt<m,n,T> & m0, const Matmnt<m,n,T> & m1 )
    {
      return( ! ( m0 == m1 ) );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<n,m,T> operator~( const Matmnt<m,n,T> & mat )
    {
      Matmnt<n,m,T> ret;
      for ( unsigned int i=0 ; i<n ; ++i )
      {
        for ( unsigned int j=0 ; j<m ; ++j )
        {
          ret[i][j] = mat[j][i];
        }
      }
      return( ret );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T> operator+( const Matmnt<m,n,T> & m0, const Matmnt<m,n,T> & m1 )
    {
      Matmnt<m,n,T> ret(m0);
      ret += m1;
      return( ret );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T> operator-( const Matmnt<m,n,T> & mat )
    {
      Matmnt<m,n,T> ret;
      for ( unsigned int i=0 ; i<m ; ++i )
      {
        ret[i] = -mat[i];
      }
      return( ret );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T> operator-( const Matmnt<m,n,T> & m0, const Matmnt<m,n,T> & m1 )
    {
      Matmnt<m,n,T> ret(m0);
      ret -= m1;
      return( ret );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T> operator*( const Matmnt<m,n,T> & mat, T s )
    {
      Matmnt<m,n,T> ret(mat);
      ret *= s;
      return( ret );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T> operator*( T s, const Matmnt<m,n,T> & mat )
    {
      return( mat * s );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Vecnt<m,T> operator*( const Matmnt<m,n,T> & mat, const Vecnt<n,T> & v )
    {
      Vecnt<m,T> ret;
      for ( unsigned int i=0 ; i<m ; ++i )
      {
        ret[i] = mat[i] * v;
      }
      return( ret );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Vecnt<n,T> operator*( const Vecnt<m,T> & v, const Matmnt<m,n,T> & mat )
    {
      Vecnt<n,T> ret;
      for ( unsigned int i=0 ; i<n ; ++i )
      {
        ret[i] = 0;
        for ( unsigned int j=0 ; j<m ; ++j )
        {
          ret[i] += v[j] * mat[j][i];
        }
      }
      return( ret );
    }

    template<unsigned int m, unsigned int n, unsigned int k, typename T>
    inline Matmnt<m,k,T> operator*( const Matmnt<m,n,T> & m0, const Matmnt<n,k,T> & m1 )
    {
      Matmnt<m,k,T> ret;
      for ( unsigned int i=0 ; i<m ; ++i )
      {
        for ( unsigned int j=0 ; j<k ; ++j )
        {
          ret[i][j] = 0;
          for ( unsigned int l=0 ; l<n ; ++l )
          {
            ret[i][j] += m0[i][l] * m1[l][j];
          }
        }
      }
      return( ret );
    }

    template<unsigned int m, unsigned int n, typename T>
    inline Matmnt<m,n,T> operator/( const Matmnt<m,n,T> & mat, T s )
    {
      Matmnt<m,n,T> ret(mat);
      ret /= s;
      return( ret );
    }

    template<unsigned int n, typename T>
    void setIdentity( Matmnt<n,n,T> & mat )
    {
      for ( unsigned int i=0 ; i<n ; ++i )
      {
        for ( unsigned int j=0 ; j<i ; ++j )
        {
          mat[i][j] = T(0);
        }
        mat[i][i] = T(1);
        for ( unsigned int j=i+1 ; j<n ; ++j )
        {
          mat[i][j] = T(0);
        }
      }
    }


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // inlined non-member functions, specialized for m,n == 3
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    template<typename T>
    inline T determinant( const Matmnt<3,3,T> & mat )
    {
      return( mat[0] * ( mat[1] ^ mat[2] ) );
    }

    template<typename T>
    inline bool invert( const Matmnt<3,3,T> & mIn, Matmnt<3,3,T> & mOut )
    {
      double adj00 =   ( mIn[1][1] * mIn[2][2] - mIn[1][2] * mIn[2][1] );
      double adj10 = - ( mIn[1][0] * mIn[2][2] - mIn[1][2] * mIn[2][0] );
      double adj20 =   ( mIn[1][0] * mIn[2][1] - mIn[1][1] * mIn[2][0] );
      double det = mIn[0][0] * adj00 + mIn[0][1] * adj10 + mIn[0][2] * adj20;
      bool ok = ( std::numeric_limits<double>::epsilon() < abs( det ) );
      if ( ok )
      {
        double invDet = 1.0 / det;
        mOut[0][0] = T(   adj00 * invDet );
        mOut[0][1] = T( - ( mIn[0][1] * mIn[2][2] - mIn[0][2] * mIn[2][1] ) * invDet );
        mOut[0][2] = T(   ( mIn[0][1] * mIn[1][2] - mIn[0][2] * mIn[1][1] ) * invDet );
        mOut[1][0] = T(   adj10 * invDet );
        mOut[1][1] = T(   ( mIn[0][0] * mIn[2][2] - mIn[0][2] * mIn[2][0] ) * invDet );
        mOut[1][2] = T( - ( mIn[0][0] * mIn[1][2] - mIn[0][2] * mIn[1][0] ) * invDet );
        mOut[2][0] = T(   adj20 * invDet );
        mOut[2][1] = T( - ( mIn[0][0] * mIn[2][1] - mIn[0][1] * mIn[2][0] ) * invDet );
        mOut[2][2] = T(   ( mIn[0][0] * mIn[1][1] - mIn[0][1] * mIn[1][0] ) * invDet );
      }
      return( ok );
    }

    template<typename T>
    Matmnt<3,3,T> & setMat( Matmnt<3,3,T> & mat, const Vecnt<3,T> & axis, T angle )
    {
      T c = cos( angle );
      T s = sin( angle );
      T t = 1 - c;
      T x = axis[0];
      T y = axis[1];
      T z = axis[2];

      mat[0] = Vecnt<3,T>( t * x * x + c,     t * x * y + s * z, t * x * z - s * y );
      mat[1] = Vecnt<3,T>( t * x * y - s * z, t * y * y + c,     t * y * z + s * x );
      mat[2] = Vecnt<3,T>( t * x * z + s * y, t * y * z - s * x, t * z * z + c     );

      return( mat );
    }

    template<typename T>
    inline Matmnt<3,3,T> & setMat( Matmnt<3,3,T> & mat, const Quatt<T> & q )
    {
      T x = q[0];
      T y = q[1];
      T z = q[2];
      T w = q[3];

      mat[0] = Vecnt<3,T>( 1 - 2 * ( y * y + z * z ), 2 * ( x * y + z * w ),     2 * ( x * z - y * w )     );
      mat[1] = Vecnt<3,T>( 2 * ( x * y - z * w ),     1 - 2 * ( x * x + z * z ), 2 * ( y * z + x * w )     );
      mat[2] = Vecnt<3,T>( 2 * ( x * z + y * w ),     2 * ( y * z - x * w ),     1 - 2 * ( x * x + y * y ) );

      return( mat );
    }


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // inlined non-member functions, specialized for m,n == 4
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    template<typename T>
    inline T determinant( const Matmnt<4,4,T> & mat )
    {
      const T a0 = mat[0][0]*mat[1][1] - mat[0][1]*mat[1][0];
      const T a1 = mat[0][0]*mat[1][2] - mat[0][2]*mat[1][0];
      const T a2 = mat[0][0]*mat[1][3] - mat[0][3]*mat[1][0];
      const T a3 = mat[0][1]*mat[1][2] - mat[0][2]*mat[1][1];
      const T a4 = mat[0][1]*mat[1][3] - mat[0][3]*mat[1][1];
      const T a5 = mat[0][2]*mat[1][3] - mat[0][3]*mat[1][2];
      const T b0 = mat[2][0]*mat[3][1] - mat[2][1]*mat[3][0];
      const T b1 = mat[2][0]*mat[3][2] - mat[2][2]*mat[3][0];
      const T b2 = mat[2][0]*mat[3][3] - mat[2][3]*mat[3][0];
      const T b3 = mat[2][1]*mat[3][2] - mat[2][2]*mat[3][1];
      const T b4 = mat[2][1]*mat[3][3] - mat[2][3]*mat[3][1];
      const T b5 = mat[2][2]*mat[3][3] - mat[2][3]*mat[3][2];
      return( a0*b5 - a1*b4 + a2*b3 + a3*b2 - a4*b1 + a5*b0 );
    }

    template<typename T>
    inline bool invert( const Matmnt<4,4,T> & mIn, Matmnt<4,4,T> & mOut )
    {
      T s0 = mIn[0][0] * mIn[1][1] - mIn[0][1] * mIn[1][0];   T c5 = mIn[2][2] * mIn[3][3] - mIn[2][3] * mIn[3][2];
      T s1 = mIn[0][0] * mIn[1][2] - mIn[0][2] * mIn[1][0];   T c4 = mIn[2][1] * mIn[3][3] - mIn[2][3] * mIn[3][1];
      T s2 = mIn[0][0] * mIn[1][3] - mIn[0][3] * mIn[1][0];   T c3 = mIn[2][1] * mIn[3][2] - mIn[2][2] * mIn[3][1];
      T s3 = mIn[0][1] * mIn[1][2] - mIn[0][2] * mIn[1][1];   T c2 = mIn[2][0] * mIn[3][3] - mIn[2][3] * mIn[3][0];
      T s4 = mIn[0][1] * mIn[1][3] - mIn[0][3] * mIn[1][1];   T c1 = mIn[2][0] * mIn[3][2] - mIn[2][2] * mIn[3][0];
      T s5 = mIn[0][2] * mIn[1][3] - mIn[0][3] * mIn[1][2];   T c0 = mIn[2][0] * mIn[3][1] - mIn[2][1] * mIn[3][0];
      T det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
      if ( det != T(0) )
      {
        T invDet = T(1.0) / det;
        mOut[0][0] = T( (   mIn[1][1] * c5 - mIn[1][2] * c4 + mIn[1][3] * c3 ) * invDet );
        mOut[0][1] = T( ( - mIn[0][1] * c5 + mIn[0][2] * c4 - mIn[0][3] * c3 ) * invDet );
        mOut[0][2] = T( (   mIn[3][1] * s5 - mIn[3][2] * s4 + mIn[3][3] * s3 ) * invDet );
        mOut[0][3] = T( ( - mIn[2][1] * s5 + mIn[2][2] * s4 - mIn[2][3] * s3 ) * invDet );
        mOut[1][0] = T( ( - mIn[1][0] * c5 + mIn[1][2] * c2 - mIn[1][3] * c1 ) * invDet );
        mOut[1][1] = T( (   mIn[0][0] * c5 - mIn[0][2] * c2 + mIn[0][3] * c1 ) * invDet );
        mOut[1][2] = T( ( - mIn[3][0] * s5 + mIn[3][2] * s2 - mIn[3][3] * s1 ) * invDet );
        mOut[1][3] = T( (   mIn[2][0] * s5 - mIn[2][2] * s2 + mIn[2][3] * s1 ) * invDet );
        mOut[2][0] = T( (   mIn[1][0] * c4 - mIn[1][1] * c2 + mIn[1][3] * c0 ) * invDet );
        mOut[2][1] = T( ( - mIn[0][0] * c4 + mIn[0][1] * c2 - mIn[0][3] * c0 ) * invDet );
        mOut[2][2] = T( (   mIn[3][0] * s4 - mIn[3][1] * s2 + mIn[3][3] * s0 ) * invDet );
        mOut[2][3] = T( ( - mIn[2][0] * s4 + mIn[2][1] * s2 - mIn[2][3] * s0 ) * invDet );
        mOut[3][0] = T( ( - mIn[1][0] * c3 + mIn[1][1] * c1 - mIn[1][2] * c0 ) * invDet );
        mOut[3][1] = T( (   mIn[0][0] * c3 - mIn[0][1] * c1 + mIn[0][2] * c0 ) * invDet );
        mOut[3][2] = T( ( - mIn[3][0] * s3 + mIn[3][1] * s1 - mIn[3][2] * s0 ) * invDet );
        mOut[3][3] = T( (   mIn[2][0] * s3 - mIn[2][1] * s1 + mIn[2][2] * s0 ) * invDet );
        return( true );
      }
      return( false );
    }

    template<typename T>
    inline bool invertTranspose( const Matmnt<4,4,T> & mIn, Matmnt<4,4,T> & mOut )
    {
      T s0 = mIn[0][0] * mIn[1][1] - mIn[0][1] * mIn[1][0];   T c5 = mIn[2][2] * mIn[3][3] - mIn[2][3] * mIn[3][2];
      T s1 = mIn[0][0] * mIn[1][2] - mIn[0][2] * mIn[1][0];   T c4 = mIn[2][1] * mIn[3][3] - mIn[2][3] * mIn[3][1];
      T s2 = mIn[0][0] * mIn[1][3] - mIn[0][3] * mIn[1][0];   T c3 = mIn[2][1] * mIn[3][2] - mIn[2][2] * mIn[3][1];
      T s3 = mIn[0][1] * mIn[1][2] - mIn[0][2] * mIn[1][1];   T c2 = mIn[2][0] * mIn[3][3] - mIn[2][3] * mIn[3][0];
      T s4 = mIn[0][1] * mIn[1][3] - mIn[0][3] * mIn[1][1];   T c1 = mIn[2][0] * mIn[3][2] - mIn[2][2] * mIn[3][0];
      T s5 = mIn[0][2] * mIn[1][3] - mIn[0][3] * mIn[1][2];   T c0 = mIn[2][0] * mIn[3][1] - mIn[2][1] * mIn[3][0];
      T det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
      if ( det != 0 )
      {
        T invDet = T(1.0) / det;
        mOut[0][0] = T( (   mIn[1][1] * c5 - mIn[1][2] * c4 + mIn[1][3] * c3 ) * invDet );
        mOut[1][0] = T( ( - mIn[0][1] * c5 + mIn[0][2] * c4 - mIn[0][3] * c3 ) * invDet );
        mOut[2][0] = T( (   mIn[3][1] * s5 - mIn[3][2] * s4 + mIn[3][3] * s3 ) * invDet );
        mOut[3][0] = T( ( - mIn[2][1] * s5 + mIn[2][2] * s4 - mIn[2][3] * s3 ) * invDet );
        mOut[0][1] = T( ( - mIn[1][0] * c5 + mIn[1][2] * c2 - mIn[1][3] * c1 ) * invDet );
        mOut[1][1] = T( (   mIn[0][0] * c5 - mIn[0][2] * c2 + mIn[0][3] * c1 ) * invDet );
        mOut[2][1] = T( ( - mIn[3][0] * s5 + mIn[3][2] * s2 - mIn[3][3] * s1 ) * invDet );
        mOut[3][1] = T( (   mIn[2][0] * s5 - mIn[2][2] * s2 + mIn[2][3] * s1 ) * invDet );
        mOut[0][2] = T( (   mIn[1][0] * c4 - mIn[1][1] * c2 + mIn[1][3] * c0 ) * invDet );
        mOut[1][2] = T( ( - mIn[0][0] * c4 + mIn[0][1] * c2 - mIn[0][3] * c0 ) * invDet );
        mOut[2][2] = T( (   mIn[3][0] * s4 - mIn[3][1] * s2 + mIn[3][3] * s0 ) * invDet );
        mOut[3][2] = T( ( - mIn[2][0] * s4 + mIn[2][1] * s2 - mIn[2][3] * s0 ) * invDet );
        mOut[0][3] = T( ( - mIn[1][0] * c3 + mIn[1][1] * c1 - mIn[1][2] * c0 ) * invDet );
        mOut[1][3] = T( (   mIn[0][0] * c3 - mIn[0][1] * c1 + mIn[0][2] * c0 ) * invDet );
        mOut[2][3] = T( ( - mIn[3][0] * s3 + mIn[3][1] * s1 - mIn[3][2] * s0 ) * invDet );
        mOut[3][3] = T( (   mIn[2][0] * s3 - mIn[2][1] * s1 + mIn[2][2] * s0 ) * invDet );
        return( true );
      }
      return( false );
    }

    template<typename T>
    inline bool invertDouble( const Matmnt<4,4,T> & mIn, Matmnt<4,4,T> & mOut )
    {
      double s0 = mIn[0][0] * mIn[1][1] - mIn[0][1] * mIn[1][0];   double c5 = mIn[2][2] * mIn[3][3] - mIn[2][3] * mIn[3][2];
      double s1 = mIn[0][0] * mIn[1][2] - mIn[0][2] * mIn[1][0];   double c4 = mIn[2][1] * mIn[3][3] - mIn[2][3] * mIn[3][1];
      double s2 = mIn[0][0] * mIn[1][3] - mIn[0][3] * mIn[1][0];   double c3 = mIn[2][1] * mIn[3][2] - mIn[2][2] * mIn[3][1];
      double s3 = mIn[0][1] * mIn[1][2] - mIn[0][2] * mIn[1][1];   double c2 = mIn[2][0] * mIn[3][3] - mIn[2][3] * mIn[3][0];
      double s4 = mIn[0][1] * mIn[1][3] - mIn[0][3] * mIn[1][1];   double c1 = mIn[2][0] * mIn[3][2] - mIn[2][2] * mIn[3][0];
      double s5 = mIn[0][2] * mIn[1][3] - mIn[0][3] * mIn[1][2];   double c0 = mIn[2][0] * mIn[3][1] - mIn[2][1] * mIn[3][0];
      double det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
      if ( ( std::numeric_limits<double>::epsilon() < abs( det ) ) )
      {
        double invDet = 1.0 / det;
        mOut[0][0] = T( (   mIn[1][1] * c5 - mIn[1][2] * c4 + mIn[1][3] * c3 ) * invDet );
        mOut[0][1] = T( ( - mIn[0][1] * c5 + mIn[0][2] * c4 - mIn[0][3] * c3 ) * invDet );
        mOut[0][2] = T( (   mIn[3][1] * s5 - mIn[3][2] * s4 + mIn[3][3] * s3 ) * invDet );
        mOut[0][3] = T( ( - mIn[2][1] * s5 + mIn[2][2] * s4 - mIn[2][3] * s3 ) * invDet );
        mOut[1][0] = T( ( - mIn[1][0] * c5 + mIn[1][2] * c2 - mIn[1][3] * c1 ) * invDet );
        mOut[1][1] = T( (   mIn[0][0] * c5 - mIn[0][2] * c2 + mIn[0][3] * c1 ) * invDet );
        mOut[1][2] = T( ( - mIn[3][0] * s5 + mIn[3][2] * s2 - mIn[3][3] * s1 ) * invDet );
        mOut[1][3] = T( (   mIn[2][0] * s5 - mIn[2][2] * s2 + mIn[2][3] * s1 ) * invDet );
        mOut[2][0] = T( (   mIn[1][0] * c4 - mIn[1][1] * c2 + mIn[1][3] * c0 ) * invDet );
        mOut[2][1] = T( ( - mIn[0][0] * c4 + mIn[0][1] * c2 - mIn[0][3] * c0 ) * invDet );
        mOut[2][2] = T( (   mIn[3][0] * s4 - mIn[3][1] * s2 + mIn[3][3] * s0 ) * invDet );
        mOut[2][3] = T( ( - mIn[2][0] * s4 + mIn[2][1] * s2 - mIn[2][3] * s0 ) * invDet );
        mOut[3][0] = T( ( - mIn[1][0] * c3 + mIn[1][1] * c1 - mIn[1][2] * c0 ) * invDet );
        mOut[3][1] = T( (   mIn[0][0] * c3 - mIn[0][1] * c1 + mIn[0][2] * c0 ) * invDet );
        mOut[3][2] = T( ( - mIn[3][0] * s3 + mIn[3][1] * s1 - mIn[3][2] * s0 ) * invDet );
        mOut[3][3] = T( (   mIn[2][0] * s3 - mIn[2][1] * s1 + mIn[2][2] * s0 ) * invDet );
        return( true );
      }
      return( false );
    }

    /*! \brief makeLookAt defines a viewing transformation.
     * \param eye The position of the eye point.
     * \param center The position of the reference point.
     * \param up The direction of the up vector.
     * \remarks The makeLookAt function creates a viewing matrix derived from an eye point, a reference point indicating the center
     * of the scene, and an up vector. The matrix maps the reference point to the negative z-axis and the eye point to the
     * origin, so that when you use a typical projection matrix, the center of the scene maps to the center of the viewport.
     * Similarly, the direction described by the up vector projected onto the viewing plane is mapped to the positive y-axis so that
     * it points upward in the viewport. The up vector must not be parallel to the line of sight from the eye to the reference point.
     * \note This documentation is adapted from gluLookAt, and is courtesy of SGI.
     */
    template <typename T>
    inline Matmnt<4,4,T> makeLookAt( const Vecnt<3,T> & eye, const Vecnt<3,T> & center, const Vecnt<3,T> & up )
    {
      Vecnt<3,T> f = center - eye;
      normalize( f );

  #ifndef NDEBUG
      // assure up is not parallel to vector from eye to center
      Vecnt<3,T> nup = up;
      normalize( nup );
      T dot = f * nup;
      MY_ASSERT( dot != T(1) && dot != T(-1) );
  #endif

      Vecnt<3,T> s = f ^ up;
      normalize( s );
      Vecnt<3,T> u = s ^ f;

      Matmnt<4,4,T> transmat;
      transmat[0] = Vec4f( T(1),      T(0),     T(0),    T(0) );
      transmat[1] = Vec4f( T(0),      T(1),     T(0),    T(0) );
      transmat[2] = Vec4f( T(0),      T(0),     T(1),    T(0) );
      transmat[3] = Vec4f( -eye[0],   -eye[1],  -eye[2],  T(1) );

      Matmnt<4,4,T> orimat;
      orimat[0] = Vec4f( s[0],    u[0],  -f[0],   T(0) );
      orimat[1] = Vec4f( s[1],    u[1],  -f[1],   T(0) );
      orimat[2] = Vec4f( s[2],    u[2],  -f[2],   T(0) );
      orimat[3] = Vec4f( T(0),    T(0),   T(0),   T(1) );

      // must premultiply translation
      return transmat * orimat;
    }

    /*! \brief makeOrtho defines an orthographic projection matrix.
     * \param left Coordinate for the left vertical clipping plane.
     * \param right Coordinate for the right vertical clipping plane.
     * \param bottom Coordinate for the bottom horizontal clipping plane.
     * \param top Coordinate for the top horizontal clipping plane.
     * \param znear The distance to the near clipping plane.  This distance is negative if the plane is behind the viewer.
     * \param zfar The distance to the far clipping plane.  This distance is negative if the plane is behind the viewer.
     * \remarks The makeOrtho function describes a perspective matrix that produces a parallel projection.  Assuming this function
     * will be used to build a camera's Projection matrix, the (left, bottom, znear) and (right, top, znear) parameters specify the
     * points on the near clipping plane that are mapped to the lower-left and upper-right corners of the window, respectively,
     * assuming that the eye is located at (0, 0, 0). The far parameter specifies the location of the far clipping plane. Both znear
     * and zfar can be either positive or negative.
     * \note This documentation is adapted from glOrtho, and is courtesy of SGI.
     */
    template <typename T>
    inline Matmnt<4,4,T> makeOrtho( T left,    T right,
                                    T bottom,  T top,
                                    T znear,   T zfar )
    {
      MY_ASSERT( (left != right) && (bottom != top) && (znear != zfar) && (zfar > znear) );

      return Matmnt<4,4,T>( { T(2)/(right-left),           T(0),                            T(0),                       T(0)
                            , T(0),                        T(2)/(top-bottom),               T(0),                       T(0)
                            , T(0),                        T(0),                            T(-2)/(zfar-znear),         T(0)
                            , -(right+left)/(right-left),  -(top+bottom)/(top-bottom),      -(zfar+znear)/(zfar-znear), T(1) } );
    }

    /*! \brief makeFrustum defines a perspective projection matrix.
     * \param left Coordinate for the left vertical clipping plane.
     * \param right Coordinate for the right vertical clipping plane.
     * \param bottom Coordinate for the bottom horizontal clipping plane.
     * \param top Coordinate for the top horizontal clipping plane.
     * \param znear The distance to the near clipping plane.  The value must be greater than zero.
     * \param zfar The distance to the far clipping plane.  The value must be greater than znear.
     * \remarks The makeFrustum function describes a perspective matrix that produces a perspective projection.  Assuming this function
     * will be used to build a camera's Projection matrix, the (left, bottom, znear) and (right, top, znear) parameters specify the
     * points on the near clipping plane that are mapped to the lower-left and upper-right corners of the window, respectively,
     * assuming that the eye is located at (0,0,0). The zfar parameter specifies the location of the far clipping plane.  Both znear
     * and zfar must be positive.
     * \note This documentation is adapted from glFrustum, and is courtesy of SGI.
     */
    template <typename T>
    inline Matmnt<4,4,T> makeFrustum( T left,    T right,
                                      T bottom,  T top,
                                      T znear,   T zfar )
    {
      // near and far must be greater than zero
      MY_ASSERT( (znear > T(0)) && (zfar > T(0)) && (zfar > znear) );
      MY_ASSERT( (left != right) && (bottom != top) && (znear != zfar) );

      T v0 =  (right+left)/(right-left);
      T v1 =  (top+bottom)/(top-bottom);
      T v2 = -(zfar+znear)/(zfar-znear);
      T v3 = T(-2)*zfar*znear/(zfar-znear);
      T v4 =  T(2)*znear/(right-left);
      T v5 =  T(2)*znear/(top-bottom);

      return Matmnt<4,4,T>( { v4,    T(0),  T(0),  T(0)
                            , T(0),  v5,    T(0),  T(0)
                            , v0,    v1,    v2,    T(-1)
                            , T(0),  T(0),  v3,    T(0)  } );
    }

    /*! \brief makePerspective builds a perspective projection matrix.
     * \param fovy The vertical field of view, in degrees.
     * \param aspect The ratio of the viewport width / height.
     * \param znear The distance to the near clipping plane.  The value must be greater than zero.
     * \param zfar The distance to the far clipping plane.  The value must be greater than znear.
     * \remarks Assuming makePerspective will be used to build a camera's Projection matrix, it specifies a viewing frustum into the
     * world coordinate system.  In general, the aspect ratio in makePerspective should match the aspect ratio of the associated
     * viewport.   For example, aspect = 2.0 means the viewer's angle of view is twice as wide in x as it is in y.  If the viewport
     * is twice as wide as it is tall, it displays the image without distortion.
     * \note This documentation is adapted from gluPerspective, and is courtesy of SGI.
     */
    template <typename T>
    inline Matmnt<4,4,T> makePerspective( T fovy, T aspect, T znear, T zfar )
    {
      MY_ASSERT( (znear > (T)0) && (zfar > (T)0) );

      T tanfov = tan( degToRad( fovy ) * (T)0.5 );
      T r      = tanfov * aspect * znear;
      T l      = -r;
      T t      = tanfov * znear;
      T b      = -t;

      return makeFrustum<T>( l, r, b, t, znear, zfar );
    }

    template<typename T>
    inline Vecnt<4,T> operator*( const Vecnt<4,T>& v, const Matmnt<4,4,T>& mat )
    {
      return Vecnt<4,T> (
        v[0] * mat[0][0] + v[1]*mat[1][0] + v[2]*mat[2][0] + v[3]*mat[3][0],
        v[0] * mat[0][1] + v[1]*mat[1][1] + v[2]*mat[2][1] + v[3]*mat[3][1],
        v[0] * mat[0][2] + v[1]*mat[1][2] + v[2]*mat[2][2] + v[3]*mat[3][2],
        v[0] * mat[0][3] + v[1]*mat[1][3] + v[2]*mat[2][3] + v[3]*mat[3][3] );
    }

    template<typename T>
    inline Matmnt<4,4,T> operator*( const Matmnt<4,4,T> & m0, const Matmnt<4,4,T> & m1 )
    {
      Matmnt<4,4,T> result;

      result[0] = Vecnt<4,T>(
        m0[0][0]*m1[0][0] + m0[0][1]*m1[1][0] + m0[0][2]*m1[2][0] + m0[0][3]*m1[3][0],
        m0[0][0]*m1[0][1] + m0[0][1]*m1[1][1] + m0[0][2]*m1[2][1] + m0[0][3]*m1[3][1],
        m0[0][0]*m1[0][2] + m0[0][1]*m1[1][2] + m0[0][2]*m1[2][2] + m0[0][3]*m1[3][2],
        m0[0][0]*m1[0][3] + m0[0][1]*m1[1][3] + m0[0][2]*m1[2][3] + m0[0][3]*m1[3][3]
      );

      result[1] = Vecnt<4,T>(
        m0[1][0]*m1[0][0] + m0[1][1]*m1[1][0] + m0[1][2]*m1[2][0] + m0[1][3]*m1[3][0],
        m0[1][0]*m1[0][1] + m0[1][1]*m1[1][1] + m0[1][2]*m1[2][1] + m0[1][3]*m1[3][1],
        m0[1][0]*m1[0][2] + m0[1][1]*m1[1][2] + m0[1][2]*m1[2][2] + m0[1][3]*m1[3][2],
        m0[1][0]*m1[0][3] + m0[1][1]*m1[1][3] + m0[1][2]*m1[2][3] + m0[1][3]*m1[3][3]
      );

      result[2] = Vecnt<4,T>(
        m0[2][0]*m1[0][0] + m0[2][1]*m1[1][0] + m0[2][2]*m1[2][0] + m0[2][3]*m1[3][0],
        m0[2][0]*m1[0][1] + m0[2][1]*m1[1][1] + m0[2][2]*m1[2][1] + m0[2][3]*m1[3][1],
        m0[2][0]*m1[0][2] + m0[2][1]*m1[1][2] + m0[2][2]*m1[2][2] + m0[2][3]*m1[3][2],
        m0[2][0]*m1[0][3] + m0[2][1]*m1[1][3] + m0[2][2]*m1[2][3] + m0[2][3]*m1[3][3]
      );

      result[3] = Vecnt<4,T>(
        m0[3][0]*m1[0][0] + m0[3][1]*m1[1][0] + m0[3][2]*m1[2][0] + m0[3][3]*m1[3][0],
        m0[3][0]*m1[0][1] + m0[3][1]*m1[1][1] + m0[3][2]*m1[2][1] + m0[3][3]*m1[3][1],
        m0[3][0]*m1[0][2] + m0[3][1]*m1[1][2] + m0[3][2]*m1[2][2] + m0[3][3]*m1[3][2],
        m0[3][0]*m1[0][3] + m0[3][1]*m1[1][3] + m0[3][2]*m1[2][3] + m0[3][3]*m1[3][3]
      );

      return result;
    }

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    // inlined non-member functions, specialized for m,n == 4, T == float
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    inline void decompose( const Matmnt<4,4,float> &mat, Vecnt<3,float> &translation
                         , Quatt<float> &orientation, Vecnt<3,float> &scaling
                         , Quatt<float> &scaleOrientation )
    {
      translation = Vecnt<3,float>( mat[3] );
      Matmnt<3,3,float> m33( mat );
      decompose( m33, orientation, scaling, scaleOrientation );
    }



    //! global identity matrix.
    extern DP_MATH_API const Mat44f  cIdentity44f;

  } // namespace math
} // namespace dp
