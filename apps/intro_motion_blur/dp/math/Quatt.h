// Copyright (c) 2012, NVIDIA Corporation. All rights reserved.
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

#include <dp/math/math.h>
#include <dp/math/Matmnt.h>
#include <dp/math/Vecnt.h>

namespace dp
{
  namespace math
  {

    template<unsigned int m, unsigned int n, typename T> class Matmnt;

    /*! \brief Quaternion class.
     *  \remarks Quaternions are an alternative to the 3x3 matrices that are typically used for 3-D
     *  rotations. A unit quaternion represents an axis in 3-D space and a rotation around that axis.
     *  Every rotation can be expressed that way. There are typedefs for the most common usage with \c
     *  float and \c double: Quatf, Quatd.
      * \note Only unit quaternions represent a rotation.  */
    template<typename T> class Quatt
    {
      public:
        /*! \brief Default constructor.
         *  \remarks For performance reasons no initialization is performed. */
        Quatt();

        /*! \brief Constructor using four scalar values.
         *  \param x X-component of the quaternion.
         *  \param y Y-component of the quaternion.
         *  \param z Z-component of the quaternion.
         *  \param w W-component of the quaternion.
         *  \note \c x, \c y, and \c z are \b not the x,y,z-component of the rotation axis, and \c w
         *  is \b not the rotation angle. If you have such values handy, use the constructor that
         *  takes the axis as a Vecnt<3,T> and an angle. */
        Quatt( T x, T y, T z, T w );

        /*! \brief Constructor using a Vecnt<4,T>.
         *  \param v Vector to construct the quaternion from.
         *  \remarks The four values of \c v are just copied over to the quaternion. It is assumed
         *  that this operation gives a normalized quaternion. */
        explicit Quatt( const Vecnt<4,T> & v );

        /*! \brief Copy constructor using a Quaternion of possibly different type.
         *  \param q The quaternion to copy. */
        template<typename S>
          explicit Quatt( const Quatt<S> & q );

        /*! \brief Constructor using an axis and an angle.
         *  \param axis Axis to rotate around.
         *  \param angle Angle in radians to rotate.
         *  \remarks The resulting quaternion represents a rotation by \c angle (in radians) around
         *  \c axis. */
        Quatt( const Vecnt<3,T> & axis, T angle );

        /*! \brief Constructor by two vectors.
         *  \param v0 Start vector.
         *  \param v1 End vector.
         *  \remarks The resulting quaternion represents the rotation that maps \a vec0 to \a vec1.
         *  \note The quaternion out of two anti-parallel vectors is not uniquely defined. We select just
         *  one out of the possible candidates, which might not be the one you would expect. For better
         *  control on the quaternion in such a case, we recommend to use the constructor out of an axis
         *  and an angle. */
        Quatt( const Vecnt<3,T> & v0, const Vecnt<3,T> & v1 );

        /*! \brief Constructor by a rotation matrix.
         *  \param rot The rotation matrix to convert to a unit quaternion.
         *  \remarks The resulting quaternion represents the same rotation as the matrix. */
        explicit Quatt( const Matmnt<3,3,T> & rot );

        /*! \brief Normalize the quaternion.
         *  \return A reference to \c this, as the normalized quaternion.
         *  \remarks It is always assumed, that a quaternion is normalized. But when getting data from
         *  outside or due to numerically instabilities, a quaternion might become unnormalized. You
         *  can use this function then to normalize it again. */
        Quatt<T> & normalize();

        /*! \brief Non-constant subscript operator.
         *  \param i Index to the element to use (i=0,1,2,3).
         *  \return A reference to the \a i th element of the quaternion. */
        template<typename S>
          T & operator[]( S i );

        /*! \brief Constant subscript operator.
         *  \param i Index to the element to use (i=0,1,2,3).
         *  \return A reference to the constant \a i th element of the quaternion. */
        template<typename S>
          const T & operator[]( S i ) const;

        /*! \brief Quaternion assignment operator from a Quaternion of possibly different type.
         *  \param q The quaternion to assign.
         *  \return A reference to \c this, as the assigned Quaternion from q. */
        template<typename S>
          Quatt<T> & operator=( const Quatt<S> & q );

        /*! \brief Quaternion multiplication with a quaternion and assignment operator.
         *  \param q A Quaternion to multiply with.
         *  \return A reference to \c this, as the product (or concatenation) of \c this and \a q.
         *  \remarks Multiplying two quaternions give a quaternion that represents the concatenation
         *  of the rotations represented by the two quaternions. */
        Quatt<T> & operator*=( const Quatt<T> & q );

        /*! \brief Quaternion division by a quaternion and assignment operator.
         *  \param q A Quaternion to divide by.
         *  \return A reference to \c this, as the division of \c this by \a q.
         *  \remarks Dividing a quaternion by an other gives a quaternion that represents the
         *  concatenation of the rotation represented by the numerator quaternion and the rotation
         *  represented by the conjugated denumerator quaternion. */
        Quatt<T> & operator/=( const Quatt<T> & q );

      private:
        T m_quat[4];
    };


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // non-member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    /*! \brief Decompose a quaternion into an axis and an angle.
     *  \param q A reference to the constant quaternion to decompose.
     *  \param axis A reference to the resulting axis.
     *  \param angle A reference to the resulting angle. */
    template<typename T>
      void decompose( const Quatt<T> & q, Vecnt<3,T> & axis, T & angle );

    /*! \brief Determine the distance between two quaternions.
     *  \param q0 A reference to the left constant quaternion.
     *  \param q1 A reference to the right constant quaternion.
     *  \return The euclidean distance between \a q0 and \c q1, interpreted as Vecnt<4,T>. */
    template<typename T>
      T distance( const Quatt<T> & q0, const Quatt<T> & q1 );

    /*! \brief Test if a quaternion is normalized.
     *  \param q A reference to the constant quaternion to test.
     *  \param eps An optional value giving the allowed epsilon. The default is a type dependent value.
     *  \return \c true if the quaternion is normalized, otherwise \c false.
     *  \remarks A quaternion \a q is considered to be normalized, when it's magnitude differs less
     *  than some small value \a eps from one. */
    template<typename T>
      bool isNormalized( const Quatt<T> & q, T eps = std::numeric_limits<T>::epsilon() * 8 )
    {
      return( std::abs( magnitude( q ) - 1 ) <= eps );
    }

    /*! \brief Test if a quaternion is null.
     *  \param q A reference to the constant quaternion to test.
     *  \param eps An optional value giving the allowed epsilon. The default is a type dependent value.
     *  \return \c true if the quaternion is null, otherwise \c false.
     *  \remarks A quaternion \a q is considered to be null, if it's magnitude is less than some small
     *  value \a eps. */
    template<typename T>
      bool isNull( const Quatt<T> & q, T eps = std::numeric_limits<T>::epsilon() )
    {
      return( magnitude( q ) <= eps );
    }

    /*! \brief Determine the magnitude of a quaternion.
     *  \param q A reference to the quaternion to determine the magnitude of.
     *  \return The magnitude of the quaternion \a q.
     *  \remarks The magnitude of a normalized quaternion is one. */
    template<typename T>
      T magnitude( const Quatt<T> & q );

    /*! \brief Quaternion equality operator.
     *  \param q0 A reference to the constant left operand.
     *  \param q1 A reference to the constant right operand.
     *  \return \c true if the quaternions \a q0 and \a q are equal, otherwise \c false.
     *  \remarks Two quaternions are considered to be equal, if each component of the one quaternion
     *  deviates less than epsilon from the corresponding element of the other quaternion. */
    template<typename T>
      bool operator==( const Quatt<T> & q0, const Quatt<T> & q1 );

    /*! \brief Quaternion inequality operator.
     *  \param q0 A reference to the constant left operand.
     *  \param q1 A reference to the constant right operand.
     *  \return \c true if \a q0 is not equal to \a q1, otherwise \c false
     *  \remarks Two quaternions are considered to be equal, if each component of the one quaternion
     *  deviates less than epsilon from the corresponding element of the other quaternion. */
    template<typename T>
      bool operator!=( const Quatt<T> & q0, const Quatt<T> & q1 );

    /*! \brief Quaternion conjugation operator.
     *  \param q A reference to the constant quaternion to conjugate.
     *  \return The conjugation of \a q.
     *  \remarks The conjugation of a quaternion is a rotation of the same angle around the negated
     *  axis. */
    template<typename T>
      Quatt<T> operator~( const Quatt<T> & q );

    /*! \brief Negation operator.
     *  \param q A reference to the constant quaternion to get the negation from.
     *  \return The negation of \a q.
     *  \remarks The negation of a quaternion is a rotation around the same axis by the negated angle. */
    template<typename T>
      Quatt<T> operator-( const Quatt<T> & q );

    /*! \brief Quaternion multiplication operator.
     *  \param q0 A reference to the constant left operand.
     *  \param q1 A reference to the constant right operand.
     *  \return The product of \a q0 with \a a1.
     *  \remarks Multiplying two quaternions gives a quaternion that represents the concatenation of
     *  the rotation represented by the two quaternions. Besides rounding errors, the following
     *  equation holds:
     *  \code
     *    Matmnt<3,3,T>( q0 * q1 ) == Matmnt<3,3,T>( q0 ) * Matmnt<3,3,T>( q1 )
     *  \endcode */
    template<typename T>
      Quatt<T> operator*( const Quatt<T> & q0, const Quatt<T> & q1 );

    /*! \brief Quaternion multiplication operator with a vector.
     *  \param q A reference to the constant left operand.
     *  \param v A reference to the constant right operand.
     *  \return The vector \a v rotated by the inverse of \a q.
     *  \remarks Multiplying a quaternion \a q with a vector \a v applies the inverse rotation
     *  represented by \a q to \a v. */
    template<typename T>
      Vecnt<3,T> operator*( const Quatt<T> & q, const Vecnt<3,T> & v );

    /*! \brief Vector multiplication operator with a quaternion.
     *  \param v A reference to the constant left operand.
     *  \param q A reference to the constant right operand.
     *  \return The vector \a v rotated by \a q.
     *  \remarks Multiplying a vector \a v by a quaternion \a q applies the rotation represented by
     *  \a q to \a v. */
    template<typename T>
      Vecnt<3,T> operator*( const Vecnt<3,T> & v, const Quatt<T> & q );

    /*! \brief Quaternion division operator.
     *  \param q0 A reference to the constant left operand.
     *  \param q1 A reference to the constant right operand.
     *  \return A quaternion representing the quotient of \a q0 and \a q1.
     *  \remarks Dividing a quaternion by an other gives a quaternion that represents the
     *  concatenation of the rotation represented by the numerator quaternion and the rotation
     *  represented by the conjugated denumerator quaternion. */
    template<typename T>
      Quatt<T> operator/( const Quatt<T> & q0, const Quatt<T> & q1 );


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // non-member functions, specialized for T == float
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    /*! \brief Spherical linear interpolation between two quaternions \a q0 and \a q1.
     *  \param alpha The interpolation parameter.
     *  \param q0 The starting value.
     *  \param q1 The ending value.
     *  \return The quaternion that represents the spherical linear interpolation between \a q0 and \a
     *  q1. */
    DP_MATH_API Quatt<float> lerp( float alpha, const Quatt<float> & q0, const Quatt<float> & q1 );

    /*! \brief Spherical linear interpolation between two quaternions \a q0 and \a q1.
     *  \param alpha The interpolation parameter.
     *  \param q0 The starting value.
     *  \param q1 The ending value.
     *  \param qr The quaternion that represents the spherical linear interpolation between \a q0 and \a
     *  q1. */
    DP_MATH_API void lerp( float alpha, const Quatt<float> & q0, const Quatt<float> & q1, Quatt<float> &qr );

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // Convenience type definitions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    typedef Quatt<float>  Quatf;
    typedef Quatt<double> Quatd;


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // inlined member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    template<typename T>
    inline Quatt<T>::Quatt()
    {
    }

    template<typename T>
    inline Quatt<T>::Quatt( T x, T y, T z, T w )
    {
      m_quat[0] = x;
      m_quat[1] = y;
      m_quat[2] = z;
      m_quat[3] = w;
    }

    template<typename T>
    inline Quatt<T>::Quatt( const Vecnt<4,T> & v )
    {
      m_quat[0] = v[0];
      m_quat[1] = v[1];
      m_quat[2] = v[2];
      m_quat[3] = v[3];
    }

    template<typename T>
    template<typename S>
    inline Quatt<T>::Quatt( const Quatt<S> & q )
    {
      *this = q;
    }

    template<typename T>
    inline Quatt<T>::Quatt( const Vecnt<3,T> & axis, T angle )
    {
      T dummy = sin( T(0.5) * angle );
      m_quat[0] = axis[0] * dummy;
      m_quat[1] = axis[1] * dummy;
      m_quat[2] = axis[2] * dummy;
      m_quat[3] = cos( T(0.5) * angle );
    }

    template<typename T>
    inline Quatt<T>::Quatt( const Vecnt<3,T> & v0, const Vecnt<3,T> & v1 )
    {
      Vecnt<3,T> axis = v0 ^ v1;
      axis.normalize();
      T cosAngle = clamp( v0 * v1, -1.0f, 1.0f );   // make sure, cosine is in [-1.0,1.0]!
      if ( cosAngle + 1.0f < std::numeric_limits<T>::epsilon() )
      {
        // In case v0 and v1 are (closed to) anti-parallel, the standard
        // procedure would not create a valid quaternion.
        // As the rotation axis isn't uniquely defined in that case, we
        // just pick one.
        axis = orthonormal( v0 );
      }
      T s = sqrt( T(0.5) * ( 1 - cosAngle ) );
      m_quat[0] = axis[0] * s;
      m_quat[1] = axis[1] * s;
      m_quat[2] = axis[2] * s;
      m_quat[3] = sqrt( T(0.5) * ( 1 + cosAngle ) );
    }

    template<typename T>
    inline Quatt<T>::Quatt( const Matmnt<3,3,T> & rot )
    {
      T tr = rot[0][0] + rot[1][1] + rot[2][2] + 1;
      if ( std::numeric_limits<T>::epsilon() < tr )
      {
        T s = sqrt( tr );
        m_quat[3] = T(0.5) * s;
        s = T(0.5) / s;
        m_quat[0] = ( rot[1][2] - rot[2][1] ) * s;
        m_quat[1] = ( rot[2][0] - rot[0][2] ) * s;
        m_quat[2] = ( rot[0][1] - rot[1][0] ) * s;
      }
      else
      {
        unsigned int i = 0;
        if ( rot[i][i] < rot[1][1] )
        {
          i = 1;
        }
        if ( rot[i][i] < rot[2][2] )
        {
          i = 2;
        }
        unsigned int j = ( i + 1 ) % 3;
        unsigned int k = ( j + 1 ) % 3;
        T s = sqrt( rot[i][i] - rot[j][j] - rot[k][k] + 1 );
        m_quat[i] = T(0.5) * s;
        s = T(0.5) / s;
        m_quat[j] = ( rot[i][j] + rot[j][i] ) * s;
        m_quat[k] = ( rot[i][k] + rot[k][i] ) * s;
        m_quat[3] = ( rot[k][j] - rot[j][k] ) * s;
      }
      normalize();
    }

    template<typename T>
    inline Quatt<T> & Quatt<T>::normalize()
    {
      T mag = sqrt( square(m_quat[0]) + square(m_quat[1]) + square(m_quat[2]) + square(m_quat[3]) );
      T invMag = T(1) / mag;
      for ( int i=0 ; i<4 ; i++ )
      {
        m_quat[i] = m_quat[i] * invMag;
      }
      return( *this );
    }

    template<>
    inline Quatt<float> & Quatt<float>::normalize()
    {
      *this = (Quatt<double>(*this)).normalize();
      return( *this );
    }

    template<typename T>
    template<typename S>
    inline T & Quatt<T>::operator[]( S i )
    {
      MY_STATIC_ASSERT( std::numeric_limits<S>::is_integer );
      MY_ASSERT( 0 <= i && i <= 3 );
      return( m_quat[i] );
    }

    template<typename T>
    template<typename S>
    inline const T & Quatt<T>::operator[]( S i ) const
    {
      MY_STATIC_ASSERT( std::numeric_limits<S>::is_integer );
      MY_ASSERT( 0 <= i && i <= 3 );
      return( m_quat[i] );
    }

    template<typename T>
    template<typename S>
    inline Quatt<T> & Quatt<T>::operator=( const Quatt<S> & q )
    {
      m_quat[0] = T(q[0]);
      m_quat[1] = T(q[1]);
      m_quat[2] = T(q[2]);
      m_quat[3] = T(q[3]);
      return( *this );
    }

    template<typename T>
    inline Quatt<T> & Quatt<T>::operator*=( const Quatt<T> & q )
    {
      *this = *this * q;
      return( *this );
    }

    template<typename T>
    inline Quatt<T> & Quatt<T>::operator/=( const Quatt<T> & q )
    {
      *this = *this / q;
      return( *this );
    }


    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // inlined non-member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    template<typename T>
    inline void decompose( const Quatt<T> & q, Vecnt<3,T> & axis, T & angle )
    {
      angle = 2 * acos( q[3] );
      if ( angle < std::numeric_limits<T>::epsilon() )
      {
        //  no angle to rotate about => take just any one
        axis[0] = 0.0f;
        axis[1] = 0.0f;
        axis[2] = 1.0f;
      }
      else
      {
        T dummy = 1 / sin( T(0.5) * angle );
        axis[0] = q[0] * dummy;
        axis[1] = q[1] * dummy;
        axis[2] = q[2] * dummy;
        axis.normalize();
      }
    }

    template<typename T>
    inline T distance( const Quatt<T> & q0, const Quatt<T> & q1 )
    {
      return( sqrt( square( q0[0] - q1[0] )
                  + square( q0[1] - q1[1] )
                  + square( q0[2] - q1[2] )
                  + square( q0[3] - q1[3] ) ) );
    }

    template<typename T>
    inline T magnitude( const Quatt<T> & q )
    {
      return( sqrt( q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3] ) );
    }

    template<typename T>
    inline bool operator==( const Quatt<T> & q0, const Quatt<T> & q1 )
    {
      return(   ( std::abs( q0[0] - q1[0] ) < std::numeric_limits<T>::epsilon() )
            &&  ( std::abs( q0[1] - q1[1] ) < std::numeric_limits<T>::epsilon() )
            &&  ( std::abs( q0[2] - q1[2] ) < std::numeric_limits<T>::epsilon() )
            &&  ( std::abs( q0[3] - q1[3] ) < std::numeric_limits<T>::epsilon() ) );
    }

    template<typename T>
    inline bool operator!=( const Quatt<T> & q0, const Quatt<T> & q1 )
    {
      return( ! ( q0 == q1 ) );
    }

    template<typename T>
    inline Quatt<T> operator~( const Quatt<T> & q )
    {
      return( Quatt<T>( -q[0], -q[1], -q[2], q[3] ) );
    }

    template<typename T>
    inline Quatt<T> operator-( const Quatt<T> & q )
    {
      return( Quatt<T>( q[0], q[1], q[2], -q[3] ) );
    }

    template<typename T>
    inline Quatt<T> operator*( const Quatt<T> & q0, const Quatt<T> & q1 )
    {
      Quatt<T> q( q0[3]*q1[0] + q0[0]*q1[3] - q0[1]*q1[2] + q0[2]*q1[1]
                , q0[3]*q1[1] + q0[0]*q1[2] + q0[1]*q1[3] - q0[2]*q1[0]
                , q0[3]*q1[2] - q0[0]*q1[1] + q0[1]*q1[0] + q0[2]*q1[3]
                , q0[3]*q1[3] - q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2] );
      q.normalize();
      return( q );
    }

    template<typename T>
    inline Vecnt<3,T> operator*( const Quatt<T> & q, const Vecnt<3,T> & v )
    {
      return( Matmnt<3,3,T>(q) * v );
    }

    template<typename T>
    inline Vecnt<3,T> operator*( const Vecnt<3,T> & v, const Quatt<T> & q )
    {
      return( v * Matmnt<3,3,T>(q) );
    }

    template<typename T>
    inline Quatt<T> operator/( const Quatt<T> & q0, const Quatt<T> & q1 )
    {
      return( q0 * ~q1 /*/ magnitude( q1 )*/ );
    }

  } // namespace math
} // namespace dp
