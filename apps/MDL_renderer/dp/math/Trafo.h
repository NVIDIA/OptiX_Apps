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

#include <dp/math/Quatt.h>
#include <dp/math/Vecnt.h>

namespace dp
{
  namespace math
  {

    //! Transformation class.
    /** This class is used to ease transformation handling. It has an interface to rotate, scale,
      * and translate and can produce a \c Mat44f that combines them.  */
    class Trafo
    {
      public:
        //! Constructor: initialized to identity
        DP_MATH_API Trafo( void );

        //! Copy Constructor
        DP_MATH_API Trafo( const Trafo &rhs ); //!< Trafo to copy

        //! Get the center of rotation of this transformation.
        /** \returns  \c Vec3f that describes the center or rotation. */
        DP_MATH_API const Vec3f & getCenter( void ) const;

        //! Get the rotational part of this transformation.
        /** \returns  \c Quatf that describes the rotational part */
        DP_MATH_API const Quatf & getOrientation( void ) const;

        //! Get the scale orientation part of this transform.
        /** \return \c Quatf that describes the scale orientational part */
        DP_MATH_API const Quatf & getScaleOrientation( void ) const;

        //! Get the scaling part of this transformation.
        /** \returns  \c Vec3f that describes the scaling part  */
        DP_MATH_API const Vec3f & getScaling( void ) const;

        //! Get the translational part of this transformation.
        /** \returns  \c Vec3f that describes the translational part  */
        DP_MATH_API const Vec3f & getTranslation( void ) const;

        /*! \brief Get the current transformation.
         *  \return The \c Mat44f that describes the transformation.
         *  \remarks The transformation is the concatenation of the center translation C, scale
         *  orientation SO, scaling S, rotation R, and translation T, by the following formula:
         *  \code
         *    M = -C * SO^-1 * S * SO * R * C * T
         *  \endcode
         *  \sa getInverse */
        DP_MATH_API const Mat44f& getMatrix( void ) const;

        /*! \brief Get the current inverse transformation.
         *  \return The \c Mat44f that describes the inverse transformation.
         *  \remarks The inverse transformation is the concatenation of the center translation C,
         *  scale orientation SO, scaling S, rotation R, and translation T, by the following
         *  formula:
         *  \code
         *    M = T^-1 * C^-1 * R^-1 * SO^-1 * S^-1 * SO * -C^-1
         *  \endcode
         *  \sa getMatrix */
        DP_MATH_API Mat44f getInverse( void ) const;

        //! Set the center of ration of the transformation.
        DP_MATH_API void setCenter( const Vec3f &center              //!<  center of rotation
                               );

        //! Set the \c Trafo to identity.
        DP_MATH_API void setIdentity( void );

        //! Set the rotational part of the transformation, using a quaternion.
        DP_MATH_API void setOrientation( const Quatf &orientation    //!<  rotational part of transformation
                                    );

        //! Set the scale orientational part of the transformation.
        DP_MATH_API void setScaleOrientation( const Quatf &scaleOrientation  //!< scale orientational part of transform
                                         );

        //! Set the scaling part of the transformation.
        DP_MATH_API void setScaling( const Vec3f &scaling            //!<  scaling part of transformation
                                );

        //! Set the translational part of the transformation.
        DP_MATH_API void setTranslation( const Vec3f &translation    //!<  translational part of transformation
                                    );

        //! Set the complete transformation by a Mat44f.
        /** The matrix is internally decomposed into a translation, rotation, scaleOrientation, and scaling. */
        DP_MATH_API void setMatrix( const Mat44f &matrix   //!< complete transformation
                               );

        //! Copy operator.
        DP_MATH_API Trafo & operator=( const Trafo &rhs //!< Trafo to copy
                         );

        //! Equality operator.
        /** \returns \c true if \c this is equal to \a t, otherwise \c false  */
        DP_MATH_API bool operator==( const Trafo &t    //!<  \c Trafo to compare with
                                ) const;

        DP_MATH_API bool operator!=( const Trafo &t ) const;

      private:
        DP_MATH_API void decompose() const;

        mutable Mat44f  m_matrix;
        mutable Vec3f   m_center;             //!< center of rotation
        mutable Quatf   m_orientation;        //!< orientational part of the transformation
        mutable Quatf   m_scaleOrientation;   //!< scale orientation
        mutable Vec3f   m_scaling;            //!< scaling part of the transformation
        mutable Vec3f   m_translation;        //!< translational part of the transformation

        mutable bool    m_matrixValid;
        mutable bool    m_decompositionValid;
    };

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // non-member functions
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    /*! \relates dp::math::Trafo
     *  This calculates the linear interpolation \code ( 1 - alpha ) * t0 + alpha * t1 \endcode */
    DP_MATH_API Trafo lerp( float alpha      //!<  interpolation parameter
                       , const Trafo &t0  //!<  starting value
                       , const Trafo &t1  //!<  ending value
                       );

    DP_MATH_API void lerp( float alpha, const Trafo & t0, const Trafo & t1, Trafo & tr );

    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    // inlines
    // - - - - -  - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    inline const Vec3f & Trafo::getCenter( void ) const
    {
      if ( !m_decompositionValid )
      {
        decompose();
      }
      return( m_center );
    }

    inline const Quatf & Trafo::getOrientation( void ) const
    {
      if ( !m_decompositionValid )
      {
        decompose();
      }
      return( m_orientation );
    }

    inline const Quatf & Trafo::getScaleOrientation( void ) const
    {
      if ( !m_decompositionValid )
      {
        decompose();
      }
      return( m_scaleOrientation );
    }

    inline const Vec3f & Trafo::getScaling( void ) const
    {
      if ( !m_decompositionValid )
      {
        decompose();
      }
      return( m_scaling );
    }

    inline const Vec3f & Trafo::getTranslation( void ) const
    {
      if ( !m_decompositionValid )
      {
        decompose();
      }
      return( m_translation );
    }

    inline void Trafo::setCenter( const Vec3f &center )
    {
      if ( !m_decompositionValid )
      {
        decompose();
      }
      m_matrixValid = false;
      m_center = center;
    }

    inline void Trafo::setOrientation( const Quatf &orientation )
    {
      if ( !m_decompositionValid )
      {
        decompose();
      }
      m_matrixValid = false;
      m_orientation = orientation;
    }

    inline void Trafo::setScaleOrientation( const Quatf &scaleOrientation )
    {
      if ( !m_decompositionValid )
      {
        decompose();
      }
      m_matrixValid = false;
      m_scaleOrientation = scaleOrientation;
    }

    inline void Trafo::setScaling( const Vec3f &scaling )
    {
      if ( !m_decompositionValid )
      {
        decompose();
      }
      m_matrixValid = false;
      m_scaling = scaling;
    }

    inline void Trafo::setTranslation( const Vec3f &translation )
    {
      if ( !m_decompositionValid )
      {
        decompose();
      }
      m_matrixValid = false;
      m_translation = translation;
    }

    inline bool Trafo::operator!=( const Trafo & t ) const
    {
      return( ! ( *this == t ) );
    }

    inline void lerp( float alpha, const Trafo & t0, const Trafo & t1, Trafo & tr )
    {
      tr = lerp( alpha, t0, t1 );
    }

  } // namespace math
} // namespace dp
