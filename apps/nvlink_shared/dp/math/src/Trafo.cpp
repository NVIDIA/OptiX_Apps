// Copyright (c) 2002-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/math/Trafo.h>


namespace dp
{
  namespace math
  {

    Trafo::Trafo( void )
    : m_center( Vec3f( 0.0f, 0.0f, 0.0f ) )
    , m_orientation( Quatf( 0.0f, 0.0f, 0.0f, 1.0f ) )
    , m_scaleOrientation( Quatf( 0.0f, 0.0f, 0.0f, 1.0f ) )
    , m_scaling( Vec3f( 1.0f, 1.0f, 1.0f ) )
    , m_translation( Vec3f( 0.0f, 0.0f, 0.0f ) )
    , m_matrixValid( false )
    , m_decompositionValid( true )
    {
    };

    Trafo::Trafo( const Trafo &rhs )
    : m_matrixValid( rhs.m_matrixValid )
    , m_decompositionValid( rhs.m_decompositionValid )
    {
      if ( m_matrixValid )
      {
        m_matrix = rhs.m_matrix;
      }
      if ( m_decompositionValid )
      {
        m_center            = rhs.m_center;
        m_orientation       = rhs.m_orientation;
        m_scaleOrientation  = rhs.m_scaleOrientation;
        m_scaling           = rhs.m_scaling;
        m_translation       = rhs.m_translation;
      }
    }

    Trafo & Trafo::operator=( const Trafo & rhs )
    {
      MY_ASSERT( rhs.m_matrixValid || rhs.m_decompositionValid );
      if (&rhs != this)
      {
        if ( rhs.m_decompositionValid )
        {
          m_center            = rhs.m_center;
          m_orientation       = rhs.m_orientation;
          m_scaleOrientation  = rhs.m_scaleOrientation;
          m_scaling           = rhs.m_scaling;
          m_translation       = rhs.m_translation;
        }
        if ( rhs.m_matrixValid )
        {
          m_matrix = rhs.m_matrix;
        }

        m_matrixValid = rhs.m_matrixValid;
        m_decompositionValid = rhs.m_decompositionValid;
      }
      return *this;
    }

    const Mat44f& Trafo::getMatrix( void ) const
    {
      MY_ASSERT( m_matrixValid || m_decompositionValid );

      if ( !m_matrixValid )
      {
        //  Calculates -C * SO^-1 * S * SO * R * C * T, with
        //    C     being the center translation
        //    SO    being the scale orientation
        //    S     being the scale
        //    R     being the rotation
        //    T     being the translation
        Mat33f soInv( -m_scaleOrientation );
        Mat44f m0( Vec4f( soInv[0], 0.0f )
                 , Vec4f( soInv[1], 0.0f )
                 , Vec4f( soInv[2], 0.0f )
                 , Vec4f( -m_center*soInv, 1.0f ) );
        Mat33f rot( m_scaleOrientation * m_orientation );
        Mat44f m1( Vec4f( m_scaling[0] * rot[0], 0.0f )
                 , Vec4f( m_scaling[1] * rot[1], 0.0f )
                 , Vec4f( m_scaling[2] * rot[2], 0.0f )
                 , Vec4f( m_center + m_translation, 1.0f ) );
        m_matrix = m0 * m1;
        m_matrixValid = true;
      }
      return( m_matrix );
    }

    Mat44f Trafo::getInverse( void ) const
    {
      dp::math::Mat44f inverse = getMatrix(); // Automatically makes the matrix valid.
      if ( !inverse.invert() )
      {
        // Inverting the matrix directly failed.
        // Try the more robust but also more expensive decomposition.
        // (Note that this still barely handles a uniform scaling of 2.2723e-6,
        //  while the direct matrix inversion already dropped below the
        //  std::numeric_limits<double>::epsilon() of 2.2204460492503131e-016 for the determinant.)
        if ( !m_decompositionValid )
        {
          decompose();
        }
        //  Calculates T^-1 * C^-1 * R^-1 * SO^-1 * S^-1 * SO * -C^-1, with
        //    C     being the center translation
        //    SO    being the scale orientation
        //    S     being the scale
        //    R     being the rotation
        //    T     being the translation
        Mat33f rot( -m_orientation * -m_scaleOrientation );
        Mat44f m0;
        m0[0] = Vec4f( rot[0], 0.0f );
        m0[1] = Vec4f( rot[1], 0.0f );
        m0[2] = Vec4f( rot[2], 0.0f );
        m0[3] = Vec4f( ( -m_center - m_translation ) * rot, 1.0f );

        Mat33f so( m_scaleOrientation );

        Mat44f m1;
        m1[0] = Vec4f( so[0], 0.0f ) / m_scaling[0];
        m1[1] = Vec4f( so[1], 0.0f ) / m_scaling[1];
        m1[2] = Vec4f( so[2], 0.0f ) / m_scaling[2];
        m1[3] = Vec4f( m_center, 1.0f );

        inverse = m0 * m1;
      }
      return inverse;
    }

    void Trafo::setIdentity( void )
    {
      m_center = Vec3f( 0.0f, 0.0f, 0.0f );
      m_orientation = Quatf( 0.0f, 0.0f, 0.0f, 1.0f );
      m_scaling = Vec3f( 1.0f, 1.0f, 1.0f );
      m_scaleOrientation = Quatf( 0.0f, 0.0f, 0.0f, 1.0f );
      m_translation = Vec3f( 0.0f, 0.0f, 0.0f );
      m_matrixValid = false;
      m_decompositionValid = true;
    }

    void Trafo::setMatrix( const Mat44f &matrix )
    {
      m_matrix = matrix;
      m_matrixValid = true;
      m_decompositionValid = false;
    }

    void Trafo::decompose() const
    {
      MY_ASSERT( m_matrixValid );
      MY_ASSERT( m_decompositionValid == false );

      dp::math::decompose( m_matrix, m_translation, m_orientation, m_scaling, m_scaleOrientation);

      m_decompositionValid = true;
    }


    bool Trafo::operator==( const Trafo &t ) const
    {
      if ( m_decompositionValid && t.m_decompositionValid )
      {
        return(
              ( getCenter()           == t.getCenter()            )
          &&  ( getOrientation()      == t.getOrientation()       )
          &&  ( getScaling()          == t.getScaling()           )
          &&  ( getScaleOrientation() == t.getScaleOrientation()  )
          &&  ( getTranslation()      == t.getTranslation()       ) );
      }
      else
      {
        return getMatrix() == t.getMatrix();
      }

    }

    Trafo lerp( float alpha, const Trafo &t0, const Trafo &t1 )
    {
      Trafo t;
      t.setCenter( lerp( alpha, t0.getCenter(), t1.getCenter() ) );
      t.setOrientation( lerp( alpha, t0.getOrientation(), t1.getOrientation() ) );
      t.setScaling( lerp( alpha, t0.getScaling(), t1.getScaling() ) );
      t.setScaleOrientation( lerp( alpha, t0.getScaleOrientation(), t1.getScaleOrientation() ) );
      t.setTranslation( lerp( alpha, t0.getTranslation(), t1.getTranslation() ) );
      return( t );
    }

  } // namespace math
} // namespace dp
