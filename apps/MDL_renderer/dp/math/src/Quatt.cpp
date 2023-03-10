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


#include <dp/math/Quatt.h>

namespace dp
{
  namespace math
  {
    template<typename T>
    Quatt<T> _lerp( T alpha, const Quatt<T> & q0, const Quatt<T> & q1 )
    {
      //  cosine theta = dot product of q0 and q1
      T cosTheta = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];

      //  if q1 is on opposite hemisphere from q0, use -q1 instead
      bool flip = ( cosTheta < 0 );
      if ( flip )
      {
        cosTheta = - cosTheta;
      }

      T beta;
      if ( 1 - cosTheta < std::numeric_limits<T>::epsilon() )
      {
        //  if q1 is (within precision limits) the same as q0, just linear interpolate between q0 and q1.
        beta = 1 - alpha;
      }
      else
      {
        //  normal case
        T theta = acos( cosTheta );
        T oneOverSinTheta = 1 / sin( theta );
        beta = sin( theta * ( 1 - alpha ) ) * oneOverSinTheta;
        alpha = sin( theta * alpha ) * oneOverSinTheta;
      }

      if ( flip )
      {
        alpha = - alpha;
      }

      return( Quatt<T>( beta * q0[0] + alpha * q1[0]
                      , beta * q0[1] + alpha * q1[1]
                      , beta * q0[2] + alpha * q1[2]
                      , beta * q0[3] + alpha * q1[3] ) );
    }

    Quatt<float> lerp( float alpha, const Quatt<float> & q0, const Quatt<float> & q1 )
    {
      return( _lerp( alpha, q0, q1 ) );
    }

    void lerp( float alpha, const Quatt<float> & q0, const Quatt<float> & q1, Quatt<float> &qr )
    {
      qr = _lerp( alpha, q0, q1 );
    }

  } // namespace math
} // namespace dp
