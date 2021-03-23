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


#include <dp/math/math.h>

#include <limits.h>


namespace dp
{
  namespace math
  {

    DP_MATH_API float _atof( const std::string &str )
    {
      int pre = 0;
      int post = 0;
      float divisor = 1.0f;
      bool negative = false;
      size_t i = 0;
      while ( ( i < str.length() ) && ( ( str[i] == ' ' ) || ( str[i] == '\t' ) ) )
      {
        i++;
      }
      if ( ( i < str.length() ) && ( ( str[i] == '+' ) || ( str[i] == '-' ) ) )
      {
        negative = ( str[i] == '-' );
        i++;
      }
      while ( ( i < str.length() ) && isdigit( str[i] ) )
      {
        pre = pre * 10 + ( str[i] - 0x30 );
        i++;
      }
      if ( ( i < str.length() ) && ( str[i] == '.' ) )
      {
        i++;
        while ( ( i < str.length() ) && isdigit( str[i] ) && ( post <= INT_MAX/10-9 ) )
        {
          post = post * 10 + ( str[i] - 0x30 );
          divisor *= 10;
          i++;
        }
        while ( ( i < str.length() ) && isdigit( str[i] ) )
        {
          i++;
        }
      }
      if ( negative )
      {
        pre = - pre;
        post = - post;
      }
      float f = post ? pre + post / divisor : (float)pre;
      if ( ( i < str.length() ) && ( ( str[i] == 'd' ) || ( str[i] == 'D' ) || ( str[i] == 'e' ) || ( str[i] == 'E' ) ) )
      {
        i++;
        negative = false;
        if ( ( i < str.length() ) && ( ( str[i] == '+' ) || ( str[i] == '-' ) ) )
        {
          negative = ( str[i] == '-' );
          i++;
        }
        int exponent = 0;
        while ( ( i < str.length() ) && isdigit( str[i] ) )
        {
          exponent = exponent * 10 + ( str[i] - 0x30 );
          i++;
        }
        if ( negative )
        {
          exponent = - exponent;
        }
        f *= pow( 10.0f, float(exponent) );
      }
    #if !defined( NDEBUG )
      float z = (float)atof( str.c_str() );
      MY_ASSERT( fabsf( f - z ) < FLT_EPSILON * std::max( 1.0f, fabsf( z ) ) );
    #endif
      return( f );
    }

  } // namespace math
} // namespace dp
