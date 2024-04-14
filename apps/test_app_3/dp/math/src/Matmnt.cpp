// Copyright (c) 2012-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <dp/math/Matmnt.h>

using std::abs;

namespace dp
{
  namespace math
  {
    const Mat44f cIdentity44f( { 1.0f, 0.0f, 0.0f, 0.0f
                               , 0.0f, 1.0f, 0.0f, 0.0f
                               , 0.0f, 0.0f, 1.0f, 0.0f
                               , 0.0f, 0.0f, 0.0f, 1.0f } );

    template<unsigned int n, typename T>
    static T _colNorm( const Matmnt<n,n,T> &mat )
    {
      T max(0);
      for ( unsigned int i=0 ; i<n ; ++i )
      {
        T sum(0);
        for ( unsigned int j=0; j<n ; ++j )
        {
          sum += abs( mat[j][i] );
        }
        if ( max < sum )
        {
          max = sum;
        }
      }
      return( max );
    }

    template<unsigned int n, typename T>
    static T _rowNorm( const Matmnt<n,n,T> &mat )
    {
      T max(0);
      for ( unsigned int i=0 ; i<n ; ++i )
      {
        T sum(0);
        for ( unsigned int j=0 ; j<n ; ++ j )
        {
          sum += abs( mat[i][j] );
        }
        if ( max < sum )
        {
          max = sum;
        }
      }
      return( max );
    }

    template<unsigned int n, typename T>
    static int _maxColumn( const Matmnt<n,n,T> &mat )
    {
      T max(0);
      int col(-1);
      for ( unsigned int i=0 ; i<n ; ++i )
      {
        for ( unsigned int j=0 ; j<n ; ++j )
        {
          T tmp = mat[i][j];
          if ( tmp < 0 )
          {
            tmp = - tmp;
          }
          if ( max < tmp )
          {
            max = tmp;
            col = j;
          }
        }
      }
      return( col );
    }

    //  Make v for Householder reflection to zero all v components but first
    template<typename T>
    static void _makeReflector( Vecnt<3,T> &v )
    {
      T s = sqrt( v * v );
      v[2] += ( v[2] < 0 ) ? -s : s;
      s = sqrt( T(2) / ( v * v ) );
      v *= s;
    }

    //  Apply Householder reflection represented by u to column vectors of M
    template<unsigned int n, typename T>
    static void _reflectCols( Matmnt<n,n,T> &M, const Vecnt<n,T> &u )
    {
      Vecnt<n,T> v = u * M;
      for ( unsigned int i=0 ; i<n ; ++i )
      {
        for ( unsigned int j=0 ; j<n ; ++j )
        {
          M[j][i] -= u[j] * v[i];
        }
      }
    }

    //  Apply Householder reflection represented by u to row vectors of M
    template<unsigned int n, typename T>
    static void _reflectRows( Matmnt<n,n,T> &M, const Vecnt<n,T> &u )
    {
      Vecnt<n,T> v = M * u;
      for ( unsigned int i=0 ; i<n ; ++i )
      {
        for ( unsigned int j=0 ; j<n ; ++j )
        {
          M[i][j] -= u[j] * v[i];
        }
      }
    }

    //  Find orthogonal factor Q of rank 1 (or less) of mk -> write to mk
    template<typename T>
    static void _decomposeRank1( Matmnt<3,3,T> & mk )
    {
      // if rank(mk) is 1 we should find a non-zero column in M
      int col = _maxColumn( mk );
      if ( col < 0 )
      {
        //  rank 0
        setIdentity( mk );
      }
      else
      {
        Vecnt<3,T> v1( mk[0][col], mk[1][col], mk[2][col] );
        _makeReflector( v1 );
        _reflectCols( mk, v1 );
        Vecnt<3,T> v2( mk[2][0], mk[2][1], mk[2][2] );
        _makeReflector( v2 );
        _reflectRows( mk, v2 );
        T s = mk[2][2];
        setIdentity( mk );
        if ( s < 0 )
        {
          mk[2][2] = T(-1);
        }
        _reflectCols( mk, v1 );
        _reflectRows( mk, v2 );
      }
    }

    //  Find orthogonal factor Q or rank 2 (or less) of mk using adjoint transpose -> write to mk
    template<typename T>
    static void _decomposeRank2( Matmnt<3,3,T> & mk, const Matmnt<3,3,T> mAdjTk )
    {
      //  if rank(mk) is 2, we should find a non-zero column in mAdjTk
      int col = _maxColumn( mAdjTk );
      if ( col < 0 )
      {
        //  rank 1
        _decomposeRank1( mk );
      }
      else
      {
        Vecnt<3,T> v1( mAdjTk[0][col], mAdjTk[1][col], mAdjTk[2][col] );
        _makeReflector( v1 );
        _reflectCols( mk, v1 );
        Vecnt<3,T> v2 = mk[0] ^ mk[1];
        _makeReflector( v2 );
        _reflectRows( mk, v2 );
        if ( mk[0][1] * mk[1][0] < mk[0][0] * mk[1][1] )
        {
          T c = mk[1][1] + mk[0][0];
          T s = mk[1][0] - mk[0][1];
          T d = sqrt( c * c + s * s );
          c /= d;
          s /= d;
          mk[0][0] = c;
          mk[0][1] = -s;
          mk[1][0] = s;
          mk[1][1] = c;
        }
        else
        {
          T c = mk[1][1] - mk[0][0];
          T s = mk[1][0] + mk[0][1];
          T d = sqrt( c * c + s * s );
          c /= d;
          s /= d;
          mk[0][0] = -c;
          mk[0][1] = s;
          mk[1][0] = s;
          mk[1][1] = c;
        }
        mk[0][2] = T(0);
        mk[1][2] = T(0);
        mk[2][0] = T(0);
        mk[2][1] = T(0);
        mk[2][2] = T(1);
        _reflectCols( mk, v1 );
        _reflectRows( mk, v2 );
      }
    }

    template<typename T>
    static T _polarDecomposition( const Matmnt<3,3,T> &mat, Matmnt<3,3,T> &rot, Matmnt<3,3,T> &sca )
    {
      Matmnt<3,3,T> mk = ~mat;
      T det;
      T eCol;
      T mCol = _colNorm( mk );
      T mRow = _rowNorm( mk );
      do
      {
        Matmnt<3, 3, T> mAdjTk{ mk[1] ^ mk[2], mk[2] ^ mk[0], mk[0] ^ mk[1] };
        det = mk[0] * mAdjTk[0];
        T absDet = abs( det );
        if ( std::numeric_limits<T>::epsilon() < absDet )
        {
          T mAdjTCol = _colNorm( mAdjTk );
          T mAdjTRow = _rowNorm( mAdjTk );
          T gamma = sqrt( sqrt( ( mAdjTCol * mAdjTRow ) / ( mCol * mRow ) ) / absDet );
          Matmnt<3,3,T> ek = mk;
          mk = T(0.5) * ( gamma * mk + mAdjTk / ( gamma * det ) );
          ek -= mk;
          eCol = _colNorm( ek );
          mCol = _colNorm( mk );
          mRow = _rowNorm( mk );
        }
        else
        {
          //  rank 2
          _decomposeRank2( mk, mAdjTk );
          break;
        }
      } while( ( 2 * mCol * std::numeric_limits<T>::epsilon() ) < eCol );
      if ( det < T(0) )
      {
        mk = - mk;
      }
      rot = ~mk;

      sca = mat * mk;
      for ( unsigned int i=0 ; i<3 ; ++i )
      {
        for ( unsigned int j=i+1 ; j<3 ; ++j )
        {
          sca[i][j] = sca[j][i] = T(0.5) * ( sca[i][j] + sca[j][i] );
        }
      }
      return( det );
    }

    template<typename T>
    static Vecnt<3,T> _spectralDecomposition( const Matmnt<3,3,T> &mat, Matmnt<3,3,T> &u )
    {
      setIdentity( u );
      Vecnt<3,T> diag( mat[0][0], mat[1][1], mat[2][2] );
      Vecnt<3,T> offd( mat[2][1], mat[0][2], mat[1][0] );
      bool done = false;
      for ( int sweep=20 ; 0<sweep && ! done ; sweep-- )
      {
        T sm = abs(offd[0]) + abs(offd[1]) + abs(offd[2]);
        done = ( sm < std::numeric_limits<T>::epsilon() );
        if ( !done )
        {
          done = true;
          for ( int i=2 ; 0<=i ; i-- )
          {
            int p = (i+1)%3;
            int q = (p+1)%3;
            T absOffDi = abs(offd[i]);
            if ( std::numeric_limits<T>::epsilon() < absOffDi )
            {
              done = false;
              T g = 100 * absOffDi;
              T h = diag[q] - diag[p];
              T absh = abs( h );
              T t;
              if ( absh + g == absh )
              {
                t = offd[i] / h;
              }
              else
              {
                T theta = T(0.5) * h / offd[i];
                t = 1 / ( abs(theta) + sqrt(theta*theta+1) );
                if ( theta < 0 )
                {
                  t = -t;
                }
              }
              T c = 1 / sqrt( t*t+1);
              T s = t * c;
              T tau = s / (c + 1);
              T ta = t * offd[i];
              offd[i] = 0;
              diag[p] -= ta;
              diag[q] += ta;
              T offdq = offd[q];
              offd[q] -= s * ( offd[p] + tau * offd[q] );
              offd[p] += s * ( offdq   - tau * offd[p] );
              for ( int j=2 ; 0<=j ; j-- )
              {
                T a = u[p][j];
                T b = u[q][j];
                u[p][j] -= s * ( b + tau * a );
                u[q][j] += s * ( a - tau * b );
              }
            }
          }
        }
      }
      return( diag );
    }

    void decompose( const Matmnt<3,3,float> &mat, Quatt<float> &orientation, Vecnt<3,float> &scaling
                  , Quatt<float> &scaleOrientation )
    {
#if 0

      Matmnt<3,3,float> rot, sca;
      float det = _polarDecomposition( mat, rot, sca );
#if !defined(NDEBUG)
      {
        Matmnt<3,3,float> diff = mat - (float)sign(det) * sca * rot;
        float eps = std::max( 1.0f, maxElement( sca ) ) * std::numeric_limits<float>::epsilon();
      }
#endif

      orientation = Quatt<float>(rot);

      //  get the scaling out of sca
      Matmnt<3,3,float> so;
      scaling = (float)sign(det) * _spectralDecomposition( sca, so );
#if !defined( NDEBUG )
      {
        Matmnt<3,3,float> k( scaling[0], 0, 0,  0, scaling[1], 0,  0, 0, scaling[2] );
        Matmnt<3,3,float> diff = sca - ~so * k * so;
        float eps = std::max( 1.0f, maxElement( scaling ) ) * std::numeric_limits<float>::epsilon();
      }
#endif

      scaleOrientation = Quatt<float>( so );

#if !defined( NDEBUG )
      {
        Matmnt<3,3,float> ms( scaling[0], 0, 0, 0, scaling[1], 0, 0, 0, scaling[2] );
        Matmnt<3,3,float> mso( so );
        Matmnt<3,3,float> diff = mat - ~mso * ms * mso * rot;
        float eps = std::max( 1.0f, maxElement( scaling ) ) * std::numeric_limits<float>::epsilon();
      }
#endif

#else

      Matmnt<3,3,double> rot, sca;
      double det = _polarDecomposition( Matmnt<3,3,double>(mat), rot, sca );
#if !defined(NDEBUG)
      {
        Matmnt<3,3,double> diff = Matmnt<3,3,double>(mat) - sca * rot;
        double eps = std::max( 1.0, maxElement( sca ) ) * std::numeric_limits<float>::epsilon();
      }
#endif

      orientation = Quatt<float>(Quatt<double>(rot));

      //  get the scaling out of sca
      Matmnt<3,3,double> so;
      scaling = _spectralDecomposition( sca, so );
#if !defined( NDEBUG )
      {
        Matmnt<3, 3, double> k( { (double)scaling[0], 0.0, 0.0, 0.0, (double)scaling[1], 0.0, 0.0, 0.0, (double)scaling[2] } );
        Matmnt<3,3,double> diff = sca - ~so * k * so;
        double eps = std::max( 1.0f, maxElement( scaling ) ) * std::numeric_limits<float>::epsilon();
        int a = 0;
      }
#endif

      scaleOrientation = Quatt<float>(Quatt<double>( so ));

#if !defined( NDEBUG )
      Matmnt<3, 3, double> ms( { (double)scaling[0], 0.0, 0.0, 0.0, (double)scaling[1], 0.0, 0.0, 0.0, (double)scaling[2] } );
      Matmnt<3,3,double> mso( so );
      Matmnt<3,3,float> diff = mat - Matmnt<3,3,float>(~mso * ms * mso * rot);
      float eps = std::max( 1.0f, maxElement( scaling ) ) * std::numeric_limits<float>::epsilon();
#endif

#endif
    }

  } // namespace math
} // namespace dp
