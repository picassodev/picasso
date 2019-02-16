#ifndef HARLOW_DENSELINEARALGEBRA_HPP
#define HARLOW_DENSELINEARALGEBRA_HPP

#include <Kokkos_Core.hpp>
#include <cmath>
#include <cassert>

namespace Harlow
{
namespace DenseLinearAlgebra
{
//---------------------------------------------------------------------------//
// Compute the determinant of a 3x3 matrix.
template<class Real>
KOKKOS_INLINE_FUNCTION
Real determinant( const Real m[3][3] )
{
    return
        m[0][0] * m[1][1] * m[2][2] +
        m[0][1] * m[1][2] * m[2][0] +
        m[0][2] * m[1][0] * m[2][1] -
        m[0][2] * m[1][1] * m[2][0] -
        m[0][1] * m[1][0] * m[2][2] -
        m[0][0] * m[1][2] * m[2][1];
}

//---------------------------------------------------------------------------//
// Compute the inverse of a 3x3 matrix with a precomputed determinant.
template<class Real>
KOKKOS_INLINE_FUNCTION
void inverse( const Real m[3][3], Real det_m, Real m_inv[3][3] )
{
    Real det_m_inv = 1.0 / det_m;

    m_inv[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) * det_m_inv;
    m_inv[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) * det_m_inv;
    m_inv[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) * det_m_inv;

    m_inv[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) * det_m_inv;
    m_inv[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) * det_m_inv;
    m_inv[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) * det_m_inv;

    m_inv[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) * det_m_inv;
    m_inv[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) * det_m_inv;
    m_inv[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) * det_m_inv;
}

//---------------------------------------------------------------------------//
// Compute the inverse of a 3x3 matrix.
template<class Real>
KOKKOS_INLINE_FUNCTION
void inverse( const Real m[3][3], Real m_inv[3][3] )
{
    Real det_m = determinant( m );
    inverse( m, det_m, m_inv );
}

//---------------------------------------------------------------------------//
// Matrix vector multiply. A*x = y
template<class Real>
KOKKOS_INLINE_FUNCTION
void multiply( const Real a[3][3], const Real x[3], Real y[3] )
{
    for ( int i = 0; i < 3; ++i )
    {
        y[i] = 0.0;
        for ( int j = 0; j < 3; ++j )
            y[i] += a[i][j] * x[j];
    }
}

//---------------------------------------------------------------------------//
// Matrix Matrix  multiply. A*B = C
template<class Real>
KOKKOS_INLINE_FUNCTION
void multiply_AB( const Real a[3][3], const Real b[3][3], Real c[3][3] )
{
   for ( int i = 0; i < 3; ++i )
   {
        for ( int j = 0; j < 3; ++j )
        {
            c[i][j] = 0.0;    
            for ( int k = 0; k < 3; ++k )
                c[i][j] += a[i][k] * b[k][j];
        }
   }
}

//---------------------------------------------------------------------------//
// Transpose Matrix A^{T}
template<class Real>
KOKKOS_INLINE_FUNCTION
void transpose( const Real a[3][3], Real transpose_a[3][3] )
{
   for ( int i = 0; i < 3; ++i )
   {
       for ( int j = 0; j < 3; ++j )
           transpose_a[i][j] = a[j][i];
   }
}

//---------------------------------------------------------------------------//
// s: 3 eignenvalues returned by A[0][0], A[1][1], A[2][2] in the end
// x: 3 eigenvectors
// Jacobi Method is applied with 1.oe-10 for theta convergence
template<class Real>
KOKKOS_INLINE_FUNCTION
void eigen( const Real a[3][3], Real s[3], Real X[3][3] )
{
   // pi
   Real pi = atan(1.0)*4.0;
   
   Real A[3][3];
   // intialize X with I and copy a into A
   for(int i=0; i<3; i++)
   {
      for(int j=0; j<3; j++)
      {
         X[i][j] = ( i == j) ? 1.0 : 0.0;
         A[i][j] = a[i][j];
      }
   }

   Real theta;
   Real R[3][3];
   Real RT[3][3];   // R^T
   Real RTA[3][3];  // R^T * A
   Real XR[3][3];   // X*R
   // iterate until theta < 1.0e-10
   do{
        // find the biggest values among  A_ij except diagonal element 
        // record the index i,j into r,c
        Real temp_big = fabs(A[0][1]);
        int r = 0;  // row
        int c = 1;  // column
        for(int i=0; i<3; i++)
        {
           for(int j=1; j<3; j++)
           {
              if(i == j) continue;
              if( temp_big < abs( A[i][j] ) )
              {
                 temp_big =  abs( A[i][j] );
                 r = i;
                 c = j;
              }
           }
        }
       
        // initial Rotational Matrix R = I
        for(int i=0; i<3; i++)
        {
           for(int j=0; j<3; j++)
              R[i][j] = ( i == j) ? 1.0 : 0.0;
        }
 
        // determine angle theta that make the A_rc  = 0
        theta = (A[r][r] == A[c][c]) ? pi/4.0 : atan( 2.0*A[r][c]/(A[r][r]-A[c][c]) )/2.0;

        // construct Rotational Matrix R by replacing component rr, rc, cr, cc only
        R[r][r] = cos(theta);
        R[r][c] = -sin(theta);
        R[c][r] = sin(theta);
        R[c][c] = cos(theta);
        
        // rotate by computing R^(T) * A *  R
        transpose( R, RT);
        multiply_AB( RT, A, RTA );
        multiply_AB( RTA, R, A );
 
        // calculate X*R and store it into X again for next iteration
        multiply_AB( X, R, XR);
        for(int i=0; i<3; i++)
        {
           for(int j=0; j<3; j++)
              X[i][j] = XR[i][j];
        }
 
    } while(fabs(theta) >= 1.0e-10);

   // Descending order for eigenvalue and corresponding eigenvector
   Real temp;
   Real temp_vec[3];
   
   if(A[0][0] < A[1][1])
   {
      temp    = A[0][0];
      A[0][0] = A[1][1];
      A[1][1] = temp;
      
      for(int i=0; i<3; i++)
      {
         temp_vec[i] = X[i][0];
         X[i][0]     = X[i][1];
         X[i][1]     = temp_vec[i];
      }
   }
  
   if(A[1][1] < A[2][2])
   {
      temp    = A[1][1];
      A[1][1] = A[2][2];
      A[2][2] = temp;
      
      for(int i=0; i<3; i++)
      {
         temp_vec[i] = X[i][1];
         X[i][1]     = X[i][2];
         X[i][2]     = temp_vec[i];
      }
   }

   if(A[0][0] < A[1][1])
   {
      temp    = A[0][0];
      A[0][0] = A[1][1];
      A[1][1] = temp;
      
      for(int i=0; i<3; i++)
      {
         temp_vec[i] = X[i][0];
         X[i][0]     = X[i][1];
         X[i][1]     = temp_vec[i];
      }
   }
  
   // return s, X
   for(int i=0; i<3; i++)
      s[i] = A[i][i];

}


//---------------------------------------------------------------------------//
// 3 by 3 matrix SVD
template<class Real>
KOKKOS_INLINE_FUNCTION
void svd( const Real A[3][3], Real U[3][3], Real S[3], Real V[3][3])
{
   // if matrix A is singular, throw error and stop simulation
   Real det_A = determinant(A);
   if( fabs(det_A) == 0.0 )
   {
       printf("Error, deformation gradient matrix cannot be sigular\n");
       assert(1);
   }
   
   // A^T
   Real AT[3][3];
   transpose(A, AT);

   // calculate A^T * A;
   Real ATA[3][3];
   multiply_AB( AT, A, ATA);

   // eigenvalue matrix S and eigenvector matrix  V from A = A^T * A
   Real eigen_value[3];
   eigen( ATA, eigen_value, V);
   
   for(int i=0; i<3; i++)
   {
         S[i]= sqrt(eigen_value[i]);
   }

   // calculate U from U = a*V*inv(S);
   Real AV[3][3];
   multiply_AB(A, V, AV);
 
   for(int i=0; i<3; i++)
   {
      for(int j=0; j<3; j++)
         U[i][j] = AV[i][j]/S[j];
   } 
}   
   

//---------------------------------------------------------------------------//

} // end namespace DenseLinearAlgebra
} // end namespace Harlow

#endif // end HARLOW_DENSELINEARALGEBRA_HPP
