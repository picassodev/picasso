#ifndef HARLOW_DENSELINEARALGEBRA_HPP
#define HARLOW_DENSELINEARALGEBRA_HPP

#include <Kokkos_Core.hpp>

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

} // end namespace DenseLinearAlgebra
} // end namespace Harlow

#endif // end HARLOW_DENSELINEARALGEBRA_HPP
