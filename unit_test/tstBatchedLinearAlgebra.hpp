/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Picasso_BatchedLinearAlgebra.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <gtest/gtest.h>

#include <random>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
void matrixTest()
{
    // Check a basic matrix.
    LinearAlgebra::Matrix<double, 2, 3> a = { { 1.2, -3.5, 5.4 },
                                              { 8.6, 2.6, -0.1 } };
    EXPECT_EQ( a.stride_0(), 3 );
    EXPECT_EQ( a.stride_1(), 1 );
    EXPECT_EQ( a.stride( 0 ), 3 );
    EXPECT_EQ( a.stride( 1 ), 1 );
    EXPECT_EQ( a.extent( 0 ), 2 );
    EXPECT_EQ( a.extent( 1 ), 3 );

    EXPECT_EQ( a( 0, 0 ), 1.2 );
    EXPECT_EQ( a( 0, 1 ), -3.5 );
    EXPECT_EQ( a( 0, 2 ), 5.4 );
    EXPECT_EQ( a( 1, 0 ), 8.6 );
    EXPECT_EQ( a( 1, 1 ), 2.6 );
    EXPECT_EQ( a( 1, 2 ), -0.1 );

    // Check rows.
    for ( int i = 0; i < 2; ++i )
    {
        auto row = a.row( i );
        EXPECT_EQ( row.stride_0(), 1 );
        EXPECT_EQ( row.stride( 0 ), 1 );
        EXPECT_EQ( row.extent( 0 ), 3 );
        for ( int j = 0; j < 3; ++j )
            EXPECT_EQ( row( j ), a( i, j ) );
    }

    // Check columns.
    for ( int j = 0; j < 3; ++j )
    {
        auto column = a.column( j );
        EXPECT_EQ( column.stride_0(), 3 );
        EXPECT_EQ( column.stride( 0 ), 3 );
        EXPECT_EQ( column.extent( 0 ), 2 );
        for ( int i = 0; i < 2; ++i )
            EXPECT_EQ( column( i ), a( i, j ) );
    }

    // Check matrix view.
    LinearAlgebra::MatrixView<double, 2, 3> a_view( a.data(), a.stride_0(),
                                                    a.stride_1() );
    EXPECT_EQ( a_view.stride_0(), 3 );
    EXPECT_EQ( a_view.stride_1(), 1 );
    EXPECT_EQ( a_view.stride( 0 ), 3 );
    EXPECT_EQ( a_view.stride( 1 ), 1 );
    EXPECT_EQ( a_view.extent( 0 ), 2 );
    EXPECT_EQ( a_view.extent( 1 ), 3 );

    EXPECT_EQ( a_view( 0, 0 ), 1.2 );
    EXPECT_EQ( a_view( 0, 1 ), -3.5 );
    EXPECT_EQ( a_view( 0, 2 ), 5.4 );
    EXPECT_EQ( a_view( 1, 0 ), 8.6 );
    EXPECT_EQ( a_view( 1, 1 ), 2.6 );
    EXPECT_EQ( a_view( 1, 2 ), -0.1 );

    // Check view rows.
    for ( int i = 0; i < 2; ++i )
    {
        auto row = a_view.row( i );
        EXPECT_EQ( row.extent( 0 ), 3 );
        for ( int j = 0; j < 3; ++j )
            EXPECT_EQ( row( j ), a_view( i, j ) );
    }

    // Check view columns.
    for ( int j = 0; j < 3; ++j )
    {
        auto column = a.column( j );
        EXPECT_EQ( column.extent( 0 ), 2 );
        for ( int i = 0; i < 2; ++i )
            EXPECT_EQ( column( i ), a_view( i, j ) );
    }

    // Check a deep copy.
    auto a_c = a;
    EXPECT_EQ( a_c.stride_0(), 3 );
    EXPECT_EQ( a_c.stride_1(), 1 );
    EXPECT_EQ( a_c.stride( 0 ), 3 );
    EXPECT_EQ( a_c.stride( 1 ), 1 );
    EXPECT_EQ( a_c.extent( 0 ), 2 );
    EXPECT_EQ( a_c.extent( 1 ), 3 );

    EXPECT_EQ( a_c( 0, 0 ), 1.2 );
    EXPECT_EQ( a_c( 0, 1 ), -3.5 );
    EXPECT_EQ( a_c( 0, 2 ), 5.4 );
    EXPECT_EQ( a_c( 1, 0 ), 8.6 );
    EXPECT_EQ( a_c( 1, 1 ), 2.6 );
    EXPECT_EQ( a_c( 1, 2 ), -0.1 );

    // Check a shallow transpose copy.
    auto a_t = ~a;
    EXPECT_EQ( a_t.extent( 0 ), 3 );
    EXPECT_EQ( a_t.extent( 1 ), 2 );

    EXPECT_EQ( a_t( 0, 0 ), 1.2 );
    EXPECT_EQ( a_t( 1, 0 ), -3.5 );
    EXPECT_EQ( a_t( 2, 0 ), 5.4 );
    EXPECT_EQ( a_t( 0, 1 ), 8.6 );
    EXPECT_EQ( a_t( 1, 1 ), 2.6 );
    EXPECT_EQ( a_t( 2, 1 ), -0.1 );

    // Check expression rows.
    for ( int i = 0; i < 3; ++i )
    {
        auto row = a_t.row( i );
        EXPECT_EQ( row.extent( 0 ), 2 );
        for ( int j = 0; j < 2; ++j )
            EXPECT_EQ( row( j ), a_t( i, j ) );
    }

    // Check expression columns.
    for ( int j = 0; j < 2; ++j )
    {
        auto column = a_t.column( j );
        EXPECT_EQ( column.extent( 0 ), 3 );
        for ( int i = 0; i < 3; ++i )
            EXPECT_EQ( column( i ), a_t( i, j ) );
    }

    // Check transpose of transpose shallow copy.
    auto a_t_t = ~a_t;
    EXPECT_EQ( a_t_t.extent( 0 ), 2 );
    EXPECT_EQ( a_t_t.extent( 1 ), 3 );

    EXPECT_EQ( a_t_t( 0, 0 ), 1.2 );
    EXPECT_EQ( a_t_t( 0, 1 ), -3.5 );
    EXPECT_EQ( a_t_t( 0, 2 ), 5.4 );
    EXPECT_EQ( a_t_t( 1, 0 ), 8.6 );
    EXPECT_EQ( a_t_t( 1, 1 ), 2.6 );
    EXPECT_EQ( a_t_t( 1, 2 ), -0.1 );

    // Check a transpose deep copy.
    LinearAlgebra::Matrix<double, 3, 2> a_t_c = ~a;
    EXPECT_EQ( a_t_c.stride_0(), 2 );
    EXPECT_EQ( a_t_c.stride_1(), 1 );
    EXPECT_EQ( a_t_c.stride( 0 ), 2 );
    EXPECT_EQ( a_t_c.stride( 1 ), 1 );
    EXPECT_EQ( a_t_c.extent( 0 ), 3 );
    EXPECT_EQ( a_t_c.extent( 1 ), 2 );

    EXPECT_EQ( a_t_c( 0, 0 ), 1.2 );
    EXPECT_EQ( a_t_c( 1, 0 ), -3.5 );
    EXPECT_EQ( a_t_c( 2, 0 ), 5.4 );
    EXPECT_EQ( a_t_c( 0, 1 ), 8.6 );
    EXPECT_EQ( a_t_c( 1, 1 ), 2.6 );
    EXPECT_EQ( a_t_c( 2, 1 ), -0.1 );

    // Check scalar assignment and operator().
    a = 43.3;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
        {
            EXPECT_EQ( a( i, j ), 43.3 );

            a( i, j ) = -10.2;
            EXPECT_EQ( a( i, j ), -10.2 );
        }

    // Check default initialization.
    LinearAlgebra::Matrix<double, 1, 2> b;
    EXPECT_EQ( b.stride_0(), 2 );
    EXPECT_EQ( b.stride_1(), 1 );
    EXPECT_EQ( b.stride( 0 ), 2 );
    EXPECT_EQ( b.stride( 1 ), 1 );
    EXPECT_EQ( b.extent( 0 ), 1 );
    EXPECT_EQ( b.extent( 1 ), 2 );

    // Check scalar constructor.
    LinearAlgebra::Matrix<double, 2, 3> c = 32.3;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_EQ( c( i, j ), 32.3 );

    // Check scalar multiplication.
    auto d = 2.0 * c;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_EQ( d( i, j ), 64.6 );

    // Check scalar division.
    auto e = d / 2.0;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_EQ( e( i, j ), 32.3 );

    // Check addition assignment.
    LinearAlgebra::Matrix<double, 2, 3> f = e;
    f += d;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_DOUBLE_EQ( f( i, j ), 96.9 );

    // Check subtraction assignment.
    f -= d;
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_DOUBLE_EQ( f( i, j ), 32.3 );

    // Check identity
    LinearAlgebra::Matrix<double, 3, 3> I;
    LinearAlgebra::identity( I );
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_DOUBLE_EQ( I( i, j ), ( i == j ) ? 1.0 : 0.0 );

    // Check trace.
    auto tr_I = LinearAlgebra::trace( I );
    EXPECT_DOUBLE_EQ( 3.0, tr_I );

    // Check deep copy.
    LinearAlgebra::Matrix<double, 2, 3> g;
    LinearAlgebra::deepCopy( g, f );
    for ( int i = 0; i < 2; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_DOUBLE_EQ( g( i, j ), 32.3 );

    // 1x1 matrix test.
    LinearAlgebra::Matrix<double, 1, 1> obo = 3.2;
    obo += ~obo * obo;
    EXPECT_DOUBLE_EQ( 13.44, obo( 0, 0 ) );
    obo -= ~obo * obo;
    EXPECT_DOUBLE_EQ( -167.1936, obo( 0, 0 ) );
    obo = 12.0;
    EXPECT_DOUBLE_EQ( 12.0, obo( 0, 0 ) );
    LinearAlgebra::Matrix<double, 1, 1> obo2 = ~obo * obo;
    EXPECT_DOUBLE_EQ( 144.0, obo2( 0, 0 ) );
}

void tensor3Test()
{
    using Kokkos::ALL;

    // Check a basic Tensor3
    LinearAlgebra::Tensor3<double, 3, 4, 2> t = {
        { { 2.3, -1.1 }, { 2.0, -3.2 }, { -8.3, -9.1 }, { 1.4, 5.8 } },
        { { 7.2, 4.5 }, { -2.5, -2.8 }, { 3.1, 4.0 }, { -7.7, 6.4 } },
        { { -9.0, 8.2 }, { 0.3, -1.9 }, { -3.5, 6.6 }, { 4.0, 1.4 } } };

    EXPECT_EQ( t.stride( 0 ), 8 );
    EXPECT_EQ( t.stride( 1 ), 2 );
    EXPECT_EQ( t.stride( 2 ), 1 );
    EXPECT_EQ( t.stride_0(), 8 );
    EXPECT_EQ( t.stride_1(), 2 );
    EXPECT_EQ( t.stride_2(), 1 );
    EXPECT_EQ( t.extent( 0 ), 3 );
    EXPECT_EQ( t.extent( 1 ), 4 );
    EXPECT_EQ( t.extent( 2 ), 2 );

    EXPECT_EQ( t( 0, 0, 0 ), 2.3 );
    EXPECT_EQ( t( 0, 0, 1 ), -1.1 );
    EXPECT_EQ( t( 0, 1, 0 ), 2.0 );
    EXPECT_EQ( t( 0, 1, 1 ), -3.2 );
    EXPECT_EQ( t( 0, 2, 0 ), -8.3 );
    EXPECT_EQ( t( 0, 2, 1 ), -9.1 );
    EXPECT_EQ( t( 0, 3, 0 ), 1.4 );
    EXPECT_EQ( t( 0, 3, 1 ), 5.8 );
    EXPECT_EQ( t( 1, 0, 0 ), 7.2 );
    EXPECT_EQ( t( 1, 0, 1 ), 4.5 );
    EXPECT_EQ( t( 1, 1, 0 ), -2.5 );
    EXPECT_EQ( t( 1, 1, 1 ), -2.8 );
    EXPECT_EQ( t( 1, 2, 0 ), 3.1 );
    EXPECT_EQ( t( 1, 2, 1 ), 4.0 );
    EXPECT_EQ( t( 1, 3, 0 ), -7.7 );
    EXPECT_EQ( t( 1, 3, 1 ), 6.4 );
    EXPECT_EQ( t( 2, 0, 0 ), -9.0 );
    EXPECT_EQ( t( 2, 0, 1 ), 8.2 );
    EXPECT_EQ( t( 2, 1, 0 ), 0.3 );
    EXPECT_EQ( t( 2, 1, 1 ), -1.9 );
    EXPECT_EQ( t( 2, 2, 0 ), -3.5 );
    EXPECT_EQ( t( 2, 2, 1 ), 6.6 );
    EXPECT_EQ( t( 2, 3, 0 ), 4.0 );
    EXPECT_EQ( t( 2, 3, 1 ), 1.4 );

    auto m2_1 = t.matrix( ALL(), ALL(), 0 );
    EXPECT_EQ( m2_1.extent( 0 ), 3 );
    EXPECT_EQ( m2_1.extent( 1 ), 4 );

    EXPECT_EQ( m2_1( 0, 0 ), 2.3 );
    EXPECT_EQ( m2_1( 0, 1 ), 2.0 );
    EXPECT_EQ( m2_1( 0, 2 ), -8.3 );
    EXPECT_EQ( m2_1( 0, 3 ), 1.4 );
    EXPECT_EQ( m2_1( 1, 0 ), 7.2 );
    EXPECT_EQ( m2_1( 1, 1 ), -2.5 );
    EXPECT_EQ( m2_1( 1, 2 ), 3.1 );
    EXPECT_EQ( m2_1( 1, 3 ), -7.7 );
    EXPECT_EQ( m2_1( 2, 0 ), -9.0 );
    EXPECT_EQ( m2_1( 2, 1 ), 0.3 );
    EXPECT_EQ( m2_1( 2, 2 ), -3.5 );
    EXPECT_EQ( m2_1( 2, 3 ), 4.0 );

    auto m2_2 = t.matrix( ALL(), 0, ALL() );
    EXPECT_EQ( m2_2.extent( 0 ), 3 );
    EXPECT_EQ( m2_2.extent( 1 ), 2 );

    EXPECT_EQ( m2_2( 0, 0 ), 2.3 );
    EXPECT_EQ( m2_2( 0, 1 ), -1.1 );
    EXPECT_EQ( m2_2( 1, 0 ), 7.2 );
    EXPECT_EQ( m2_2( 1, 1 ), 4.5 );
    EXPECT_EQ( m2_2( 2, 0 ), -9.0 );
    EXPECT_EQ( m2_2( 2, 1 ), 8.2 );

    auto m2_3 = t.matrix( 0, ALL(), ALL() );
    EXPECT_EQ( m2_3.extent( 0 ), 4 );
    EXPECT_EQ( m2_3.extent( 1 ), 2 );

    EXPECT_EQ( m2_3( 0, 0 ), 2.3 );
    EXPECT_EQ( m2_3( 0, 1 ), -1.1 );
    EXPECT_EQ( m2_3( 1, 0 ), 2.0 );
    EXPECT_EQ( m2_3( 1, 1 ), -3.2 );
    EXPECT_EQ( m2_3( 2, 0 ), -8.3 );
    EXPECT_EQ( m2_3( 2, 1 ), -9.1 );
    EXPECT_EQ( m2_3( 3, 0 ), 1.4 );
    EXPECT_EQ( m2_3( 3, 1 ), 5.8 );

    auto v1_1 = t.vector( ALL(), 0, 0 );
    EXPECT_EQ( v1_1.extent( 0 ), 3 );

    EXPECT_EQ( v1_1( 0 ), 2.3 );
    EXPECT_EQ( v1_1( 1 ), 7.2 );
    EXPECT_EQ( v1_1( 2 ), -9.0 );

    auto v1_2 = t.vector( 0, ALL(), 0 );
    EXPECT_EQ( v1_2.extent( 0 ), 4 );

    EXPECT_EQ( v1_2( 0 ), 2.3 );
    EXPECT_EQ( v1_2( 1 ), 2.0 );
    EXPECT_EQ( v1_2( 2 ), -8.3 );
    EXPECT_EQ( v1_2( 3 ), 1.4 );

    auto v1_3 = t.vector( 0, 0, ALL() );
    EXPECT_EQ( v1_3.extent( 0 ), 2 );

    EXPECT_EQ( v1_3( 0 ), 2.3 );
    EXPECT_EQ( v1_3( 1 ), -1.1 );

    // Check tensor3 view
    LinearAlgebra::Tensor3View<double, 3, 4, 2> t_view(
        t.data(), t.stride_0(), t.stride_1(), t.stride_2() );

    EXPECT_EQ( t_view.stride( 0 ), 8 );
    EXPECT_EQ( t_view.stride( 1 ), 2 );
    EXPECT_EQ( t_view.stride( 2 ), 1 );
    EXPECT_EQ( t_view.stride_0(), 8 );
    EXPECT_EQ( t_view.stride_1(), 2 );
    EXPECT_EQ( t_view.stride_2(), 1 );
    EXPECT_EQ( t_view.extent( 0 ), 3 );
    EXPECT_EQ( t_view.extent( 1 ), 4 );
    EXPECT_EQ( t_view.extent( 2 ), 2 );

    EXPECT_EQ( t_view( 0, 0, 0 ), 2.3 );
    EXPECT_EQ( t_view( 0, 0, 1 ), -1.1 );
    EXPECT_EQ( t_view( 0, 1, 0 ), 2.0 );
    EXPECT_EQ( t_view( 0, 1, 1 ), -3.2 );
    EXPECT_EQ( t_view( 0, 2, 0 ), -8.3 );
    EXPECT_EQ( t_view( 0, 2, 1 ), -9.1 );
    EXPECT_EQ( t_view( 0, 3, 0 ), 1.4 );
    EXPECT_EQ( t_view( 0, 3, 1 ), 5.8 );
    EXPECT_EQ( t_view( 1, 0, 0 ), 7.2 );
    EXPECT_EQ( t_view( 1, 0, 1 ), 4.5 );
    EXPECT_EQ( t_view( 1, 1, 0 ), -2.5 );
    EXPECT_EQ( t_view( 1, 1, 1 ), -2.8 );
    EXPECT_EQ( t_view( 1, 2, 0 ), 3.1 );
    EXPECT_EQ( t_view( 1, 2, 1 ), 4.0 );
    EXPECT_EQ( t_view( 1, 3, 0 ), -7.7 );
    EXPECT_EQ( t_view( 1, 3, 1 ), 6.4 );
    EXPECT_EQ( t_view( 2, 0, 0 ), -9.0 );
    EXPECT_EQ( t_view( 2, 0, 1 ), 8.2 );
    EXPECT_EQ( t_view( 2, 1, 0 ), 0.3 );
    EXPECT_EQ( t_view( 2, 1, 1 ), -1.9 );
    EXPECT_EQ( t_view( 2, 2, 0 ), -3.5 );
    EXPECT_EQ( t_view( 2, 2, 1 ), 6.6 );
    EXPECT_EQ( t_view( 2, 3, 0 ), 4.0 );
    EXPECT_EQ( t_view( 2, 3, 1 ), 1.4 );

    // Check scalar assignment and () operator
    t = 43.3;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                EXPECT_EQ( t( i, j, k ), 43.3 );

                t( i, j, k ) = -10.2;
                EXPECT_EQ( t( i, j, k ), -10.2 );
            }
        }
    }

    // Check default initialization
    LinearAlgebra::Tensor3<double, 3, 4, 2> s;
    EXPECT_EQ( s.stride( 0 ), 8 );
    EXPECT_EQ( s.stride( 1 ), 2 );
    EXPECT_EQ( s.stride( 2 ), 1 );
    EXPECT_EQ( s.stride_0(), 8 );
    EXPECT_EQ( s.stride_1(), 2 );
    EXPECT_EQ( s.stride_2(), 1 );
    EXPECT_EQ( s.extent( 0 ), 3 );
    EXPECT_EQ( s.extent( 1 ), 4 );
    EXPECT_EQ( s.extent( 2 ), 2 );

    // Check scalar constructor
    LinearAlgebra::Tensor3<double, 3, 4, 2> u = 33.0;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                EXPECT_EQ( u( i, j, k ), 33.0 );
            }
        }
    }

    // Check scalar multiplication
    auto v = 2.0 * u;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                EXPECT_EQ( v( i, j, k ), 66.0 );
            }
        }
    }

    // Check scalar division
    auto w = u / 2.0;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                EXPECT_EQ( w( i, j, k ), 16.5 );
            }
        }
    }

    // Check addition assignment
    LinearAlgebra::Tensor3<double, 3, 4, 2> tt = w;
    tt += u;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                EXPECT_DOUBLE_EQ( tt( i, j, k ), 49.5 );
            }
        }
    }

    // Check subtraction assignment
    tt -= u;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                EXPECT_DOUBLE_EQ( tt( i, j, k ), 16.5 );
            }
        }
    }
}

void tensor4Test()
{
    using Kokkos::ALL;

    // Check a basic Tensor4
    LinearAlgebra::Tensor4<double, 3, 4, 2, 2> t = {
        { { { 2.3, -1.1 }, { 4.0, 8.7 } },
          { { 2.0, -3.2 }, { -6.9, -2.1 } },
          { { -8.3, -9.1 }, { 3.3, -4.4 } },
          { { 1.4, 5.8 }, { -5.2, -9.1 } } },
        { { { 7.2, 4.5 }, { 4.6, 8.8 } },
          { { -2.5, -2.8 }, { -1.7, 0.3 } },
          { { 3.1, 4.0 }, { 0.6, -4.8 } },
          { { -7.7, 6.4 }, { -9.2, 3.1 } } },
        { { { -9.0, 8.2 }, { 7.5, 3.4 } },
          { { 0.3, -1.9 }, { 9.2, -7.7 } },
          { { -3.5, 6.6 }, { 3.9, 2.9 } },
          { { 4.0, 1.4 }, { -6.5, -8.2 } } } };

    EXPECT_EQ( t.stride( 0 ), 16 );
    EXPECT_EQ( t.stride( 1 ), 4 );
    EXPECT_EQ( t.stride( 2 ), 2 );
    EXPECT_EQ( t.stride( 3 ), 1 );
    EXPECT_EQ( t.stride_0(), 16 );
    EXPECT_EQ( t.stride_1(), 4 );
    EXPECT_EQ( t.stride_2(), 2 );
    EXPECT_EQ( t.stride_3(), 1 );
    EXPECT_EQ( t.extent( 0 ), 3 );
    EXPECT_EQ( t.extent( 1 ), 4 );
    EXPECT_EQ( t.extent( 2 ), 2 );
    EXPECT_EQ( t.extent( 3 ), 2 );

    EXPECT_EQ( t( 0, 0, 0, 0 ), 2.3 );
    EXPECT_EQ( t( 0, 0, 0, 1 ), -1.1 );
    EXPECT_EQ( t( 0, 0, 1, 0 ), 4.0 );
    EXPECT_EQ( t( 0, 0, 1, 1 ), 8.7 );

    EXPECT_EQ( t( 0, 1, 0, 0 ), 2.0 );
    EXPECT_EQ( t( 0, 1, 0, 1 ), -3.2 );
    EXPECT_EQ( t( 0, 1, 1, 0 ), -6.9 );
    EXPECT_EQ( t( 0, 1, 1, 1 ), -2.1 );

    EXPECT_EQ( t( 0, 2, 0, 0 ), -8.3 );
    EXPECT_EQ( t( 0, 2, 0, 1 ), -9.1 );
    EXPECT_EQ( t( 0, 2, 1, 0 ), 3.3 );
    EXPECT_EQ( t( 0, 2, 1, 1 ), -4.4 );

    EXPECT_EQ( t( 0, 3, 0, 0 ), 1.4 );
    EXPECT_EQ( t( 0, 3, 0, 1 ), 5.8 );
    EXPECT_EQ( t( 0, 3, 1, 0 ), -5.2 );
    EXPECT_EQ( t( 0, 3, 1, 1 ), -9.1 );

    EXPECT_EQ( t( 1, 0, 0, 0 ), 7.2 );
    EXPECT_EQ( t( 1, 0, 0, 1 ), 4.5 );
    EXPECT_EQ( t( 1, 0, 1, 0 ), 4.6 );
    EXPECT_EQ( t( 1, 0, 1, 1 ), 8.8 );

    EXPECT_EQ( t( 1, 1, 0, 0 ), -2.5 );
    EXPECT_EQ( t( 1, 1, 0, 1 ), -2.8 );
    EXPECT_EQ( t( 1, 1, 1, 0 ), -1.7 );
    EXPECT_EQ( t( 1, 1, 1, 1 ), 0.3 );

    EXPECT_EQ( t( 1, 2, 0, 0 ), 3.1 );
    EXPECT_EQ( t( 1, 2, 0, 1 ), 4.0 );
    EXPECT_EQ( t( 1, 2, 1, 0 ), 0.6 );
    EXPECT_EQ( t( 1, 2, 1, 1 ), -4.8 );

    EXPECT_EQ( t( 1, 3, 0, 0 ), -7.7 );
    EXPECT_EQ( t( 1, 3, 0, 1 ), 6.4 );
    EXPECT_EQ( t( 1, 3, 1, 0 ), -9.2 );
    EXPECT_EQ( t( 1, 3, 1, 1 ), 3.1 );

    EXPECT_EQ( t( 2, 0, 0, 0 ), -9.0 );
    EXPECT_EQ( t( 2, 0, 0, 1 ), 8.2 );
    EXPECT_EQ( t( 2, 0, 1, 0 ), 7.5 );
    EXPECT_EQ( t( 2, 0, 1, 1 ), 3.4 );

    EXPECT_EQ( t( 2, 1, 0, 0 ), 0.3 );
    EXPECT_EQ( t( 2, 1, 0, 1 ), -1.9 );
    EXPECT_EQ( t( 2, 1, 1, 0 ), 9.2 );
    EXPECT_EQ( t( 2, 1, 1, 1 ), -7.7 );

    EXPECT_EQ( t( 2, 2, 0, 0 ), -3.5 );
    EXPECT_EQ( t( 2, 2, 0, 1 ), 6.6 );
    EXPECT_EQ( t( 2, 2, 1, 0 ), 3.9 );
    EXPECT_EQ( t( 2, 2, 1, 1 ), 2.9 );

    EXPECT_EQ( t( 2, 3, 0, 0 ), 4.0 );
    EXPECT_EQ( t( 2, 3, 0, 1 ), 1.4 );
    EXPECT_EQ( t( 2, 3, 1, 0 ), -6.5 );
    EXPECT_EQ( t( 2, 3, 1, 1 ), -8.2 );

    // Check tensor4 view
    LinearAlgebra::Tensor4View<double, 3, 4, 2, 2> t_view(
        t.data(), t.stride_0(), t.stride_1(), t.stride_2(), t.stride_3() );

    EXPECT_EQ( t_view.stride( 0 ), 16 );
    EXPECT_EQ( t_view.stride( 1 ), 4 );
    EXPECT_EQ( t_view.stride( 2 ), 2 );
    EXPECT_EQ( t_view.stride( 3 ), 1 );
    EXPECT_EQ( t_view.stride_0(), 16 );
    EXPECT_EQ( t_view.stride_1(), 4 );
    EXPECT_EQ( t_view.stride_2(), 2 );
    EXPECT_EQ( t_view.stride_3(), 1 );
    EXPECT_EQ( t_view.extent( 0 ), 3 );
    EXPECT_EQ( t_view.extent( 1 ), 4 );
    EXPECT_EQ( t_view.extent( 2 ), 2 );
    EXPECT_EQ( t_view.extent( 3 ), 2 );

    EXPECT_EQ( t_view( 0, 0, 0, 0 ), 2.3 );
    EXPECT_EQ( t_view( 0, 0, 0, 1 ), -1.1 );
    EXPECT_EQ( t_view( 0, 0, 1, 0 ), 4.0 );
    EXPECT_EQ( t_view( 0, 0, 1, 1 ), 8.7 );

    EXPECT_EQ( t_view( 0, 1, 0, 0 ), 2.0 );
    EXPECT_EQ( t_view( 0, 1, 0, 1 ), -3.2 );
    EXPECT_EQ( t_view( 0, 1, 1, 0 ), -6.9 );
    EXPECT_EQ( t_view( 0, 1, 1, 1 ), -2.1 );

    EXPECT_EQ( t_view( 0, 2, 0, 0 ), -8.3 );
    EXPECT_EQ( t_view( 0, 2, 0, 1 ), -9.1 );
    EXPECT_EQ( t_view( 0, 2, 1, 0 ), 3.3 );
    EXPECT_EQ( t_view( 0, 2, 1, 1 ), -4.4 );

    EXPECT_EQ( t_view( 0, 3, 0, 0 ), 1.4 );
    EXPECT_EQ( t_view( 0, 3, 0, 1 ), 5.8 );
    EXPECT_EQ( t_view( 0, 3, 1, 0 ), -5.2 );
    EXPECT_EQ( t_view( 0, 3, 1, 1 ), -9.1 );

    EXPECT_EQ( t_view( 1, 0, 0, 0 ), 7.2 );
    EXPECT_EQ( t_view( 1, 0, 0, 1 ), 4.5 );
    EXPECT_EQ( t_view( 1, 0, 1, 0 ), 4.6 );
    EXPECT_EQ( t_view( 1, 0, 1, 1 ), 8.8 );

    EXPECT_EQ( t_view( 1, 1, 0, 0 ), -2.5 );
    EXPECT_EQ( t_view( 1, 1, 0, 1 ), -2.8 );
    EXPECT_EQ( t_view( 1, 1, 1, 0 ), -1.7 );
    EXPECT_EQ( t_view( 1, 1, 1, 1 ), 0.3 );

    EXPECT_EQ( t_view( 1, 2, 0, 0 ), 3.1 );
    EXPECT_EQ( t_view( 1, 2, 0, 1 ), 4.0 );
    EXPECT_EQ( t_view( 1, 2, 1, 0 ), 0.6 );
    EXPECT_EQ( t_view( 1, 2, 1, 1 ), -4.8 );

    EXPECT_EQ( t_view( 1, 3, 0, 0 ), -7.7 );
    EXPECT_EQ( t_view( 1, 3, 0, 1 ), 6.4 );
    EXPECT_EQ( t_view( 1, 3, 1, 0 ), -9.2 );
    EXPECT_EQ( t_view( 1, 3, 1, 1 ), 3.1 );

    EXPECT_EQ( t_view( 2, 0, 0, 0 ), -9.0 );
    EXPECT_EQ( t_view( 2, 0, 0, 1 ), 8.2 );
    EXPECT_EQ( t_view( 2, 0, 1, 0 ), 7.5 );
    EXPECT_EQ( t_view( 2, 0, 1, 1 ), 3.4 );

    EXPECT_EQ( t_view( 2, 1, 0, 0 ), 0.3 );
    EXPECT_EQ( t_view( 2, 1, 0, 1 ), -1.9 );
    EXPECT_EQ( t_view( 2, 1, 1, 0 ), 9.2 );
    EXPECT_EQ( t_view( 2, 1, 1, 1 ), -7.7 );

    EXPECT_EQ( t_view( 2, 2, 0, 0 ), -3.5 );
    EXPECT_EQ( t_view( 2, 2, 0, 1 ), 6.6 );
    EXPECT_EQ( t_view( 2, 2, 1, 0 ), 3.9 );
    EXPECT_EQ( t_view( 2, 2, 1, 1 ), 2.9 );

    EXPECT_EQ( t_view( 2, 3, 0, 0 ), 4.0 );
    EXPECT_EQ( t_view( 2, 3, 0, 1 ), 1.4 );
    EXPECT_EQ( t_view( 2, 3, 1, 0 ), -6.5 );
    EXPECT_EQ( t_view( 2, 3, 1, 1 ), -8.2 );

    // Test general slicing operations

    auto t3_1 = t.tensor3( 1, ALL(), ALL(), ALL() );
    // Expect a <4,2,2> shaped Tensor3View
    EXPECT_EQ( t3_1.extent( 0 ), 4 );
    EXPECT_EQ( t3_1.extent( 1 ), 2 );
    EXPECT_EQ( t3_1.extent( 2 ), 2 );

    EXPECT_EQ( t3_1( 0, 0, 0 ), 7.2 );
    EXPECT_EQ( t3_1( 0, 0, 1 ), 4.5 );
    EXPECT_EQ( t3_1( 0, 1, 0 ), 4.6 );
    EXPECT_EQ( t3_1( 0, 1, 1 ), 8.8 );
    EXPECT_EQ( t3_1( 1, 0, 0 ), -2.5 );
    EXPECT_EQ( t3_1( 1, 0, 1 ), -2.8 );
    EXPECT_EQ( t3_1( 1, 1, 0 ), -1.7 );
    EXPECT_EQ( t3_1( 1, 1, 1 ), 0.3 );
    EXPECT_EQ( t3_1( 2, 0, 0 ), 3.1 );
    EXPECT_EQ( t3_1( 2, 0, 1 ), 4.0 );
    EXPECT_EQ( t3_1( 2, 1, 0 ), 0.6 );
    EXPECT_EQ( t3_1( 2, 1, 1 ), -4.8 );
    EXPECT_EQ( t3_1( 3, 0, 0 ), -7.7 );
    EXPECT_EQ( t3_1( 3, 0, 1 ), 6.4 );
    EXPECT_EQ( t3_1( 3, 1, 0 ), -9.2 );
    EXPECT_EQ( t3_1( 3, 1, 1 ), 3.1 );

    auto t3_2 = t.tensor3( ALL(), 1, ALL(), ALL() );
    // Expect a <3,2,2> shaped Tensor3View
    EXPECT_EQ( t3_2.extent( 0 ), 3 );
    EXPECT_EQ( t3_2.extent( 1 ), 2 );
    EXPECT_EQ( t3_2.extent( 2 ), 2 );

    EXPECT_EQ( t3_2( 0, 0, 0 ), 2.0 );
    EXPECT_EQ( t3_2( 0, 0, 1 ), -3.2 );
    EXPECT_EQ( t3_2( 0, 1, 0 ), -6.9 );
    EXPECT_EQ( t3_2( 0, 1, 1 ), -2.1 );
    EXPECT_EQ( t3_2( 1, 0, 0 ), -2.5 );
    EXPECT_EQ( t3_2( 1, 0, 1 ), -2.8 );
    EXPECT_EQ( t3_2( 1, 1, 0 ), -1.7 );
    EXPECT_EQ( t3_2( 1, 1, 1 ), 0.3 );
    EXPECT_EQ( t3_2( 2, 0, 0 ), 0.3 );
    EXPECT_EQ( t3_2( 2, 0, 1 ), -1.9 );
    EXPECT_EQ( t3_2( 2, 1, 0 ), 9.2 );
    EXPECT_EQ( t3_2( 2, 1, 1 ), -7.7 );

    auto t3_3 = t.tensor3( ALL(), ALL(), ALL(), 0 );
    // Expect a <3,4,2> shaped Tensor3View
    EXPECT_EQ( t3_3.extent( 0 ), 3 );
    EXPECT_EQ( t3_3.extent( 1 ), 4 );
    EXPECT_EQ( t3_3.extent( 2 ), 2 );

    EXPECT_EQ( t3_3( 0, 0, 0 ), 2.3 );
    EXPECT_EQ( t3_3( 0, 0, 1 ), 4.0 );
    EXPECT_EQ( t3_3( 0, 1, 0 ), 2.0 );
    EXPECT_EQ( t3_3( 0, 1, 1 ), -6.9 );
    EXPECT_EQ( t3_3( 0, 2, 0 ), -8.3 );
    EXPECT_EQ( t3_3( 0, 2, 1 ), 3.3 );
    EXPECT_EQ( t3_3( 0, 3, 0 ), 1.4 );
    EXPECT_EQ( t3_3( 0, 3, 1 ), -5.2 );
    EXPECT_EQ( t3_3( 1, 0, 0 ), 7.2 );
    EXPECT_EQ( t3_3( 1, 0, 1 ), 4.6 );
    EXPECT_EQ( t3_3( 1, 1, 0 ), -2.5 );
    EXPECT_EQ( t3_3( 1, 1, 1 ), -1.7 );
    EXPECT_EQ( t3_3( 1, 2, 0 ), 3.1 );
    EXPECT_EQ( t3_3( 1, 2, 1 ), 0.6 );
    EXPECT_EQ( t3_3( 1, 3, 0 ), -7.7 );
    EXPECT_EQ( t3_3( 1, 3, 1 ), -9.2 );
    EXPECT_EQ( t3_3( 2, 0, 0 ), -9.0 );
    EXPECT_EQ( t3_3( 2, 0, 1 ), 7.5 );
    EXPECT_EQ( t3_3( 2, 1, 0 ), 0.3 );
    EXPECT_EQ( t3_3( 2, 1, 1 ), 9.2 );
    EXPECT_EQ( t3_3( 2, 2, 0 ), -3.5 );
    EXPECT_EQ( t3_3( 2, 2, 1 ), 3.9 );
    EXPECT_EQ( t3_3( 2, 3, 0 ), 4.0 );
    EXPECT_EQ( t3_3( 2, 3, 1 ), -6.5 );

    auto t3_4 = t.tensor3( ALL(), ALL(), 1, ALL() );
    // Expect a <3,4,2> shaped Tensor3View
    EXPECT_EQ( t3_4.extent( 0 ), 3 );
    EXPECT_EQ( t3_4.extent( 1 ), 4 );
    EXPECT_EQ( t3_4.extent( 2 ), 2 );

    EXPECT_EQ( t3_4( 0, 0, 0 ), 4.0 );
    EXPECT_EQ( t3_4( 0, 0, 1 ), 8.7 );
    EXPECT_EQ( t3_4( 0, 1, 0 ), -6.9 );
    EXPECT_EQ( t3_4( 0, 1, 1 ), -2.1 );
    EXPECT_EQ( t3_4( 0, 2, 0 ), 3.3 );
    EXPECT_EQ( t3_4( 0, 2, 1 ), -4.4 );
    EXPECT_EQ( t3_4( 0, 3, 0 ), -5.2 );
    EXPECT_EQ( t3_4( 0, 3, 1 ), -9.1 );
    EXPECT_EQ( t3_4( 1, 0, 0 ), 4.6 );
    EXPECT_EQ( t3_4( 1, 0, 1 ), 8.8 );
    EXPECT_EQ( t3_4( 1, 1, 0 ), -1.7 );
    EXPECT_EQ( t3_4( 1, 1, 1 ), 0.3 );
    EXPECT_EQ( t3_4( 1, 2, 0 ), 0.6 );
    EXPECT_EQ( t3_4( 1, 2, 1 ), -4.8 );
    EXPECT_EQ( t3_4( 1, 3, 0 ), -9.2 );
    EXPECT_EQ( t3_4( 1, 3, 1 ), 3.1 );
    EXPECT_EQ( t3_4( 2, 0, 0 ), 7.5 );
    EXPECT_EQ( t3_4( 2, 0, 1 ), 3.4 );
    EXPECT_EQ( t3_4( 2, 1, 0 ), 9.2 );
    EXPECT_EQ( t3_4( 2, 1, 1 ), -7.7 );
    EXPECT_EQ( t3_4( 2, 2, 0 ), 3.9 );
    EXPECT_EQ( t3_4( 2, 2, 1 ), 2.9 );
    EXPECT_EQ( t3_4( 2, 3, 0 ), -6.5 );
    EXPECT_EQ( t3_4( 2, 3, 1 ), -8.2 );

    auto m2_1 = t.matrix( ALL(), ALL(), 0, 0 );
    // Expect a <3,4> shaped MatrixView
    EXPECT_EQ( m2_1.extent( 0 ), 3 );
    EXPECT_EQ( m2_1.extent( 1 ), 4 );

    EXPECT_EQ( m2_1( 0, 0 ), 2.3 );
    EXPECT_EQ( m2_1( 0, 1 ), 2.0 );
    EXPECT_EQ( m2_1( 0, 2 ), -8.3 );
    EXPECT_EQ( m2_1( 0, 3 ), 1.4 );
    EXPECT_EQ( m2_1( 1, 0 ), 7.2 );
    EXPECT_EQ( m2_1( 1, 1 ), -2.5 );
    EXPECT_EQ( m2_1( 1, 2 ), 3.1 );
    EXPECT_EQ( m2_1( 1, 3 ), -7.7 );
    EXPECT_EQ( m2_1( 2, 0 ), -9.0 );
    EXPECT_EQ( m2_1( 2, 1 ), 0.3 );
    EXPECT_EQ( m2_1( 2, 2 ), -3.5 );
    EXPECT_EQ( m2_1( 2, 3 ), 4.0 );

    auto m2_2 = t.matrix( ALL(), 0, ALL(), 0 );
    // Expect a <3,2> shaped MatrixView
    EXPECT_EQ( m2_2.extent( 0 ), 3 );
    EXPECT_EQ( m2_2.extent( 1 ), 2 );

    EXPECT_EQ( m2_2( 0, 0 ), 2.3 );
    EXPECT_EQ( m2_2( 0, 1 ), 4.0 );
    EXPECT_EQ( m2_2( 1, 0 ), 7.2 );
    EXPECT_EQ( m2_2( 1, 1 ), 4.6 );
    EXPECT_EQ( m2_2( 2, 0 ), -9.0 );
    EXPECT_EQ( m2_2( 2, 1 ), 7.5 );

    auto m2_3 = t.matrix( 0, ALL(), ALL(), 0 );
    // Expect a <4,2> shaped MatrixView
    EXPECT_EQ( m2_3.extent( 0 ), 4 );
    EXPECT_EQ( m2_3.extent( 1 ), 2 );

    EXPECT_EQ( m2_3( 0, 0 ), 2.3 );
    EXPECT_EQ( m2_3( 0, 1 ), 4.0 );
    EXPECT_EQ( m2_3( 1, 0 ), 2.0 );
    EXPECT_EQ( m2_3( 1, 1 ), -6.9 );
    EXPECT_EQ( m2_3( 2, 0 ), -8.3 );
    EXPECT_EQ( m2_3( 2, 1 ), 3.3 );
    EXPECT_EQ( m2_3( 3, 0 ), 1.4 );
    EXPECT_EQ( m2_3( 3, 1 ), -5.2 );

    auto m2_4 = t.matrix( 0, ALL(), 0, ALL() );
    // Expect a <4,2> shaped MatrixView
    EXPECT_EQ( m2_4.extent( 0 ), 4 );
    EXPECT_EQ( m2_4.extent( 1 ), 2 );

    EXPECT_EQ( m2_4( 0, 0 ), 2.3 );
    EXPECT_EQ( m2_4( 0, 1 ), -1.1 );
    EXPECT_EQ( m2_4( 1, 0 ), 2.0 );
    EXPECT_EQ( m2_4( 1, 1 ), -3.2 );
    EXPECT_EQ( m2_4( 2, 0 ), -8.3 );
    EXPECT_EQ( m2_4( 2, 1 ), -9.1 );
    EXPECT_EQ( m2_4( 3, 0 ), 1.4 );
    EXPECT_EQ( m2_4( 3, 1 ), 5.8 );

    auto m2_5 = t.matrix( 0, 0, ALL(), ALL() );
    // Expect a <2,2> shaped MatrixView
    EXPECT_EQ( m2_5.extent( 0 ), 2 );
    EXPECT_EQ( m2_5.extent( 1 ), 2 );

    EXPECT_EQ( m2_5( 0, 0 ), 2.3 );
    EXPECT_EQ( m2_5( 0, 1 ), -1.1 );
    EXPECT_EQ( m2_5( 1, 0 ), 4.0 );
    EXPECT_EQ( m2_5( 1, 1 ), 8.7 );

    auto m2_6 = t.matrix( ALL(), 0, 0, ALL() );
    // Expect a <3,2> shaped MatrixView
    EXPECT_EQ( m2_6.extent( 0 ), 3 );
    EXPECT_EQ( m2_6.extent( 1 ), 2 );

    EXPECT_EQ( m2_6( 0, 0 ), 2.3 );
    EXPECT_EQ( m2_6( 0, 1 ), -1.1 );
    EXPECT_EQ( m2_6( 1, 0 ), 7.2 );
    EXPECT_EQ( m2_6( 1, 1 ), 4.5 );
    EXPECT_EQ( m2_6( 2, 0 ), -9.0 );
    EXPECT_EQ( m2_6( 2, 1 ), 8.2 );

    auto v1 = t.vector( 0, 0, 0, ALL() );
    // Expect a <2> shaped VectorView
    EXPECT_EQ( v1.extent( 0 ), 2 );

    EXPECT_EQ( v1( 0 ), 2.3 );
    EXPECT_EQ( v1( 1 ), -1.1 );

    auto v1_1 = t.vector( 2, 2, 1, ALL() );
    // Expect a <2> shaped VectorView
    EXPECT_EQ( v1_1.extent( 0 ), 2 );

    EXPECT_EQ( v1_1( 0 ), 3.9 );
    EXPECT_EQ( v1_1( 1 ), 2.9 );

    auto v1_2 = t.vector( 0, 0, ALL(), 0 );
    // Expect a <2> shaped VectorView
    EXPECT_EQ( v1_2.extent( 0 ), 2 );

    EXPECT_EQ( v1_2( 0 ), 2.3 );
    EXPECT_EQ( v1_2( 1 ), 4.0 );

    auto v1_3 = t.vector( 0, ALL(), 0, 0 );
    // Expect a <4> shaped VectorView
    EXPECT_EQ( v1_3.extent( 0 ), 4 );

    EXPECT_EQ( v1_3( 0 ), 2.3 );
    EXPECT_EQ( v1_3( 1 ), 2.0 );
    EXPECT_EQ( v1_3( 2 ), -8.3 );
    EXPECT_EQ( v1_3( 3 ), 1.4 );

    auto v1_4 = t.vector( ALL(), 0, 0, 0 );
    // Expect a <3> shaped VectorView
    EXPECT_EQ( v1_4.extent( 0 ), 3 );

    EXPECT_EQ( v1_4( 0 ), 2.3 );
    EXPECT_EQ( v1_4( 1 ), 7.2 );
    EXPECT_EQ( v1_4( 2 ), -9.0 );

    auto v1_5 = t.vector( ALL(), 2, 1, 0 );
    // Expect a <3> shaped VectorView
    EXPECT_EQ( v1_5.extent( 0 ), 3 );

    EXPECT_EQ( v1_5( 0 ), 3.3 );
    EXPECT_EQ( v1_5( 1 ), 0.6 );
    EXPECT_EQ( v1_5( 2 ), 3.9 );

    // Check scalar assignment and () operator
    t = 43.3;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                for ( int l = 0; l < 2; ++l )
                {
                    EXPECT_EQ( t( i, j, k, l ), 43.3 );

                    t( i, j, k, l ) = -10.2;
                    EXPECT_EQ( t( i, j, k, l ), -10.2 );
                }
            }
        }
    }

    // Check default initialization
    LinearAlgebra::Tensor4<double, 3, 4, 2, 2> s;
    EXPECT_EQ( s.stride( 0 ), 16 );
    EXPECT_EQ( s.stride( 1 ), 4 );
    EXPECT_EQ( s.stride( 2 ), 2 );
    EXPECT_EQ( s.stride( 3 ), 1 );
    EXPECT_EQ( s.stride_0(), 16 );
    EXPECT_EQ( s.stride_1(), 4 );
    EXPECT_EQ( s.stride_2(), 2 );
    EXPECT_EQ( s.stride_3(), 1 );
    EXPECT_EQ( s.extent( 0 ), 3 );
    EXPECT_EQ( s.extent( 1 ), 4 );
    EXPECT_EQ( s.extent( 2 ), 2 );
    EXPECT_EQ( s.extent( 3 ), 2 );

    // Check scalar constructor
    LinearAlgebra::Tensor4<double, 3, 4, 2, 2> u = 33.0;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                for ( int l = 0; l < 2; ++l )
                {
                    EXPECT_EQ( u( i, j, k, l ), 33.0 );
                }
            }
        }
    }

    // Check scalar multiplication
    auto v = 2.0 * u;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                for ( int l = 0; l < 2; ++l )
                {
                    EXPECT_EQ( v( i, j, k, l ), 66.0 );
                }
            }
        }
    }

    // Check scalar division
    auto w = u / 2.0;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                for ( int l = 0; l < 2; ++l )
                {
                    EXPECT_EQ( w( i, j, k, l ), 16.5 );
                }
            }
        }
    }

    // Check addition assignment
    LinearAlgebra::Tensor4<double, 3, 4, 2, 2> tt = w;
    tt += u;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                for ( int l = 0; l < 2; ++l )
                {
                    EXPECT_DOUBLE_EQ( tt( i, j, k, l ), 49.5 );
                }
            }
        }
    }

    // Check subtraction assignment
    tt -= u;
    for ( int i = 0; i < 3; ++i )
    {
        for ( int j = 0; j < 4; ++j )
        {
            for ( int k = 0; k < 2; ++k )
            {
                for ( int l = 0; l < 2; ++l )
                {
                    EXPECT_DOUBLE_EQ( tt( i, j, k, l ), 16.5 );
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
void vectorTest()
{
    // Make a basic vector.
    LinearAlgebra::Vector<double, 3> x = { 1.2, -3.5, 5.4 };
    EXPECT_EQ( x.stride_0(), 1 );
    EXPECT_EQ( x.stride( 0 ), 1 );
    EXPECT_EQ( x.extent( 0 ), 3 );

    EXPECT_EQ( x( 0 ), 1.2 );
    EXPECT_EQ( x( 1 ), -3.5 );
    EXPECT_EQ( x( 2 ), 5.4 );

    // Check a vector view.
    LinearAlgebra::VectorView<double, 3> x_view( x.data(), x.stride_0() );
    EXPECT_EQ( x_view.stride_0(), 1 );
    EXPECT_EQ( x_view.stride( 0 ), 1 );
    EXPECT_EQ( x_view.extent( 0 ), 3 );

    EXPECT_EQ( x_view( 0 ), 1.2 );
    EXPECT_EQ( x_view( 1 ), -3.5 );
    EXPECT_EQ( x_view( 2 ), 5.4 );

    // Check a shallow copy
    auto x_c = x;
    EXPECT_EQ( x_c.stride_0(), 1 );
    EXPECT_EQ( x_c.stride( 0 ), 1 );
    EXPECT_EQ( x_c.extent( 0 ), 3 );

    EXPECT_EQ( x_c( 0 ), 1.2 );
    EXPECT_EQ( x_c( 1 ), -3.5 );
    EXPECT_EQ( x_c( 2 ), 5.4 );

    // Check scalar assignment and operator()
    x = 43.3;
    for ( int i = 0; i < 3; ++i )
    {
        EXPECT_EQ( x( i ), 43.3 );

        x( i ) = -10.2;
        EXPECT_EQ( x( i ), -10.2 );
    }

    // Check default initialization
    LinearAlgebra::Vector<double, 2> y;
    EXPECT_EQ( y.stride_0(), 1 );
    EXPECT_EQ( y.stride( 0 ), 1 );
    EXPECT_EQ( y.extent( 0 ), 2 );

    // Check scalar constructor.
    LinearAlgebra::Vector<double, 3> c = 32.3;
    for ( int i = 0; i < 3; ++i )
        EXPECT_EQ( c( i ), 32.3 );

    // Check scalar multiplication.
    auto d = 2.0 * c;
    for ( int i = 0; i < 3; ++i )
        EXPECT_EQ( d( i ), 64.6 );

    // Check scalar division.
    auto z = d / 2.0;
    for ( int i = 0; i < 3; ++i )
        EXPECT_EQ( z( i ), 32.3 );

    // Check cross product.
    LinearAlgebra::Vector<double, 3> e0 = { 1.0, 0.0, 0.0 };
    LinearAlgebra::Vector<double, 3> e1 = { 0.0, 1.0, 0.0 };
    auto e2 = e0 % e1;
    EXPECT_EQ( e2( 0 ), 0.0 );
    EXPECT_EQ( e2( 1 ), 0.0 );
    EXPECT_EQ( e2( 2 ), 1.0 );

    // Check element product.
    LinearAlgebra::Vector<double, 2> f = { 2.0, 1.0 };
    LinearAlgebra::Vector<double, 2> g = { 4.0, 2.0 };
    auto h = f & g;
    EXPECT_EQ( h( 0 ), 8.0 );
    EXPECT_EQ( h( 1 ), 2.0 );

    // Check element division.
    auto j = f | g;
    EXPECT_EQ( j( 0 ), 0.5 );
    EXPECT_EQ( j( 1 ), 0.5 );

    // Check addition assignment.
    LinearAlgebra::Vector<double, 3> q = z;
    q += d;
    for ( int i = 0; i < 3; ++i )
        EXPECT_DOUBLE_EQ( q( i ), 96.9 );

    // Check subtraction assignment.
    q -= d;
    for ( int i = 0; i < 3; ++i )
        EXPECT_DOUBLE_EQ( q( i ), 32.3 );

    // Check diagonal matrix construction.
    auto q_diag = LinearAlgebra::diagonal( q );
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            EXPECT_DOUBLE_EQ( q_diag( i, j ), ( i == j ) ? q( i ) : 0.0 );

    // Check deep copy.
    LinearAlgebra::Vector<double, 3> w;
    LinearAlgebra::deepCopy( w, q );
    for ( int i = 0; i < 3; ++i )
        EXPECT_DOUBLE_EQ( w( i ), 32.3 );

    // Size 1 vector test.
    // FIXME: construction of length 1 vector fails with NVCC.
    /*
    LinearAlgebra::Vector<double, 1> sov = 3.2;
    sov += ~sov * sov;
    EXPECT_DOUBLE_EQ( 13.44, sov( 0 ) );
    sov -= ~sov * sov;
    EXPECT_DOUBLE_EQ( -167.1936, sov( 0 ) );
    sov = 12.0;
    EXPECT_DOUBLE_EQ( 12.0, sov( 0 ) );
    LinearAlgebra::Vector<double, 1> sov2 = ~sov * sov;
    EXPECT_DOUBLE_EQ( 144.0, sov2( 0 ) );
    */
}

//---------------------------------------------------------------------------//
void quaternionTest()
{
    // Make a basic quaternion.
    LinearAlgebra::Quaternion<double> x = { 1.2, -3.5, 5.4, -2.4 };
    EXPECT_EQ( x.stride_0(), 1 );
    EXPECT_EQ( x.stride( 0 ), 1 );
    EXPECT_EQ( x.extent( 0 ), 4 );

    EXPECT_EQ( x( 0 ), 1.2 );
    EXPECT_EQ( x( 1 ), -3.5 );
    EXPECT_EQ( x( 2 ), 5.4 );
    EXPECT_EQ( x( 3 ), -2.4 );

    // Check scalar and vector parts
    auto x_vec = x.vector();
    EXPECT_EQ( x.scalar(), 1.2 );
    EXPECT_EQ( x_vec( 0 ), -3.5 );
    EXPECT_EQ( x_vec( 1 ), 5.4 );
    EXPECT_EQ( x_vec( 2 ), -2.4 );

    // Check scalar + vector constructor
    LinearAlgebra::Quaternion<double> xz = {
        3.4, LinearAlgebra::Vector<double, 3>{ 0.6, -3.7, 9.2 } };
    EXPECT_EQ( xz( 0 ), 3.4 );
    EXPECT_EQ( xz( 1 ), 0.6 );
    EXPECT_EQ( xz( 2 ), -3.7 );
    EXPECT_EQ( xz( 3 ), 9.2 );

    // Change the vector part
    // TODO: xz.vector() = x.vector() doesn't work.
    auto xz_vec = xz.vector();
    for ( int d = 0; d < 3; ++d )
        xz_vec( d ) = x_vec( d );

    // static_assert(std::is_same<decltype(x_vec), double>::value, "Types do not
    // match");

    EXPECT_EQ( xz( 0 ), 3.4 );
    EXPECT_EQ( xz( 1 ), -3.5 );
    EXPECT_EQ( xz( 2 ), 5.4 );
    EXPECT_EQ( xz( 3 ), -2.4 );

    // Check a quaternion view.
    LinearAlgebra::QuaternionView<double> x_view( x.data(), x.stride_0() );
    EXPECT_EQ( x_view.stride_0(), 1 );
    EXPECT_EQ( x_view.stride( 0 ), 1 );
    EXPECT_EQ( x_view.extent( 0 ), 4 );

    EXPECT_EQ( x_view( 0 ), 1.2 );
    EXPECT_EQ( x_view( 1 ), -3.5 );
    EXPECT_EQ( x_view( 2 ), 5.4 );
    EXPECT_EQ( x_view( 3 ), -2.4 );

    // Check a shallow copy
    auto x_c = x;
    EXPECT_EQ( x_c.stride_0(), 1 );
    EXPECT_EQ( x_c.stride( 0 ), 1 );
    EXPECT_EQ( x_c.extent( 0 ), 4 );

    EXPECT_EQ( x_c( 0 ), 1.2 );
    EXPECT_EQ( x_c( 1 ), -3.5 );
    EXPECT_EQ( x_c( 2 ), 5.4 );
    EXPECT_EQ( x_c( 3 ), -2.4 );

    // Check scalar assignment and operator()
    x = 43.3;
    for ( int i = 0; i < 4; ++i )
    {
        EXPECT_EQ( x( i ), 43.3 );

        x( i ) = -10.2;
        EXPECT_EQ( x( i ), -10.2 );
    }

    // Check default initialization
    LinearAlgebra::Quaternion<double> y;
    EXPECT_EQ( y.stride_0(), 1 );
    EXPECT_EQ( y.stride( 0 ), 1 );
    EXPECT_EQ( y.extent( 0 ), 4 );

    // Check scalar constructor.
    LinearAlgebra::Quaternion<double> c = 32.3;
    for ( int i = 0; i < 4; ++i )
        EXPECT_EQ( c( i ), 32.3 );

    // Check scalar multiplication.
    auto d = 2.0 * c;
    for ( int i = 0; i < 4; ++i )
        EXPECT_EQ( d( i ), 64.6 );

    // Check scalar division.
    auto z = d / 2.0;
    for ( int i = 0; i < 4; ++i )
        EXPECT_EQ( z( i ), 32.3 );

    // Check quaternion-quaternion product.
    LinearAlgebra::Quaternion<double> f = { 1.0, 3.0, 2.0, 4.0 };
    LinearAlgebra::Quaternion<double> g = { 2.0, 3.0, 4.0, 5.0 };
    auto h = f & g;
    EXPECT_EQ( h( 0 ), -35.0 );
    EXPECT_EQ( h( 1 ), 3.0 );
    EXPECT_EQ( h( 2 ), 5.0 );
    EXPECT_EQ( h( 3 ), 19.0 );

    // Check quaternion-quaternion division.
    double eps = 1e-12;
    auto j = f | g;
    EXPECT_NEAR( j( 0 ), 13.0 / 18.0, eps );
    EXPECT_NEAR( j( 1 ), 1.0 / 6.0, eps );
    EXPECT_NEAR( j( 2 ), 1.0 / 18.0, eps );
    EXPECT_NEAR( j( 3 ), -1.0 / 18.0, eps );

    // Check addition assignment.
    LinearAlgebra::Quaternion<double> q = z;
    q += d;
    for ( int i = 0; i < 3; ++i )
        EXPECT_DOUBLE_EQ( q( i ), 96.9 );

    // Check subtraction assignment.
    q -= d;
    for ( int i = 0; i < 3; ++i )
        EXPECT_DOUBLE_EQ( q( i ), 32.3 );

    // Check deep copy.
    LinearAlgebra::Quaternion<double> w;
    LinearAlgebra::deepCopy( w, q );
    for ( int i = 0; i < 4; ++i )
        EXPECT_DOUBLE_EQ( w( i ), 32.3 );
}

void quaMatRotTest()
{
    double pi = 4.0 * Kokkos::atan( 1.0 );
    double phi = pi / 2.0;

    LinearAlgebra::Quaternion<double> q = { Kokkos::cos( phi / 2.0 ), 0.0, 0.0,
                                            Kokkos::sin( phi / 2.0 ) };

    // Convert the quaternion to its equivalent rotation matrix
    auto rot_mat = static_cast<Mat3<double>>( q );

    // Unit basis vector along x-axis
    Vec3<double> e1 = { 1.0, 0.0, 0.0 };

    // Apply the quaternion rotation matrix and check the result
    auto e1_rotated = rot_mat * e1;

    double eps = 1e-15;

    EXPECT_NEAR( e1_rotated( 0 ), 0.0, eps );
    EXPECT_NEAR( e1_rotated( 1 ), 1.0, eps );
    EXPECT_NEAR( e1_rotated( 2 ), 0.0, eps );

    auto e2 = rot_mat * e1_rotated;

    // Now test a composition of rotations
    auto q2 = q & q;

    auto pi_rot_mat = static_cast<Mat3<double>>( q2 );

    auto e2_q = pi_rot_mat * e1;

    EXPECT_NEAR( e2_q( 0 ), -1.0, eps );
    EXPECT_NEAR( e2_q( 1 ), 0.0, eps );
    EXPECT_NEAR( e2_q( 2 ), 0.0, eps );

    EXPECT_NEAR( e2_q( 0 ), e2( 0 ), eps );
    EXPECT_NEAR( e2_q( 1 ), e2( 1 ), eps );
    EXPECT_NEAR( e2_q( 2 ), e2( 2 ), eps );

    // Now test vector rotation via direct quaternion-vector conjugation
    LinearAlgebra::Quaternion<double> p = { 0.0, e1( 0 ), e1( 1 ), e1( 2 ) };

    // Perform the conjugation
    auto p_rot = ( q & p ) & ~q;

    // The vector part of the p_rot quaternion corresponds to the rotated vector
    EXPECT_NEAR( p_rot( 1 ), e1_rotated( 0 ), eps );
    EXPECT_NEAR( p_rot( 2 ), e1_rotated( 1 ), eps );
    EXPECT_NEAR( p_rot( 3 ), e1_rotated( 2 ), eps );

    // Perform a quaternion-matrix conjugation
    Mat3<double> I;
    LinearAlgebra::identity( I );

    auto I_rot = I & q;

    EXPECT_NEAR( I_rot( 0, 0 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 0, 1 ), 1.0, eps );
    EXPECT_NEAR( I_rot( 0, 2 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 1, 0 ), -1.0, eps );
    EXPECT_NEAR( I_rot( 1, 1 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 1, 2 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 2, 0 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 2, 1 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 2, 2 ), 1.0, eps );

    // Test if the matrix-representation of q times I is the same result
    I_rot = ~rot_mat * I;

    EXPECT_NEAR( I_rot( 0, 0 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 0, 1 ), 1.0, eps );
    EXPECT_NEAR( I_rot( 0, 2 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 1, 0 ), -1.0, eps );
    EXPECT_NEAR( I_rot( 1, 1 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 1, 2 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 2, 0 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 2, 1 ), 0.0, eps );
    EXPECT_NEAR( I_rot( 2, 2 ), 1.0, eps );
}

//---------------------------------------------------------------------------//
void viewTest()
{
    double m[2][3] = { { 1.2, -3.5, 5.4 }, { 8.6, 2.6, -0.1 } };
    LinearAlgebra::MatrixView<double, 2, 3> a( &m[0][0], 3, 1 );
    EXPECT_EQ( a.stride_0(), 3 );
    EXPECT_EQ( a.stride_1(), 1 );
    EXPECT_EQ( a.stride( 0 ), 3 );
    EXPECT_EQ( a.stride( 1 ), 1 );
    EXPECT_EQ( a.extent( 0 ), 2 );
    EXPECT_EQ( a.extent( 1 ), 3 );

    EXPECT_EQ( a( 0, 0 ), 1.2 );
    EXPECT_EQ( a( 0, 1 ), -3.5 );
    EXPECT_EQ( a( 0, 2 ), 5.4 );
    EXPECT_EQ( a( 1, 0 ), 8.6 );
    EXPECT_EQ( a( 1, 1 ), 2.6 );
    EXPECT_EQ( a( 1, 2 ), -0.1 );

    double v[6] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };

    LinearAlgebra::VectorView<double, 6> x1( &v[0], 1 );
    EXPECT_EQ( x1.stride_0(), 1 );
    EXPECT_EQ( x1.stride( 0 ), 1 );
    EXPECT_EQ( x1.extent( 0 ), 6 );
    for ( int i = 0; i < 6; ++i )
        EXPECT_EQ( x1( i ), 1.0 * i );

    LinearAlgebra::VectorView<double, 3> x2( &v[0], 2 );
    EXPECT_EQ( x2.stride_0(), 2 );
    EXPECT_EQ( x2.stride( 0 ), 2 );
    EXPECT_EQ( x2.extent( 0 ), 3 );
    for ( int i = 0; i < 3; ++i )
        EXPECT_EQ( x2( i ), 2.0 * i );

    LinearAlgebra::VectorView<double, 2> x3( &v[1], 3 );
    EXPECT_EQ( x3.stride_0(), 3 );
    EXPECT_EQ( x2.stride( 0 ), 2 );
    EXPECT_EQ( x3.extent( 0 ), 2 );
    for ( int i = 0; i < 2; ++i )
        EXPECT_EQ( x3( i ), 1.0 + 3.0 * i );
}

//---------------------------------------------------------------------------//
void matAddTest()
{
    LinearAlgebra::Matrix<double, 1, 2> a = { { 2.0, 1.0 } };
    LinearAlgebra::Matrix<double, 1, 2> b = { { 2.0, 3.0 } };

    auto c = a + b;
    EXPECT_EQ( c.extent( 0 ), 1 );
    EXPECT_EQ( c.extent( 1 ), 2 );
    EXPECT_EQ( c( 0, 0 ), 4.0 );
    EXPECT_EQ( c( 0, 1 ), 4.0 );

    auto d = ~a + ~b;
    EXPECT_EQ( d.extent( 0 ), 2 );
    EXPECT_EQ( d.extent( 1 ), 1 );
    EXPECT_EQ( d( 0, 0 ), 4.0 );
    EXPECT_EQ( d( 1, 0 ), 4.0 );

    LinearAlgebra::Matrix<double, 2, 1> e = { { 2.0 }, { 3.0 } };

    auto f = ~a + e;
    EXPECT_EQ( f.extent( 0 ), 2 );
    EXPECT_EQ( f.extent( 1 ), 1 );
    EXPECT_EQ( f( 0, 0 ), 4.0 );
    EXPECT_EQ( f( 1, 0 ), 4.0 );

    auto g = a + ~e;
    EXPECT_EQ( g.extent( 0 ), 1 );
    EXPECT_EQ( g.extent( 1 ), 2 );
    EXPECT_EQ( g( 0, 0 ), 4.0 );
    EXPECT_EQ( g( 0, 1 ), 4.0 );
}

//---------------------------------------------------------------------------//
void matSubTest()
{
    LinearAlgebra::Matrix<double, 1, 2> a = { { 2.0, 1.0 } };
    LinearAlgebra::Matrix<double, 1, 2> b = { { 2.0, 3.0 } };

    auto c = a - b;
    EXPECT_EQ( c.extent( 0 ), 1 );
    EXPECT_EQ( c.extent( 1 ), 2 );
    EXPECT_EQ( c( 0, 0 ), 0.0 );
    EXPECT_EQ( c( 0, 1 ), -2.0 );

    auto d = ~a - ~b;
    EXPECT_EQ( d.extent( 0 ), 2 );
    EXPECT_EQ( d.extent( 1 ), 1 );
    EXPECT_EQ( d( 0, 0 ), 0.0 );
    EXPECT_EQ( d( 1, 0 ), -2.0 );

    LinearAlgebra::Matrix<double, 2, 1> e = { { 2.0 }, { 3.0 } };

    auto f = ~a - e;
    EXPECT_EQ( f.extent( 0 ), 2 );
    EXPECT_EQ( f.extent( 1 ), 1 );
    EXPECT_EQ( f( 0, 0 ), 0.0 );
    EXPECT_EQ( f( 1, 0 ), -2.0 );

    auto g = a - ~e;
    EXPECT_EQ( g.extent( 0 ), 1 );
    EXPECT_EQ( g.extent( 1 ), 2 );
    EXPECT_EQ( g( 0, 0 ), 0.0 );
    EXPECT_EQ( g( 0, 1 ), -2.0 );
}

//---------------------------------------------------------------------------//
void matMatTest()
{
    // Square test.
    LinearAlgebra::Matrix<double, 2, 2> a = { { 2.0, 1.0 }, { 2.0, 1.0 } };
    LinearAlgebra::Matrix<double, 2, 2> b = { { 2.0, 3.0 }, { 2.0, -1.0 } };

    auto c = a * b;
    EXPECT_EQ( c.extent( 0 ), 2 );
    EXPECT_EQ( c.extent( 1 ), 2 );
    EXPECT_EQ( c( 0, 0 ), 6.0 );
    EXPECT_EQ( c( 0, 1 ), 5.0 );
    EXPECT_EQ( c( 1, 0 ), 6.0 );
    EXPECT_EQ( c( 1, 1 ), 5.0 );

    c = ~a * b;
    EXPECT_EQ( c( 0, 0 ), 8.0 );
    EXPECT_EQ( c( 0, 1 ), 4.0 );
    EXPECT_EQ( c( 1, 0 ), 4.0 );
    EXPECT_EQ( c( 1, 1 ), 2.0 );

    c = a * ~b;
    EXPECT_EQ( c( 0, 0 ), 7.0 );
    EXPECT_EQ( c( 0, 1 ), 3.0 );
    EXPECT_EQ( c( 1, 0 ), 7.0 );
    EXPECT_EQ( c( 1, 1 ), 3.0 );

    c = ~a * ~b;
    EXPECT_EQ( c( 0, 0 ), 10.0 );
    EXPECT_EQ( c( 0, 1 ), 2.0 );
    EXPECT_EQ( c( 1, 0 ), 5.0 );
    EXPECT_EQ( c( 1, 1 ), 1.0 );

    // Non square test.
    LinearAlgebra::Matrix<double, 2, 1> f = { { 3.0 }, { 1.0 } };
    LinearAlgebra::Matrix<double, 1, 2> g = { { 2.0, 1.0 } };

    auto h = f * g;
    EXPECT_EQ( h.extent( 0 ), 2 );
    EXPECT_EQ( h.extent( 1 ), 2 );
    EXPECT_EQ( h( 0, 0 ), 6.0 );
    EXPECT_EQ( h( 0, 1 ), 3.0 );
    EXPECT_EQ( h( 1, 0 ), 2.0 );
    EXPECT_EQ( h( 1, 1 ), 1.0 );

    auto j = f * ~f;
    EXPECT_EQ( j.extent( 0 ), 2 );
    EXPECT_EQ( j.extent( 1 ), 2 );
    EXPECT_EQ( j( 0, 0 ), 9.0 );
    EXPECT_EQ( j( 0, 1 ), 3.0 );
    EXPECT_EQ( j( 1, 0 ), 3.0 );
    EXPECT_EQ( j( 1, 1 ), 1.0 );

    auto k = ~f * f;
    EXPECT_EQ( k.extent( 0 ), 1 );
    EXPECT_EQ( k.extent( 1 ), 1 );
    EXPECT_EQ( k( 0, 0 ), 10.0 );

    auto m = ~f * ~g;
    EXPECT_EQ( m.extent( 0 ), 1 );
    EXPECT_EQ( m.extent( 1 ), 1 );
    EXPECT_EQ( m( 0, 0 ), 7.0 );
}

//---------------------------------------------------------------------------//
void matVecTest()
{
    // Square test.
    LinearAlgebra::Matrix<double, 2, 2> a = { { 3.0, 2.0 }, { 1.0, 2.0 } };
    LinearAlgebra::Vector<double, 2> x = { 3.0, 1.0 };

    auto y = a * x;
    EXPECT_EQ( y.extent( 0 ), 2 );
    EXPECT_EQ( y( 0 ), 11.0 );
    EXPECT_EQ( y( 1 ), 5.0 );

    y = ~a * x;
    EXPECT_EQ( y( 0 ), 10.0 );
    EXPECT_EQ( y( 1 ), 8.0 );

    auto b = ~x * a;
    EXPECT_EQ( b.extent( 0 ), 1 );
    EXPECT_EQ( b.extent( 1 ), 2 );
    EXPECT_EQ( b( 0, 0 ), 10.0 );
    EXPECT_EQ( b( 0, 1 ), 8.0 );

    b = ~x * ~a;
    EXPECT_EQ( b( 0, 0 ), 11.0 );
    EXPECT_EQ( b( 0, 1 ), 5.0 );

    // Non square test.
    LinearAlgebra::Matrix<double, 1, 2> c = { { 1.0, 2.0 } };
    LinearAlgebra::Vector<double, 2> f = { 3.0, 2.0 };

    auto g = c * f;
    EXPECT_EQ( g.extent( 0 ), 1 );
    EXPECT_EQ( g( 0 ), 7.0 );

    auto h = ~f * ~c;
    EXPECT_EQ( h.extent( 0 ), 1 );
    EXPECT_EQ( h.extent( 1 ), 1 );
    EXPECT_EQ( h( 0, 0 ), 7.0 );

    LinearAlgebra::Matrix<double, 2, 1> j = { { 1.0 }, { 2.0 } };

    auto k = ~j * f;
    EXPECT_EQ( k.extent( 0 ), 1 );
    EXPECT_EQ( k( 0 ), 7.0 );

    auto l = ~f * j;
    EXPECT_EQ( l.extent( 0 ), 1 );
    EXPECT_EQ( k( 0 ), 7.0 );
}

//---------------------------------------------------------------------------//
void vecAddTest()
{
    LinearAlgebra::Vector<double, 2> a = { 2.0, 1.0 };
    LinearAlgebra::Vector<double, 2> b = { 2.0, 3.0 };

    auto c = a + b;
    EXPECT_EQ( c( 0 ), 4.0 );
    EXPECT_EQ( c( 1 ), 4.0 );
}

//---------------------------------------------------------------------------//
void vecSubTest()
{
    LinearAlgebra::Vector<double, 2> a = { 2.0, 1.0 };
    LinearAlgebra::Vector<double, 2> b = { 2.0, 3.0 };

    auto c = a - b;
    EXPECT_EQ( c( 0 ), 0.0 );
    EXPECT_EQ( c( 1 ), -2.0 );
}

//---------------------------------------------------------------------------//
void vecVecTest()
{
    LinearAlgebra::Vector<double, 2> x = { 1.0, 2.0 };
    LinearAlgebra::Vector<double, 2> y = { 2.0, 3.0 };

    auto dot = ~x * y;
    EXPECT_EQ( dot.extent( 0 ), 1 );
    EXPECT_EQ( dot.extent( 1 ), 1 );
    EXPECT_EQ( dot, 8.0 );

    auto inner = x * ~y;
    EXPECT_EQ( inner.extent( 0 ), 2 );
    EXPECT_EQ( inner.extent( 1 ), 2 );
    EXPECT_EQ( inner( 0, 0 ), 2.0 );
    EXPECT_EQ( inner( 0, 1 ), 3.0 );
    EXPECT_EQ( inner( 1, 0 ), 4.0 );
    EXPECT_EQ( inner( 1, 1 ), 6.0 );
}

//---------------------------------------------------------------------------//
void expressionTest()
{
    LinearAlgebra::Matrix<double, 2, 2> a = { { 2.0, 1.0 }, { 2.0, 1.0 } };
    LinearAlgebra::Matrix<double, 2, 2> i = { { 1.0, 0.0 }, { 0.0, 1.0 } };
    LinearAlgebra::Vector<double, 2> x = { 1.0, 2.0 };

    auto op1 = a + ~a;
    EXPECT_EQ( op1( 0, 0 ), 4.0 );
    EXPECT_EQ( op1( 0, 1 ), 3.0 );
    EXPECT_EQ( op1( 1, 0 ), 3.0 );
    EXPECT_EQ( op1( 1, 1 ), 2.0 );

    auto op2 = 0.5 * ( a + ~a );
    EXPECT_EQ( op2( 0, 0 ), 2.0 );
    EXPECT_EQ( op2( 0, 1 ), 1.5 );
    EXPECT_EQ( op2( 1, 0 ), 1.5 );
    EXPECT_EQ( op2( 1, 1 ), 1.0 );

    auto op3 = 0.5 * ( a + ~a ) * i;
    EXPECT_EQ( op3( 0, 0 ), 2.0 );
    EXPECT_EQ( op3( 0, 1 ), 1.5 );
    EXPECT_EQ( op3( 1, 0 ), 1.5 );
    EXPECT_EQ( op3( 1, 1 ), 1.0 );

    auto op4 = x * ~x;
    EXPECT_EQ( op4( 0, 0 ), 1.0 );
    EXPECT_EQ( op4( 0, 1 ), 2.0 );
    EXPECT_EQ( op4( 1, 0 ), 2.0 );
    EXPECT_EQ( op4( 1, 1 ), 4.0 );

    auto op5 = 0.5 * ( a + ~a ) * i + ( x * ~x );
    EXPECT_EQ( op5( 0, 0 ), 3.0 );
    EXPECT_EQ( op5( 0, 1 ), 3.5 );
    EXPECT_EQ( op5( 1, 0 ), 3.5 );
    EXPECT_EQ( op5( 1, 1 ), 5.0 );
}

//---------------------------------------------------------------------------//
template <int N>
void linearSolveTest()
{
    LinearAlgebra::Matrix<double, N, N> A;
    LinearAlgebra::Vector<double, N> x0;

    std::default_random_engine engine( 349305 );
    std::uniform_real_distribution<double> dist( 0.0, 1.0 );
    for ( int i = 0; i < N; ++i )
        x0( i ) = dist( engine );
    for ( int i = 0; i < N; ++i )
        for ( int j = 0; j < N; ++j )
            A( i, j ) = dist( engine );

    double eps = 1.0e-12;

    auto b = A * x0;
    auto x1 = A ^ b;
    for ( int i = 0; i < N; ++i )
        EXPECT_NEAR( x0( i ), x1( i ), eps );

    auto c = ~A * x0;
    auto x2 = ~A ^ c;
    for ( int i = 0; i < N; ++i )
        EXPECT_NEAR( x0( i ), x2( i ), eps );
}

//---------------------------------------------------------------------------//
void matrixExponentialTest()
{
    // Test a random 3x3 matrix exponentiation
    LinearAlgebra::Matrix<double, 3, 3> A;
    A = { { 0.261653840929, 0.276910681439, 0.374636284772 },
          { 0.024295417821, 0.570296571637, 0.431548657634 },
          { 0.535801487928, 0.192472912864, 0.948361671535 } };

    auto A_exp = LinearAlgebra::exponential( A );

    double eps = 1.0e-12;

    EXPECT_NEAR( A_exp( 0, 0 ), 1.493420196372, eps );
    EXPECT_NEAR( A_exp( 0, 1 ), 0.513684901522, eps );
    EXPECT_NEAR( A_exp( 0, 2 ), 0.848390478207, eps );
    EXPECT_NEAR( A_exp( 1, 0 ), 0.255994538136, eps );
    EXPECT_NEAR( A_exp( 1, 1 ), 1.880234291898, eps );
    EXPECT_NEAR( A_exp( 1, 2 ), 0.981626296936, eps );
    EXPECT_NEAR( A_exp( 2, 0 ), 1.057456162099, eps );
    EXPECT_NEAR( A_exp( 2, 1 ), 0.57315415314, eps );
    EXPECT_NEAR( A_exp( 2, 2 ), 2.914674968894, eps );

    // Test the exp(0) = 1 identity (for matrices)
    LinearAlgebra::Matrix<double, 2, 2> B_zeros;
    B_zeros = 0.0;

    auto B_exp = LinearAlgebra::exponential( B_zeros );

    EXPECT_NEAR( B_exp( 0, 0 ), 1.0, eps );
    EXPECT_NEAR( B_exp( 0, 1 ), 0.0, eps );
    EXPECT_NEAR( B_exp( 1, 0 ), 0.0, eps );
    EXPECT_NEAR( B_exp( 1, 1 ), 1.0, eps );
}

//---------------------------------------------------------------------------//
template <int N>
void kernelTest()
{
    int size = 10;
    Kokkos::View<double* [N][N], Kokkos::LayoutLeft, TEST_MEMSPACE> view_a(
        "a", size );
    Kokkos::View<double* [N], Kokkos::LayoutRight, TEST_MEMSPACE> view_x0(
        "x0", size );
    Kokkos::View<double* [N], Kokkos::LayoutLeft, TEST_MEMSPACE> view_x1(
        "x1", size );
    Kokkos::View<double* [N], Kokkos::LayoutRight, TEST_MEMSPACE> view_x2(
        "x2", size );

    Kokkos::Random_XorShift64_Pool<TEST_EXECSPACE> pool( 3923423 );
    Kokkos::fill_random( view_a, pool, 1.0 );
    Kokkos::fill_random( view_x0, pool, 1.0 );

    Kokkos::parallel_for(
        "test_la_kernel", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, size ),
        KOKKOS_LAMBDA( const int i ) {
            // Get views.
            LinearAlgebra::MatrixView<double, N, N> A_v(
                &view_a( i, 0, 0 ), view_a.stride_1(), view_a.stride_2() );
            LinearAlgebra::VectorView<double, N> x0_v( &view_x0( i, 0 ),
                                                       view_x0.stride_1() );
            LinearAlgebra::VectorView<double, N> x1_v( &view_x1( i, 0 ),
                                                       view_x1.stride_1() );
            LinearAlgebra::VectorView<double, N> x2_v( &view_x2( i, 0 ),
                                                       view_x2.stride_1() );

            // Gather.
            typename decltype( A_v )::copy_type A = A_v;
            typename decltype( x0_v )::copy_type x0 = x0_v;
            typename decltype( x1_v )::copy_type x1 = x1_v;
            typename decltype( x2_v )::copy_type x2 = x2_v;

            // Create a composite operator via an expression.
            auto op = 0.75 * ( A + 0.5 * ~A );

            // Do work.
            auto b = op * x0;
            x1 = op ^ b;

            auto c = ~op * x0;
            x2 = ~op ^ c;

            // Scatter
            x1_v = x1;
            x2_v = x2;
        } );

    auto x0_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view_x0 );
    auto x1_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view_x1 );
    auto x2_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), view_x2 );

    double eps = 1.0e-11;
    for ( int i = 0; i < size; ++i )
        for ( int d = 0; d < N; ++d )
        {
            EXPECT_NEAR( x0_host( i, d ), x1_host( i, d ), eps );
            EXPECT_NEAR( x0_host( i, d ), x2_host( i, d ), eps );
        }
}

//---------------------------------------------------------------------------//
/*
void eigendecompositionTest()
{
    LinearAlgebra::Matrix<double, 4, 4> A = {
        { 1, 2, 3, 4 }, { 1, 4, 3, 2 }, { 1, 1, 2, 1 }, { 2, 1, 4, 1 } };

    LinearAlgebra::Vector<double, 4> e_real;
    LinearAlgebra::Vector<double, 4> e_imag;
    LinearAlgebra::Matrix<double, 4, 4> u_left;
    LinearAlgebra::Matrix<double, 4, 4> u_right;

    LinearAlgebra::eigendecomposition( A, e_real, e_imag, u_left, u_right );

    // Check real eigenvalues
    EXPECT_FLOAT_EQ( 7.699393945291155, e_real( 0 ) );
    EXPECT_FLOAT_EQ( 2.045317716995864, e_real( 1 ) );
    EXPECT_FLOAT_EQ( -1.584394776530825, e_real( 2 ) );
    EXPECT_FLOAT_EQ( -0.1603168857561950, e_real( 3 ) );

    // No complex eigenvalues
    EXPECT_FLOAT_EQ( 0.0, e_imag( 0 ) );
    EXPECT_FLOAT_EQ( 0.0, e_imag( 1 ) );
    EXPECT_FLOAT_EQ( 0.0, e_imag( 2 ) );
    EXPECT_FLOAT_EQ( 0.0, e_imag( 3 ) );

    // Check right eigenvector.
    EXPECT_FLOAT_EQ( -0.5772831579170088, -u_right( 0, 0 ) );
    EXPECT_FLOAT_EQ( -0.6262445857026205, -u_right( 1, 0 ) );
    EXPECT_FLOAT_EQ( -0.2879754733142472, -u_right( 2, 0 ) );
    EXPECT_FLOAT_EQ( -0.4377579253799459, -u_right( 3, 0 ) );

    EXPECT_FLOAT_EQ( -0.3935576398313519, -u_right( 0, 1 ) );
    EXPECT_FLOAT_EQ( 0.8091338559680495, -u_right( 1, 1 ) );
    EXPECT_FLOAT_EQ( -0.1154744868351802, -u_right( 2, 1 ) );
    EXPECT_FLOAT_EQ( -0.4208092562513370, -u_right( 3, 1 ) );

    EXPECT_FLOAT_EQ( -0.8496726353484932, -u_right( 0, 2 ) );
    EXPECT_FLOAT_EQ( -0.09431174848941145, -u_right( 1, 2 ) );
    EXPECT_FLOAT_EQ( 0.1227267748072847, -u_right( 2, 2 ) );
    EXPECT_FLOAT_EQ( 0.5040831732781987, -u_right( 3, 2 ) );

    EXPECT_FLOAT_EQ( 0.8702121374980452, -u_right( 0, 3 ) );
    EXPECT_FLOAT_EQ( 0.1109808921593327, -u_right( 1, 3 ) );
    EXPECT_FLOAT_EQ( -0.4773906708806565, -u_right( 2, 3 ) );
    EXPECT_FLOAT_EQ( 0.05012209774858267, -u_right( 3, 3 ) );

    LinearAlgebra::Matrix<double, 4, 4> identity = 0.0;
    for ( int i = 0; i < 4; ++i )
        identity( i, i ) = 1.0;

    // Check right eigenvalue identity.
    for ( int i = 0; i < 4; ++i )
    {
        auto ur_ident = ( A - e_real( i ) * identity ) * u_right.column( i );
        for ( int j = 0; j < 4; ++j )
        {
            EXPECT_FLOAT_EQ( 1.0, 1.0 - ur_ident( j ) );
        }
    }

    // Check left eigenvalue identity.
    for ( int i = 0; i < 4; ++i )
    {
        auto ul_ident = ( ~A - e_real( i ) * identity ) * u_left.row( i );
        for ( int j = 0; j < 4; ++j )
        {
            EXPECT_FLOAT_EQ( 1.0, 1.0 - ul_ident( j ) );
        }
    }

    // Check inverse of right eigenvectors.
    auto ur_inv = LinearAlgebra::inverse( u_right );

    EXPECT_FLOAT_EQ( -0.3525146608046934, -ur_inv( 0, 0 ) );
    EXPECT_FLOAT_EQ( -0.5469292354121372, -ur_inv( 0, 1 ) );
    EXPECT_FLOAT_EQ( -0.8218494119447979, -ur_inv( 0, 2 ) );
    EXPECT_FLOAT_EQ( -0.4964279684321965, -ur_inv( 0, 3 ) );

    EXPECT_FLOAT_EQ( -0.3648950659157645, -ur_inv( 1, 0 ) );
    EXPECT_FLOAT_EQ( 0.8089980125933259, -ur_inv( 1, 1 ) );
    EXPECT_FLOAT_EQ( -0.5126586109009551, -ur_inv( 1, 2 ) );
    EXPECT_FLOAT_EQ( -0.3388853394692466, -ur_inv( 1, 3 ) );

    EXPECT_FLOAT_EQ( -0.6246987297746988, -ur_inv( 2, 0 ) );
    EXPECT_FLOAT_EQ( 0.1823766827069634, -ur_inv( 2, 1 ) );
    EXPECT_FLOAT_EQ( -0.9702233208689864, -ur_inv( 2, 2 ) );
    EXPECT_FLOAT_EQ( 1.201157386148598, -ur_inv( 2, 3 ) );

    EXPECT_FLOAT_EQ( 0.1403135639951463, -ur_inv( 3, 0 ) );
    EXPECT_FLOAT_EQ( 0.1811222598704855, -ur_inv( 3, 1 ) );
    EXPECT_FLOAT_EQ( -1.724375790756904, -ur_inv( 3, 2 ) );
    EXPECT_FLOAT_EQ( 0.6902226666411088, -ur_inv( 3, 3 ) );

    // Check the right eigenvector operator identity.
    LinearAlgebra::Matrix<double, 4, 4> lambda = 0.0;
    for ( int i = 0; i < 4; ++i )
        lambda( i, i ) = e_real( i );
    auto r_op_ident = u_right * lambda * ur_inv;
    for ( int i = 0; i < 4; ++i )
        for ( int j = 0; j < 4; ++j )
            EXPECT_FLOAT_EQ( A( i, j ), r_op_ident( i, j ) );

    // Check inverse of left eigenvectors.
    auto ul_inv = LinearAlgebra::inverse( u_left );

    // Check the left eigenvector operator identity.
    auto l_op_ident = ~( ~u_left * lambda * ~ul_inv );
    for ( int i = 0; i < 4; ++i )
        for ( int j = 0; j < 4; ++j )
            EXPECT_FLOAT_EQ( A( i, j ), l_op_ident( i, j ) );
}
*/

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, matrix_test ) { matrixTest(); }

TEST( TEST_CATEGORY, vector_test ) { vectorTest(); }

TEST( TEST_CATEGORY, quaternion_test ) { quaternionTest(); }

TEST( TEST_CATEGORY, quaMatRot_test ) { quaMatRotTest(); }

TEST( TEST_CATEGORY, tensor3_test ) { tensor3Test(); }

TEST( TEST_CATEGORY, tensor4_test ) { tensor4Test(); }

TEST( TEST_CATEGORY, view_test ) { viewTest(); }

TEST( TEST_CATEGORY, matadd_test ) { matAddTest(); }

TEST( TEST_CATEGORY, matsub_test ) { matSubTest(); }

TEST( TEST_CATEGORY, matmat_test ) { matMatTest(); }

TEST( TEST_CATEGORY, matVec_test ) { matVecTest(); }

TEST( TEST_CATEGORY, vecsub_test ) { vecSubTest(); }

TEST( TEST_CATEGORY, vecvec_test ) { vecVecTest(); }

TEST( TEST_CATEGORY, vecVec_test ) { vecVecTest(); }

TEST( TEST_CATEGORY, expression_test ) { expressionTest(); }

TEST( TEST_CATEGORY, linearSolve_test )
{
    linearSolveTest<2>();
    linearSolveTest<2>();
    linearSolveTest<4>();
    linearSolveTest<10>();
    linearSolveTest<20>();
}

TEST( TEST_CATEGORY, kernelTest )
{
    kernelTest<2>();
    kernelTest<3>();
    kernelTest<4>();
    kernelTest<10>();
    kernelTest<20>();
}

TEST( TEST_CATEGORY, matrixExponential_test ) { matrixExponentialTest(); }

// FIXME_KOKKOSKERNELS
// TEST( TEST_CATEGORY, eigendecomposition_test ) { eigendecompositionTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
