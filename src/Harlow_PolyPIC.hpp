#ifndef HARLOW_POLYPIC_HPP
#define HARLOW_POLYPIC_HPP

#include <Cajita.hpp>

#include <Harlow_DenseLinearAlgebra.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <type_traits>
#include <cmath>

namespace Harlow
{
//---------------------------------------------------------------------------//
// Polynomial Particle-in-Cell
//---------------------------------------------------------------------------//
namespace PolyPIC
{
//---------------------------------------------------------------------------//
// Grid-to-Particle
//---------------------------------------------------------------------------//
// PolyPIC modal weights with a quadratic particle polynomial basis. Note that
// these weights are computed with respect to the logical distance between the
// particle and the node

// Mode 0.
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights( std::integral_constant<int,0>,
                         const Scalar[3],
                         const Scalar,
                         const Scalar wi[3],
                         const Scalar wj[3],
                         const Scalar wk[3],
                         const int i,
                         const int j,
                         const int k,
                         const int,
                         const int,
                         const int,
                         Scalar* mode_weights )
{
    mode_weights[0] = wi[i] * wj[j] * wk[k];
}

// Mode 1.
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights( std::integral_constant<int,1>,
                         const Scalar distance[3],
                         const Scalar rdx_2,
                         const Scalar wi[3],
                         const Scalar wj[3],
                         const Scalar wk[3],
                         const int i,
                         const int j,
                         const int k,
                         const int mod_i,
                         const int mod_j,
                         const int mod_k,
                         Scalar* mode_weights )
{
    polyPicModeWeights( std::integral_constant<int,0>(),
        distance, rdx_2, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
    mode_weights[1] = mode_weights[0] * distance[Dim::I] * 4.0 * rdx_2;
}

// Mode 2.
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights( std::integral_constant<int,2>,
                         const Scalar distance[3],
                         const Scalar rdx_2,
                         const Scalar wi[3],
                         const Scalar wj[3],
                         const Scalar wk[3],
                         const int i,
                         const int j,
                         const int k,
                         const int mod_i,
                         const int mod_j,
                         const int mod_k,
                         Scalar* mode_weights )
{
    polyPicModeWeights( std::integral_constant<int,1>(),
        distance, rdx_2, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
    mode_weights[2] = mode_weights[0] * distance[Dim::J] * 4.0 * rdx_2;
}

// Mode 3.
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights( std::integral_constant<int,3>,
                         const Scalar distance[3],
                         const Scalar rdx_2,
                         const Scalar wi[3],
                         const Scalar wj[3],
                         const Scalar wk[3],
                         const int i,
                         const int j,
                         const int k,
                         const int mod_i,
                         const int mod_j,
                         const int mod_k,
                         Scalar* mode_weights )
{
    polyPicModeWeights( std::integral_constant<int,2>(),
        distance, rdx_2, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
    mode_weights[3] = mode_weights[0] * distance[Dim::K] * 4.0 * rdx_2;
}

// Mode 4.
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights( std::integral_constant<int,4>,
                         const Scalar distance[3],
                         const Scalar rdx_2,
                         const Scalar wi[3],
                         const Scalar wj[3],
                         const Scalar wk[3],
                         const int i,
                         const int j,
                         const int k,
                         const int mod_i,
                         const int mod_j,
                         const int mod_k,
                         Scalar* mode_weights )
{
    polyPicModeWeights( std::integral_constant<int,3>(),
        distance, rdx_2, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
    mode_weights[4] = mode_weights[0] * distance[Dim::I] * distance[Dim::J] * 16.0 * rdx_2 * rdx_2;
}

// Mode 5.
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights( std::integral_constant<int,5>,
                         const Scalar distance[3],
                         const Scalar rdx_2,
                         const Scalar wi[3],
                         const Scalar wj[3],
                         const Scalar wk[3],
                         const int i,
                         const int j,
                         const int k,
                         const int mod_i,
                         const int mod_j,
                         const int mod_k,
                         Scalar* mode_weights )
{
    polyPicModeWeights(std::integral_constant<int,4>(),
        distance, rdx_2, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
    mode_weights[5] = mode_weights[0] * distance[Dim::J] * distance[Dim::K] * 16.0 * rdx_2 * rdx_2;
}

// Mode 6.
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights( std::integral_constant<int,6>,
                         const Scalar distance[3],
                         const Scalar rdx_2,
                         const Scalar wi[3],
                         const Scalar wj[3],
                         const Scalar wk[3],
                         const int i,
                         const int j,
                         const int k,
                         const int mod_i,
                         const int mod_j,
                         const int mod_k,
                         Scalar* mode_weights )
{
    polyPicModeWeights(std::integral_constant<int,5>(),
        distance, rdx_2, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
    mode_weights[6] = mode_weights[0] * distance[Dim::I] * distance[Dim::K] * 16.0 * rdx_2 * rdx_2;
}

// Mode 7.
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicModeWeights( std::integral_constant<int,7>,
                         const Scalar distance[3],
                         const Scalar rdx_2,
                         const Scalar wi[3],
                         const Scalar wj[3],
                         const Scalar wk[3],
                         const int i,
                         const int j,
                         const int k,
                         const int mod_i,
                         const int mod_j,
                         const int mod_k,
                         Scalar* mode_weights )
{
    polyPicModeWeights(std::integral_constant<int,6>(),
        distance, rdx_2, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
    mode_weights[7] = mode_weights[0] *
                      distance[Dim::I] * distance[Dim::J] * distance[Dim::K] * 64.0 * rdx_2 * rdx_2 * rdx_2;
}

// // Mode 8.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,8>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,7>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[8]  = wj[j] * wk[k] * mod_i * 4.0;
// }

// // Mode 9.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,9>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,8>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[9]  = wi[i] * wk[k] * mod_j * 4.0;
// }

// // Mode 10.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,10>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,9>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[10] = wi[i] * wj[j] * mod_k * 4.0;
// }

// // Mode 11.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,11>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,10>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[11] = wk[k] * mod_i * mod_j * 16.0;
// }

// // Mode 12.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,12>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,11>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[12] = wj[j] * mod_i * mod_k * 16.0;
// }

// // Mode 13.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,13>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,12>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[13] = wi[i] * mod_j * mod_k * 16.0;
// }

// // Mode 14.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,14>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,13>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[14] = mod_i * mod_j * mod_k * 64.0;
// }

// // Mode 15.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,15>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,14>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[15] = wj[j] * wk[k] * distance[Dim::J] * mod_i * 16.0;
// }

// // Mode 16.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,16>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,15>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[16] = wj[j] * wk[k] * distance[Dim::K] * mod_i * 16.0;
// }

// // Mode 17.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,17>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,16>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[17] = wj[j] * wk[k] *
//                        distance[Dim::J] * distance[Dim::K] * mod_i * 64.0;
// }

// // Mode 18.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,18>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,17>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[18] = wi[i] * wk[k] * distance[Dim::I] * mod_j * 16.0;
// }

// // Mode 19.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,19>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,18>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[19] = wi[i] * wk[k] * distance[Dim::K] * mod_j * 16.0;
// }

// // Mode 20.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,20>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,19>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[20] = wi[i] * wk[k] *
//                        distance[Dim::I] * distance[Dim::K] * mod_j * 64.0;
// }

// // Mode 21.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,21>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,20>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[21] = wi[i] * wj[j] * distance[Dim::I] * mod_k * 16.0;
// }

// // Mode 22.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,22>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,21>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[22] = wi[i] * wj[j] * distance[Dim::J] * mod_k * 16.0;
// }

// // Mode 23.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,23>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,22>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[23] = wi[i] * wj[j] *
//                        distance[Dim::I] * distance[Dim::J] * mod_k * 64.0;
// }

// // Mode 24.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,24>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,23>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[24] = wk[k] * distance[Dim::K] * mod_i * mod_j * 64.0;
// }

// // Mode 25.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,25>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,24>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[25] = wj[j] * distance[Dim::J] * mod_i * mod_k * 64.0;
// }

// // Mode 26.
// template<class Scalar>
// KOKKOS_INLINE_FUNCTION
// void polyPicModeWeights( std::integral_constant<int,26>,
//                          const Scalar distance[3],
//                          const Scalar wi[3],
//                          const Scalar wj[3],
//                          const Scalar wk[3],
//                          const int i,
//                          const int j,
//                          const int k,
//                          const int mod_i,
//                          const int mod_j,
//                          const int mod_k,
//                          Scalar* mode_weights )
// {
//     polyPicModeWeights(std::integral_constant<int,25>(),
//         distance, wi, wj, wk, i, j, k, mod_i, mod_j, mod_k, mode_weights );
//     mode_weights[26] = wi[i] * distance[Dim::I] * mod_j * mod_k * 64.0;
// }

//---------------------------------------------------------------------------//
// Interpolate grid node velocity to the particle.
template<int N, class SplineDataType, class VelocityView>
KOKKOS_INLINE_FUNCTION
void g2p(
    const VelocityView& node_velocity,
    const SplineDataType& sd,
    typename VelocityView::value_type u_p[N][3],
    typename std::enable_if<
    Cajita::isNode<typename SplineDataType::entity_type>::value,void*>::type = 0 )
{
    using value_type = typename VelocityView::value_type;

    auto rdx_2 = 1.0 / (sd.dx*sd.dx);

    int mod_i, mod_j, mod_k;
    value_type distance[3];
    value_type mode_weights[N];

    for ( int r = 0; r < N; ++r )
        for ( int d = 0; d < 3; ++d )
            u_p[r][d] = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Index modulos
                mod_i = ( 1 == i ) ? -2 : 1;
                mod_j = ( 1 == j ) ? -2 : 1;
                mod_k = ( 1 == k ) ? -2 : 1;

                // Compute physical distance to entity.
                distance[Dim::I] = sd.d[Dim::I][i];
                distance[Dim::J] = sd.d[Dim::J][j];
                distance[Dim::K] = sd.d[Dim::K][k];

                // Compute the mode weights.
                polyPicModeWeights(
                    std::integral_constant<int,N-1>(),
                    distance, rdx_2, sd.w[Dim::I], sd.w[Dim::J], sd.w[Dim::K],
                    i, j, k, mod_i, mod_j, mod_k,
                    mode_weights );

                // Interpolate velocity.
                for ( int r = 0; r < N; ++r )
                    for ( int d = 0; d < 3; ++d )
                        u_p[r][d] += mode_weights[r] *
                                     node_velocity( sd.s[Dim::I][i],
                                                    sd.s[Dim::J][j],
                                                    sd.s[Dim::K][k],
                                                    d );
            }
}

//---------------------------------------------------------------------------//
// Interpolate grid face velocity to the particle.
template<int N, class SplineDataType, class VelocityView>
KOKKOS_INLINE_FUNCTION
void f2p(
    const VelocityView face_velocity,
    const SplineDataType& sd,
    typename VelocityView::value_type u_p[N][3],
    typename std::enable_if<
    Cajita::isFace<typename SplineDataType::entity_type>::value,void*>::type = 0 )
{
    using value_type = typename VelocityView::value_type;
    using entity_type = typename SplineDataType::entity_type;

    auto rdx_2 = 1.0 / (sd.dx*sd.dx);

    int mod_i, mod_j, mod_k;
    value_type distance[3];
    value_type mode_weights[N];

    for ( int r = 0; r < N; ++r )
        u_p[r][entity_type::dim] = 0.0;

    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Index modulos
                mod_i = ( 1 == i ) ? -2 : 1;
                mod_j = ( 1 == j ) ? -2 : 1;
                mod_k = ( 1 == k ) ? -2 : 1;

                // Compute physical distance to entity.
                distance[Dim::I] = sd.d[Dim::I][i];
                distance[Dim::J] = sd.d[Dim::J][j];
                distance[Dim::K] = sd.d[Dim::K][k];

                // Compute the mode weights.
                polyPicModeWeights(
                    std::integral_constant<int,N-1>(),
                    distance, rdx_2, sd.w[Dim::I], sd.w[Dim::J], sd.w[Dim::K],
                    i, j, k, mod_i, mod_j, mod_k,
                    mode_weights );

                // Interpolate velocity.
                for ( int r = 0; r < N; ++r )
                    u_p[r][entity_type::dim] += mode_weights[r] *
                                                face_velocity( sd.s[Dim::I][i],
                                                               sd.s[Dim::J][j],
                                                               sd.s[Dim::K][k],
                                                               0 );
            }
}

//---------------------------------------------------------------------------//
// Interpolate MAC grid velocity to the particle.
template<int N,
         class SplineDataTypeI,
         class SplineDataTypeJ,
         class SplineDataTypeK,
         class VelocityView>
KOKKOS_INLINE_FUNCTION
void g2p(
    const VelocityView face_i_velocity,
    const VelocityView face_j_velocity,
    const VelocityView face_k_velocity,
    const SplineDataTypeI& sd_i,
    const SplineDataTypeJ& sd_j,
    const SplineDataTypeK& sd_k,
    typename VelocityView::value_type u_p[N][3] )
{
    f2p<N>( face_i_velocity, sd_i, u_p );
    f2p<N>( face_j_velocity, sd_j, u_p );
    f2p<N>( face_k_velocity, sd_k, u_p );
}

//---------------------------------------------------------------------------//
// Particle-to-Grid
//---------------------------------------------------------------------------//

// Mode 0
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,0>,
                   const Scalar[3],
                   const Scalar[3],
                   const Scalar,
                   const Scalar,
                   Scalar* coeffs )
{
    coeffs[0] = 1.0;
}

// Mode 1
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,1>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,0>(), mapping, pln, a, b, coeffs );
    coeffs[1] = mapping[Dim::I];
}

// Mode 2
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,2>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,1>(), mapping, pln, a, b, coeffs );
    coeffs[2] = mapping[Dim::J];
}

// Mode 3
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,3>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,2>(),mapping, pln, a, b, coeffs );
    coeffs[3] = mapping[Dim::K];
}

// Mode 4
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,4>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,3>(),mapping, pln, a, b, coeffs );
    coeffs[4] = mapping[Dim::I] * mapping[Dim::J];
}

// Mode 5
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,5>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,4>(),mapping, pln, a, b, coeffs );
    coeffs[5] = mapping[Dim::I] * mapping[Dim::K];
}

// Mode 6
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,6>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,5>(),mapping, pln, a, b, coeffs );
    coeffs[6] = mapping[Dim::J] * mapping[Dim::K];
}

// Mode 7
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,7>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,6>(),mapping, pln, a, b, coeffs );
    coeffs[7] = mapping[Dim::I] * mapping[Dim::J] * mapping[Dim::K];
}

// Mode 8
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,8>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,7>(),mapping, pln, a, b, coeffs );
    coeffs[8] = mapping[Dim::I] * mapping[Dim::I] -
                mapping[Dim::I] * pln[Dim::I] *
                ( 1.0 - a * pln[Dim::I] * pln[Dim::I] ) - b;
}

// Mode 9
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,9>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,8>(),mapping, pln, a, b, coeffs );
    coeffs[9] = mapping[Dim::J] * mapping[Dim::J] -
                mapping[Dim::J] * pln[Dim::J] *
                ( 1.0 - a * pln[Dim::J] * pln[Dim::J] ) - b;
}

// Mode 10
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,10>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,9>(),mapping, pln, a, b, coeffs );
    coeffs[10] = mapping[Dim::K] * mapping[Dim::K] -
                 mapping[Dim::K] * pln[Dim::K] *
                 ( 1.0 - a * pln[Dim::K] * pln[Dim::K] ) - b;
}

// Mode 11
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,11>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis(std::integral_constant<int,10>(), mapping, pln, a, b, coeffs );
    coeffs[11] = coeffs[8] * coeffs[9];
}

// Mode 12
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,12>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,11>(),mapping, pln, a, b, coeffs );
    coeffs[12] = coeffs[9] * coeffs[10];
}

// Mode 13
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,13>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,12>(),mapping, pln, a, b, coeffs );
    coeffs[13] = coeffs[8] * coeffs[10];
}

// Mode 14
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,14>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,13>(), mapping, pln, a, b, coeffs );
    coeffs[14] = coeffs[11] * coeffs[10];
}

// Mode 15
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,15>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,14>(), mapping, pln, a, b, coeffs );
    coeffs[15] = coeffs[8] * mapping[Dim::J];
}

// Mode 16
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,16>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,15>(), mapping, pln, a, b, coeffs );
    coeffs[16] = coeffs[8] * mapping[Dim::K];
}

// Mode 17
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,17>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,16>(),mapping, pln, a, b, coeffs );
    coeffs[17] = coeffs[8] * mapping[Dim::J] * mapping[Dim::K];
}

// Mode 18
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,18>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,17>(), mapping, pln, a, b, coeffs );
    coeffs[18] = coeffs[9] * mapping[Dim::I];
}

// Mode 19
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,19>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,18>(),mapping, pln, a, b, coeffs );
    coeffs[19] = coeffs[9] * mapping[Dim::K];
}

// Mode 20
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,20>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,19>(),mapping, pln, a, b, coeffs );
    coeffs[20] = coeffs[9] * mapping[Dim::I] * mapping[Dim::K];
}

// Mode 21
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,21>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,20>(),mapping, pln, a, b, coeffs );
    coeffs[21] = coeffs[10] * mapping[Dim::I];
}

// Mode 22
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,22>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,21>(),mapping, pln, a, b, coeffs );
    coeffs[22] = coeffs[10] * mapping[Dim::J];
}

// Mode 23
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,23>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,22>(),mapping, pln, a, b, coeffs );
    coeffs[23] = coeffs[10] * mapping[Dim::I] * mapping[Dim::J];
}

// Mode 24
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,24>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,23>(),mapping, pln, a, b, coeffs );
    coeffs[24] = coeffs[11] * mapping[Dim::K];
}

// Mode 25
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,25>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,24>(),mapping, pln, a, b, coeffs );
    coeffs[25] = coeffs[13] * mapping[Dim::J];
}

// Mode 26
template<class Scalar>
KOKKOS_INLINE_FUNCTION
void polyPicBasis( std::integral_constant<int,26>,
                   const Scalar mapping[3],
                   const Scalar pln[3],
                   const Scalar a,
                   const Scalar b,
                   Scalar* coeffs )
{
    polyPicBasis( std::integral_constant<int,25>(), mapping, pln, a, b, coeffs );
    coeffs[26] = coeffs[12] * mapping[Dim::I];
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum to the nodes.
template<int N, class SplineDataType, class MomentumView>
KOKKOS_INLINE_FUNCTION
void p2g(
    const typename MomentumView::original_value_type m_p,
    const typename MomentumView::original_value_type u_p[N][3],
    const SplineDataType& sd,
    const typename MomentumView::original_value_type dt,
    const MomentumView& node_momentum,
    typename std::enable_if<
    Cajita::isNode<typename SplineDataType::entity_type>::value,void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<MomentumView>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto momentum_access = node_momentum.access();

    using value_type = typename MomentumView::original_value_type;

    auto rdx = 1.0 / sd.dx;
    auto a = 4.0 * rdx * rdx;
    auto b = 0.25 * sd.dx * sd.dx;

    // Create the affine projection operator using the velocity
    // gradient.
    value_type c[3][3];
    c[0][0] = dt * u_p[1][Dim::I] + 1.0;
    c[0][1] = dt * u_p[1][Dim::J];
    c[0][2] = dt * u_p[1][Dim::K];
    c[1][0] = dt * u_p[2][Dim::I];
    c[1][1] = dt * u_p[2][Dim::J] + 1.0;
    c[1][2] = dt * u_p[2][Dim::K];
    c[2][0] = dt * u_p[3][Dim::I];
    c[2][1] = dt * u_p[3][Dim::J];
    c[2][2] = dt * u_p[3][Dim::K] + 1.0;

    // Invert the operator.
    value_type c_inv[3][3];
    DenseLinearAlgebra::inverse( c, c_inv );

    // Get the physical location of the particle in the reference
    // frame of the stencil. This is the distance between the particle
    // and the closest entity.
    value_type pln[3];
    pln[Dim::I] = (sd.x[Dim::I] - int(sd.x[Dim::I])) * sd.dx;
    pln[Dim::J] = (sd.x[Dim::J] - int(sd.x[Dim::J])) * sd.dx;
    pln[Dim::K] = (sd.x[Dim::K] - int(sd.x[Dim::K])) * sd.dx;

    // Loop data.
    value_type distance[3];
    value_type mapping[3];
    value_type coeffs[N];
    value_type wm;
    value_type u_p_d;

    // Project momentum.
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Compute physical distance to entity.
                distance[Dim::I] = sd.d[Dim::I][i];
                distance[Dim::J] = sd.d[Dim::J][j];
                distance[Dim::K] = sd.d[Dim::K][k];

                // Compute the mapping.
                DenseLinearAlgebra::matVecMultiply( c_inv, distance, mapping );

                // Get the polypic coefficients.
                polyPicBasis( std::integral_constant<int,N-1>(), mapping, pln, a, b, coeffs );

                // Weight times mass.
                wm = sd.w[Dim::I][i] *
                     sd.w[Dim::J][j] *
                     sd.w[Dim::K][k] *
                     m_p;

                // Interpolate particle momentum to the entity.
                for ( int d = 0; d < 3; ++d )
                {
                    u_p_d = 0.0;
                    for ( int r = 0; r < N; ++r )
                        u_p_d += coeffs[r] * u_p[r][d];

                    momentum_access( sd.s[Dim::I][i],
                                     sd.s[Dim::J][j],
                                     sd.s[Dim::K][k],
                                     d ) +=
                        u_p_d * wm;
                }
            }
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum to the faces.
template<int N, class SplineDataType, class MomentumView>
KOKKOS_INLINE_FUNCTION
void p2f(
    const typename MomentumView::original_value_type m_p,
    const typename MomentumView::original_value_type u_p[N][3],
    const SplineDataType& sd,
    const typename MomentumView::original_value_type c_inv[3][3],
    const MomentumView& face_momentum,
    typename std::enable_if<
    Cajita::isFace<typename SplineDataType::entity_type>::value,void*>::type = 0 )
{
    static_assert( Cajita::P2G::is_scatter_view<MomentumView>::value,
                   "P2G requires a Kokkos::ScatterView" );
    auto momentum_access = face_momentum.access();

    using value_type = typename MomentumView::original_value_type;
    using entity_type = typename SplineDataType::entity_type;

    auto rdx = 1.0 / sd.dx;
    auto a = 4.0 * rdx * rdx;
    auto b = 0.25 * sd.dx * sd.dx;

    // Get the physical location of the particle in the reference
    // frame of the stencil. This is the distance between the particle
    // and the closest entity.
    value_type pln[3];
    pln[Dim::I] = (sd.x[Dim::I] - int(sd.x[Dim::I])) * sd.dx;
    pln[Dim::J] = (sd.x[Dim::J] - int(sd.x[Dim::J])) * sd.dx;
    pln[Dim::K] = (sd.x[Dim::K] - int(sd.x[Dim::K])) * sd.dx;

    // Loop data.
    value_type distance[3];
    value_type mapping[3];
    value_type coeffs[N];
    value_type wm;
    value_type u_p_d;

    // Project momentum.
    for ( int i = 0; i < SplineDataType::num_knot; ++i )
        for ( int j = 0; j < SplineDataType::num_knot; ++j )
            for ( int k = 0; k < SplineDataType::num_knot; ++k )
            {
                // Compute physical distance to entity.
                distance[Dim::I] = sd.d[Dim::I][i];
                distance[Dim::J] = sd.d[Dim::J][j];
                distance[Dim::K] = sd.d[Dim::K][k];

                // Compute the mapping.
                DenseLinearAlgebra::matVecMultiply( c_inv, distance, mapping );

                // Get the polypic coefficients.
                polyPicBasis( std::integral_constant<int,N-1>(), mapping, pln, a, b, coeffs );

                // Weight times mass.
                wm = sd.w[Dim::I][i] *
                     sd.w[Dim::J][j] *
                     sd.w[Dim::K][k] *
                     m_p;

                // Interpolate particle momentum to the entity.
                u_p_d = 0.0;
                for ( int r = 0; r < N; ++r )
                    u_p_d += coeffs[r] * u_p[r][entity_type::dim];

                momentum_access( sd.s[Dim::I][i],
                                 sd.s[Dim::J][j],
                                 sd.s[Dim::K][k],
                                 0 ) +=
                    u_p_d * wm;
            }
}

//---------------------------------------------------------------------------//
// Interpolate particle momentum to a MAC grid.
template<int N,
         class SplineDataTypeI,
         class SplineDataTypeJ,
         class SplineDataTypeK,
         class MomentumView>
KOKKOS_INLINE_FUNCTION
void p2g(
    const typename MomentumView::original_value_type m_p,
    const typename MomentumView::original_value_type u_p[N][3],
    const SplineDataTypeI& sd_i,
    const SplineDataTypeJ& sd_j,
    const SplineDataTypeK& sd_k,
    const typename MomentumView::original_value_type dt,
    const MomentumView& face_i_momentum,
    const MomentumView& face_j_momentum,
    const MomentumView& face_k_momentum )
{
    using value_type = typename MomentumView::original_value_type;

    // Create the affine projection operator using the velocity
    // gradient.
    value_type c[3][3];
    c[0][0] = dt * u_p[1][Dim::I] + 1.0;
    c[0][1] = dt * u_p[1][Dim::J];
    c[0][2] = dt * u_p[1][Dim::K];
    c[1][0] = dt * u_p[2][Dim::I];
    c[1][1] = dt * u_p[2][Dim::J] + 1.0;
    c[1][2] = dt * u_p[2][Dim::K];
    c[2][0] = dt * u_p[3][Dim::I];
    c[2][1] = dt * u_p[3][Dim::J];
    c[2][2] = dt * u_p[3][Dim::K] + 1.0;

    // Invert the operator.
    value_type c_inv[3][3];
    DenseLinearAlgebra::inverse( c, c_inv );

    // Project to faces.
    p2f<N>( m_p, u_p, sd_i, c_inv, face_i_momentum );
    p2f<N>( m_p, u_p, sd_j, c_inv, face_j_momentum );
    p2f<N>( m_p, u_p, sd_k, c_inv, face_k_momentum );
}

//---------------------------------------------------------------------------//

} // end namespace PolyPIC
} // end namespace Harlow

#endif // end HARLOW_POLYPIC_HPP
