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

#ifndef PICASSO_PARTICLEINTERPOLATION_HPP
#define PICASSO_PARTICLEINTERPOLATION_HPP

#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Picasso_FieldTypes.hpp>

#include <Cajita.hpp>

#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Interpolation Order
//---------------------------------------------------------------------------//
template <int N>
struct InterpolationOrder
{
    static constexpr int value = N;
};

//---------------------------------------------------------------------------//
// Splines
//---------------------------------------------------------------------------//
// Indicates that the spline data should include the weight values.
struct SplineValue
{
    using spline_data_member = Cajita::SplineWeightValues;
};

// Indicates that the spline data should include the gradient of the weight
// values in the physical frame.
struct SplineGradient
{
    using spline_data_member = Cajita::SplineWeightPhysicalGradients;
};

// Indicates that the spline data should include the physical distance between
// the particle and the entities in the interpolation stencil
struct SplineDistance
{
    using spline_data_member = Cajita::SplinePhysicalDistance;
};

// Indicates that the spline data should include the physical size of the cell
// in each direction.
struct SplineCellSize
{
    using spline_data_member = Cajita::SplinePhysicalCellSize;
};

// Indicates that the spline data should include the position of the particle
// in the reference frame of the spline stencil.
struct SplineLogicalPosition
{
    using spline_data_member = Cajita::SplineLogicalPosition;
};

//---------------------------------------------------------------------------//
/*!
  \brief Create a spline of the given order on the given mesh location
  capable of evaluating the given operations at the given particle location.

  \param Location The location of the grid entities on which the spline is
  defined.

  \param Order Spline interpolation order.

  \param local_mesh The local mesh geometry to build the spline with.

  \param position The particle position vector.

  \param SplineMembers A list of the data members to be stored in the spline.

  \return The created spline.
*/
template <class Location, class Order, class PositionVector, class LocalMesh,
          class... SplineMembers>
KOKKOS_INLINE_FUNCTION auto
createSpline( Location, Order, const LocalMesh& local_mesh,
              const PositionVector& position, SplineMembers... )
{
    // FIXME - replace this with eval_type once we update Cajita to use
    // operator() of the input point data for interpolation.
    typename PositionVector::value_type x[3] = { position( 0 ), position( 1 ),
                                                 position( 2 ) };
    Cajita::SplineData<typename PositionVector::value_type, Order::value, 3,
                       typename Location::entity_type,
                       Cajita::SplineDataMemberTypes<
                           typename SplineMembers::spline_data_member...>>
        sd;
    Cajita::evaluateSpline( local_mesh, x, sd );
    return sd;
}

//---------------------------------------------------------------------------//
// Spline Grid-to-Particle
//---------------------------------------------------------------------------//
namespace G2P
{
//---------------------------------------------------------------------------//
// G2P scalar value. Requires SplineValue when constructing the spline data.
template <class ViewType, class SplineDataType, class Scalar,
          typename std::enable_if_t<!LinearAlgebra::is_vector<Scalar>::value,
                                    int> = 0>
KOKKOS_INLINE_FUNCTION void value( const SplineDataType& sd,
                                   const ViewType& view, Scalar& result )
{
    Cajita::G2P::value( view, sd, result );
}

//---------------------------------------------------------------------------//
// G2P vector value. Requires SplineValue when constructing the spline data.
template <class ViewType, class SplineDataType, class ResultVector,
          typename std::enable_if_t<
              LinearAlgebra::is_vector<ResultVector>::value, int> = 0>
KOKKOS_INLINE_FUNCTION void value( const SplineDataType& sd,
                                   const ViewType& view, ResultVector& result )
{
    // FIXME - replace this with eval_type once we update Cajita to use
    // operator() of the input point data for interpolation.
    typename ResultVector::value_type r[3];
    Cajita::G2P::value( view, sd, r );
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
    for ( int i = 0; i < 3; ++i )
        result( i ) = r[i];
}

//---------------------------------------------------------------------------//
// G2P scalar gradient. Requires SplineValue and SplineGradient when
// constructing the spline data.
template <class ViewType, class SplineDataType, class ResultVector,
          typename std::enable_if_t<
              LinearAlgebra::is_vector<ResultVector>::value, int> = 0>
KOKKOS_INLINE_FUNCTION void
gradient( const SplineDataType& sd, const ViewType& view, ResultVector& result )
{
    // FIXME - replace this with eval_type once we update Cajita to use
    // operator() of the input point data for interpolation.
    typename ResultVector::value_type r[3];
    Cajita::G2P::gradient( view, sd, r );
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
    for ( int i = 0; i < 3; ++i )
        result( i ) = r[i];
}

//---------------------------------------------------------------------------//
// G2P vector gradient. Requires SplineValue and SplineGradient when
// constructing the spline data.
template <class ViewType, class SplineDataType, class ResultMatrix,
          typename std::enable_if_t<
              LinearAlgebra::is_matrix<ResultMatrix>::value, int> = 0>
KOKKOS_INLINE_FUNCTION void
gradient( const SplineDataType& sd, const ViewType& view, ResultMatrix& result )
{
    // FIXME - replace this with eval_type once we update Cajita to use
    // operator() of the input point data for interpolation.
    typename ResultMatrix::value_type r[3][3];
    Cajita::G2P::gradient( view, sd, r );
    for ( int i = 0; i < 3; ++i )
#if defined( KOKKOS_ENABLE_PRAGMA_UNROLL )
#pragma unroll
#endif
        for ( int j = 0; j < 3; ++j )
            result( i, j ) = r[i][j];
}

//---------------------------------------------------------------------------//
// G2P vector divergence. Requires SplineValue and SplineGradient when
// constructing the spline data.
template <class ViewType, class SplineDataType, class Scalar>
KOKKOS_INLINE_FUNCTION void divergence( const SplineDataType& sd,
                                        const ViewType& view, Scalar& result )
{
    Cajita::G2P::divergence( view, sd, result );
}

//---------------------------------------------------------------------------//

} // end namespace G2P

//---------------------------------------------------------------------------//
// Spline Particle-to-Grid
//---------------------------------------------------------------------------//
namespace P2G
{
//---------------------------------------------------------------------------//
// P2G scalar value. Requires SplineValue when constructing the spline data.
template <class Scalar, class ScatterViewType, class SplineDataType,
          typename std::enable_if_t<!LinearAlgebra::is_vector<Scalar>::value,
                                    int> = 0>
KOKKOS_INLINE_FUNCTION void value( const SplineDataType& sd,
                                   const Scalar& value,
                                   const ScatterViewType& view )
{
    Cajita::P2G::value( value, sd, view );
}

//---------------------------------------------------------------------------//
// P2G vector value. Requires SplineValue when constructing the spline data.
template <class ValueVector, class ScatterViewType, class SplineDataType,
          typename std::enable_if_t<
              LinearAlgebra::is_vector<ValueVector>::value, int> = 0>
KOKKOS_INLINE_FUNCTION void value( const SplineDataType& sd,
                                   const ValueVector& value,
                                   const ScatterViewType& view )
{
    // FIXME - replace this with eval_type once we update Cajita to use
    // operator() of the input point data for interpolation.
    typename ValueVector::value_type v[3] = { value( 0 ), value( 1 ),
                                              value( 2 ) };
    Cajita::P2G::value( v, sd, view );
}

//---------------------------------------------------------------------------//
// P2G scalar gradient. Requires SplineValue and SplineGradient when
// constructing the spline data.
template <class Scalar, class ScatterViewType, class SplineDataType>
KOKKOS_INLINE_FUNCTION void gradient( const SplineDataType& sd,
                                      const Scalar& value,
                                      const ScatterViewType& view )
{
    Cajita::P2G::gradient( value, sd, view );
}

//---------------------------------------------------------------------------//
// P2G vector divergence. Requires SplineValue and SplineGradient when
// constructing the spline data.
template <class ValueVector, class ScatterViewType, class SplineDataType,
          typename std::enable_if_t<
              LinearAlgebra::is_vector<ValueVector>::value, int> = 0>
KOKKOS_INLINE_FUNCTION void divergence( const SplineDataType& sd,
                                        const ValueVector& value,
                                        const ScatterViewType& view )
{
    // FIXME - replace this with eval_type once we update Cajita to use
    // operator() of the input point data for interpolation.
    typename ValueVector::value_type v[3] = { value( 0 ), value( 1 ),
                                              value( 2 ) };
    Cajita::P2G::divergence( v, sd, view );
}

//---------------------------------------------------------------------------//
// P2G tensor divergence. Requires SplineValue and SplineGradient when
// constructing the spline data.
template <class ValueMatrix, class ScatterViewType, class SplineDataType,
          typename std::enable_if_t<
              LinearAlgebra::is_matrix<ValueMatrix>::value, int> = 0>
KOKKOS_INLINE_FUNCTION void divergence( const SplineDataType& sd,
                                        const ValueMatrix& value,
                                        const ScatterViewType& view )
{
    // FIXME - replace this with eval_type once we update Cajita to use
    // operator() of the input point data for interpolation.
    typename ValueMatrix::value_type v[3][3] = {
        { value( 0, 0 ), value( 0, 1 ), value( 0, 2 ) },
        { value( 1, 0 ), value( 1, 1 ), value( 1, 2 ) },
        { value( 2, 0 ), value( 2, 1 ), value( 2, 2 ) } };
    Cajita::P2G::divergence( v, sd, view );
}

//---------------------------------------------------------------------------//

} // end namespace P2G

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_PARTICLEINTERPOLATION_HPP
