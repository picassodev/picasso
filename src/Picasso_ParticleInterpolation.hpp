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

#include <Picasso_APIC.hpp>
#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Picasso_FieldTypes.hpp>
#include <Picasso_PolyPIC.hpp>

#include <Cabana_Grid.hpp>

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
    using spline_data_member = Cabana::Grid::SplineWeightValues;
};

// Indicates that the spline data should include the gradient of the weight
// values in the physical frame.
struct SplineGradient
{
    using spline_data_member = Cabana::Grid::SplineWeightPhysicalGradients;
};

// Indicates that the spline data should include the physical distance between
// the particle and the entities in the interpolation stencil
struct SplineDistance
{
    using spline_data_member = Cabana::Grid::SplinePhysicalDistance;
};

// Indicates that the spline data should include the physical size of the cell
// in each direction.
struct SplineCellSize
{
    using spline_data_member = Cabana::Grid::SplinePhysicalCellSize;
};

// Indicates that the spline data should include the position of the particle
// in the reference frame of the spline stencil.
struct SplineLogicalPosition
{
    using spline_data_member = Cabana::Grid::SplineLogicalPosition;
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
    Cabana::Grid::SplineData<typename PositionVector::value_type, Order::value,
                             3, typename Location::entity_type,
                             Cabana::Grid::SplineDataMemberTypes<
                                 typename SplineMembers::spline_data_member...>>
        sd;
    Cabana::Grid::evaluateSpline( local_mesh, x, sd );
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
    Cabana::Grid::G2P::value( view, sd, result );
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
    Cabana::Grid::G2P::value( view, sd, r );
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
    Cabana::Grid::G2P::gradient( view, sd, r );
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
    Cabana::Grid::G2P::gradient( view, sd, r );
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
    Cabana::Grid::G2P::divergence( view, sd, result );
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
    Cabana::Grid::P2G::value( value, sd, view );
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
    Cabana::Grid::P2G::value( v, sd, view );
}

//---------------------------------------------------------------------------//
// P2G scalar gradient. Requires SplineValue and SplineGradient when
// constructing the spline data.
template <class Scalar, class ScatterViewType, class SplineDataType>
KOKKOS_INLINE_FUNCTION void gradient( const SplineDataType& sd,
                                      const Scalar& value,
                                      const ScatterViewType& view )
{
    Cabana::Grid::P2G::gradient( value, sd, view );
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
    Cabana::Grid::P2G::divergence( v, sd, view );
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
    Cabana::Grid::P2G::divergence( v, sd, view );
}

//---------------------------------------------------------------------------//

} // end namespace P2G

//---------------------------------------------------------------------------//
template <int InterpolationOrder, class ParticleFieldType, class OldFieldType,
          class InterpolationType>
struct Particle2Grid;

//---------------------------------------------------------------------------//
// Project particle enthalpy/momentum to grid. PolyPIC variant
//---------------------------------------------------------------------------//
template <int InterpolationOrder, class ParticleFieldType, class OldFieldType>
struct Particle2Grid<InterpolationOrder, ParticleFieldType, OldFieldType,
                     PolyPicTag>
{
    // Explicit time step.
    double dt;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get particle data.
        auto f_p = Picasso::get( particle, ParticleFieldType() );
        auto v_p = Picasso::get( particle, PolyPIC::Field::Velocity() );
        auto m_p = Picasso::get( particle, Field::Mass() );
        auto x_p = Picasso::get( particle, Field::Position() );

        // Get the scatter dependencies.
        auto m_i =
            scatter_deps.get( Picasso::FieldLocation::Node(), Field::Mass() );
        auto f_i =
            scatter_deps.get( Picasso::FieldLocation::Node(), OldFieldType() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineDistance() );

        // Interpolate mass and mass-weighted enthalpy/momentum to grid with
        // PolyPIC.
        Picasso::PolyPIC::p2g( m_p, v_p, f_p, f_i, m_i, dt, spline );
    }
};

//---------------------------------------------------------------------------//
// Project particle enthalpy/momentum to grid. APIC variant
//---------------------------------------------------------------------------//
template <int InterpolationOrder, class ParticleFieldType, class OldFieldType>
struct Particle2Grid<InterpolationOrder, ParticleFieldType, OldFieldType,
                     APicTag>
{
    // Explicit time step.
    double dt;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get particle data.
        auto f_p = Picasso::get( particle, ParticleFieldType() );
        auto m_p = Picasso::get( particle, Field::Mass() );
        auto x_p = Picasso::get( particle, Field::Position() );

        // Get the scatter dependencies.
        auto m_i =
            scatter_deps.get( Picasso::FieldLocation::Node(), Field::Mass() );
        auto f_i =
            scatter_deps.get( Picasso::FieldLocation::Node(), OldFieldType() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineDistance(),
            Picasso::SplineGradient() );

        // Interpolate mass and mass-weighted enthalpy/momentum to grid with
        // APIC.
        Picasso::APIC::p2g( m_p, f_p, m_i, f_i, spline );
    }
};

//---------------------------------------------------------------------------//
// Project particle enthalpy/momentum to grid. FLIP/PIC variant
//---------------------------------------------------------------------------//
template <int InterpolationOrder, class ParticleFieldType, class OldFieldType>
struct Particle2Grid<InterpolationOrder, ParticleFieldType, OldFieldType,
                     FlipTag>
{
    // Explicit time step.
    double dt;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get particle data.
        auto f_p = Picasso::get( particle, ParticleFieldType() );
        auto m_p = Picasso::get( particle, Field::Mass() );
        auto x_p = Picasso::get( particle, Field::Position() );

        // Get the scatter dependencies.
        auto m_i =
            scatter_deps.get( Picasso::FieldLocation::Node(), Field::Mass() );
        auto f_i =
            scatter_deps.get( Picasso::FieldLocation::Node(), OldFieldType() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineDistance() );

        // Interpolate mass and mass-weighted enthalpy/momentum to grid.
        Picasso::P2G::value( spline, m_p, m_i );
        Picasso::P2G::value( spline, m_p * f_p, f_i );
    }
};

template <int InterpolationOrder, class InterpolationType>
struct Grid2ParticleVelocity;

//---------------------------------------------------------------------------//
// Update particle state. PolyPIC variant.
//---------------------------------------------------------------------------//
template <int InterpolationOrder>
struct Grid2ParticleVelocity<InterpolationOrder, PolyPicTag>
{
    // Explicit time step.
    double dt;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies& local_deps,
                ParticleViewType& particle ) const
    {
        // Get particle data.
        auto u_p = Picasso::get( particle, PolyPIC::Field::Velocity() );
        auto x_p = Picasso::get( particle, Field::Position() );

        // Get the gather dependencies.
        auto m_i =
            gather_deps.get( Picasso::FieldLocation::Node(), Field::Mass() );
        auto u_i = gather_deps.get( Picasso::FieldLocation::Node(),
                                    Field::Velocity() );
        // Get the local dependencies for getting physcial location of node
        auto x_i = local_deps.get( Picasso::FieldLocation::Node(),
                                   Picasso::Field::PhysicalPosition<3>() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineGradient() );

        // Update particle velocity using a PolyPIC update.
        Picasso::PolyPIC::g2p( u_i, u_p, spline );

        // Update particle position.
        auto x_i_updated =
            [=]( const int i, const int j, const int k, const int d )
        { return x_i( i, j, k, d ) + dt * u_i( i, j, k, d ); };
        Picasso::G2P::value( spline, x_i_updated, x_p );
    }
};

//---------------------------------------------------------------------------//
// Update particle state. APIC variant.
//---------------------------------------------------------------------------//
template <int InterpolationOrder>
struct Grid2ParticleVelocity<InterpolationOrder, APicTag>
{
    // Explicit time step.
    double dt;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies& local_deps,
                ParticleViewType& particle ) const
    {
        // Get particle data.
        auto f_p = Picasso::get( particle, APIC::Field::Velocity() );
        auto x_p = Picasso::get( particle, Field::Position() );

        // Get the gather dependencies.
        auto m_i =
            gather_deps.get( Picasso::FieldLocation::Node(), Field::Mass() );
        auto u_i = gather_deps.get( Picasso::FieldLocation::Node(),
                                    Field::Velocity() );
        // Get the local dependencies for getting physcial location of node
        auto x_i = local_deps.get( Picasso::FieldLocation::Node(),
                                   Picasso::Field::PhysicalPosition<3>() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineDistance(),
            Picasso::SplineGradient() );

        // Update particle velocity using a APIC update.
        Picasso::APIC::g2p( u_i, f_p, spline );

        // Update particle position.
        auto x_i_updated =
            [=]( const int i, const int j, const int k, const int d )
        { return x_i( i, j, k, d ) + dt * u_i( i, j, k, d ); };
        Picasso::G2P::value( spline, x_i_updated, x_p );
    }
};

//---------------------------------------------------------------------------//
// Update particle state. FLIP/PIC variant.
//---------------------------------------------------------------------------//
template <int InterpolationOrder>
struct Grid2ParticleVelocity<InterpolationOrder, FlipTag>
{
    double beta = 0.01;

    // Explicit time step.
    double dt;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType, class PositionType = Field::Position,
              class VelocityType = Field::Velocity>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies& local_deps,
                ParticleViewType& particle ) const
    {
        // Get particle data.
        auto u_p = Picasso::get( particle, VelocityType{} );
        auto x_p = Picasso::get( particle, PositionType{} );

        // Get the gather dependencies.
        auto m_i =
            gather_deps.get( Picasso::FieldLocation::Node(), Field::Mass() );
        auto u_i = gather_deps.get( Picasso::FieldLocation::Node(),
                                    Field::Velocity() );
        auto old_u_i =
            gather_deps.get( Picasso::FieldLocation::Node(), Field::OldU() );

        // Get the local dependencies for getting physcial location of node
        auto x_i = local_deps.get( Picasso::FieldLocation::Node(),
                                   Picasso::Field::PhysicalPosition<3>() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineGradient() );

        // Update particle velocity using a hybrid PIC/FLIP update.
        // Note the need for 4 indices here because this is passed to the Cabana
        // grid.
        auto d_u_i = [=]( const int i, const int j, const int k, const int d )
        {
            return ( m_i( i, j, k ) > 0.0 )
                       ? u_i( i, j, k, d ) - old_u_i( i, j, k, d )
                       : 0.0;
        };

        Picasso::Vec3<double> u_p_pic;
        // auto u_i_pic = [=]( const int i, const int j, const int k,
        //                    const int d ) { return u_i( i, j, k, d ); };
        Picasso::G2P::value( spline, u_i, u_p_pic );

        Picasso::Vec3<double> d_u_p;
        Picasso::G2P::value( spline, d_u_i, d_u_p );

        Picasso::Vec3<double> u_p_flip;

        u_p_flip = u_p + d_u_p;

        // Update particle velocity.
        u_p = beta * u_p_flip + ( 1.0 - beta ) * u_p_pic;

        // Update particle position.
        auto x_i_updated =
            [=]( const int i, const int j, const int k, const int d )
        { return x_i( i, j, k, d ) + dt * u_i( i, j, k, d ); };
        Picasso::G2P::value( spline, x_i_updated, x_p );
    }
};

} // end namespace Picasso

#endif // end PICASSO_PARTICLEINTERPOLATION_HPP
