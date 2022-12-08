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

#include <Picasso_FieldManager.hpp>
#include <Picasso_FieldTypes.hpp>
#include <Picasso_GridOperator.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_ParticleInit.hpp>
#include <Picasso_ParticleInterpolation.hpp>
#include <Picasso_ParticleList.hpp>
#include <Picasso_Types.hpp>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
// Field tags.
struct ParticleScalar : Field::Scalar<double>
{
    static std::string label() { return "particle_scalar"; }
};

struct ParticleVector : Field::Vector<double, 3>
{
    static std::string label() { return "particle_vector"; }
};

struct ParticleTensor : Field::Matrix<double, 3, 3>
{
    static std::string label() { return "particle_tensor"; }
};

struct NodeScalar : Field::Scalar<double>
{
    static std::string label() { return "node_scalar"; }
};

struct NodeVector : Field::Vector<double, 3>
{
    static std::string label() { return "node_vector"; }
};

//---------------------------------------------------------------------------//
struct ScalarValueP2G
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get output dependencies.
        auto node_scalar =
            scatter_deps.get( FieldLocation::Node(), NodeScalar() );

        // Get particle data.
        auto particle_scalar = get( particle, ParticleScalar() );

        // Node Interpolant
        auto spline = createSpline(
            FieldLocation::Node(), InterpolationOrder<1>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue() );

        // Interpolate to grid.
        P2G::value( spline, particle_scalar, node_scalar );
    }
};

//---------------------------------------------------------------------------//
struct VectorValueP2G
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get output dependencies.
        auto node_vector =
            scatter_deps.get( FieldLocation::Node(), NodeVector() );

        // Get particle data.
        auto particle_vector = get( particle, ParticleVector() );

        // Node Interpolant
        auto spline = createSpline(
            FieldLocation::Node(), InterpolationOrder<1>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue() );

        // Interpolate to grid.
        P2G::value( spline, particle_vector, node_vector );
    }
};

//---------------------------------------------------------------------------//
struct ScalarGradientP2G
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get output dependencies.
        auto node_vector =
            scatter_deps.get( FieldLocation::Node(), NodeVector() );

        // Get particle data.
        auto particle_scalar = get( particle, ParticleScalar() );

        // Node Interpolant
        auto spline = createSpline(
            FieldLocation::Node(), InterpolationOrder<1>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue(),
            SplineGradient() );

        // Interpolate to grid.
        P2G::gradient( spline, particle_scalar, node_vector );
    }
};

//---------------------------------------------------------------------------//
struct VectorDivergenceP2G
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get output dependencies.
        auto node_scalar =
            scatter_deps.get( FieldLocation::Node(), NodeScalar() );

        // Get particle data.
        auto particle_vector = get( particle, ParticleVector() );

        // Node Interpolant
        auto spline = createSpline(
            FieldLocation::Node(), InterpolationOrder<1>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue(),
            SplineGradient() );

        // Interpolate to grid.
        P2G::divergence( spline, particle_vector, node_scalar );
    }
};

//---------------------------------------------------------------------------//
struct TensorDivergenceP2G
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get output dependencies.
        auto node_vector =
            scatter_deps.get( FieldLocation::Node(), NodeVector() );

        // Get particle data.
        auto particle_tensor = get( particle, ParticleTensor() );

        // Node Interpolant
        auto spline = createSpline(
            FieldLocation::Node(), InterpolationOrder<1>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue(),
            SplineGradient() );

        // Interpolate to grid.
        P2G::divergence( spline, particle_tensor, node_vector );
    }
};

//---------------------------------------------------------------------------//
struct ScalarValueG2P
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies&,
                ParticleViewType& particle ) const
    {
        // Get input dependencies.
        auto node_scalar =
            gather_deps.get( FieldLocation::Node(), NodeScalar() );

        // Get particle data.
        auto& particle_scalar = get( particle, ParticleScalar() );

        // Node Interpolant
        auto spline = createSpline(
            FieldLocation::Node(), InterpolationOrder<1>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue() );

        // Interpolate to grid.
        G2P::value( spline, node_scalar, particle_scalar );
    }
};

//---------------------------------------------------------------------------//
struct VectorValueG2P
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies&,
                ParticleViewType& particle ) const
    {
        // Get input dependencies.
        auto node_vector =
            gather_deps.get( FieldLocation::Node(), NodeVector() );

        // Get particle data.
        auto particle_vector = get( particle, ParticleVector() );

        // Node Interpolant
        auto spline = createSpline(
            FieldLocation::Node(), InterpolationOrder<1>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue() );

        // Interpolate to grid.
        G2P::value( spline, node_vector, particle_vector );
    }
};

//---------------------------------------------------------------------------//
struct ScalarGradientG2P
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies&,
                ParticleViewType& particle ) const
    {
        // Get input dependencies.
        auto node_scalar =
            gather_deps.get( FieldLocation::Node(), NodeScalar() );

        // Get particle data.
        auto particle_vector = get( particle, ParticleVector() );

        // Node Interpolant
        auto spline = createSpline(
            FieldLocation::Node(), InterpolationOrder<1>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue(),
            SplineGradient() );

        // Interpolate to grid.
        G2P::gradient( spline, node_scalar, particle_vector );
    }
};

//---------------------------------------------------------------------------//
struct VectorGradientG2P
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies&,
                ParticleViewType& particle ) const
    {
        // Get input dependencies.
        auto node_vector =
            gather_deps.get( FieldLocation::Node(), NodeVector() );

        // Get particle data.
        auto particle_tensor = get( particle, ParticleTensor() );

        // Node Interpolant
        auto spline = createSpline(
            FieldLocation::Node(), InterpolationOrder<1>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue(),
            SplineGradient() );

        // Interpolate to grid.
        G2P::gradient( spline, node_vector, particle_tensor );
    }
};

//---------------------------------------------------------------------------//
struct VectorDivergenceG2P
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies&,
                ParticleViewType& particle ) const
    {
        // Get input dependencies.
        auto node_vector =
            gather_deps.get( FieldLocation::Node(), NodeVector() );

        // Get particle data.
        auto& particle_scalar = get( particle, ParticleScalar() );

        // Node Interpolant
        auto spline = createSpline(
            FieldLocation::Node(), InterpolationOrder<1>(), local_mesh,
            get( particle, Field::LogicalPosition<3>() ), SplineValue(),
            SplineGradient() );

        // Interpolate to grid.
        G2P::divergence( spline, node_vector, particle_scalar );
    }
};

//---------------------------------------------------------------------------//
void interpolationTest()
{
    // Global bounding box.
    double cell_size = 0.05;
    std::array<int, 3> global_num_cell = { 18, 22, 39 };
    std::array<double, 3> global_low_corner = { -1.2, 0.1, 1.1 };
    std::array<double, 3> global_high_corner = {
        global_low_corner[0] + cell_size * global_num_cell[0],
        global_low_corner[1] + cell_size * global_num_cell[1],
        global_low_corner[2] + cell_size * global_num_cell[2] };

    // Get inputs for mesh.
    auto inputs = parse( "particle_interpolation_test.json" );
    Kokkos::Array<double, 6> global_box = {
        global_low_corner[0],  global_low_corner[1],  global_low_corner[2],
        global_high_corner[0], global_high_corner[1], global_high_corner[2] };
    int minimum_halo_size = 0;

    // Make mesh.
    auto mesh = createUniformMesh( TEST_MEMSPACE(), inputs, global_box,
                                   minimum_halo_size, MPI_COMM_WORLD );

    // Get a set of locall-owned node indices for testing.
    auto node_space = mesh->localGrid()->indexSpace(
        Cajita::Own(), Cajita::Node(), Cajita::Local() );

    // Make a particle list.
    using list_type =
        ParticleList<UniformMesh<TEST_MEMSPACE>, Field::LogicalPosition<3>,
                     ParticleScalar, ParticleVector, ParticleTensor>;
    list_type particles( "test_particles", mesh );
    using particle_type = typename list_type::particle_type;

    // Particle initialization functor. Make particles everywhere.
    auto particle_init_func =
        KOKKOS_LAMBDA( const double x[3], const double, particle_type& p )
    {
        for ( int d = 0; d < 3; ++d )
            get( p, Field::LogicalPosition<3>(), d ) = x[d];
        return true;
    };

    // Initialize particles. Put one particle in the center of every cell.
    int ppc = 1;
    initializeParticles( InitUniform(), TEST_EXECSPACE(), ppc,
                         particle_init_func, particles );
    int num_particle = particles.size();

    // Get slices.
    auto scalar_p = particles.slice( ParticleScalar() );
    auto vector_p = particles.slice( ParticleVector() );
    auto tensor_p = particles.slice( ParticleTensor() );

    // Make mirror
    auto particles_host =
        Cabana::create_mirror_view( Kokkos::HostSpace(), particles.aosoa() );
    auto scalar_p_host = Cabana::slice<1>( particles_host );
    auto vector_p_host = Cabana::slice<2>( particles_host );
    auto tensor_p_host = Cabana::slice<3>( particles_host );

    // Make a field manager.
    auto fm = createFieldManager( mesh );

    // Create P2G operators.
    using p2gsv_scatter =
        ScatterDependencies<FieldLayout<FieldLocation::Node, NodeScalar>>;
    auto p2gsv_op = createGridOperator( mesh, p2gsv_scatter() );
    p2gsv_op->setup( *fm );

    using p2gvv_scatter =
        ScatterDependencies<FieldLayout<FieldLocation::Node, NodeVector>>;
    auto p2gvv_op = createGridOperator( mesh, p2gvv_scatter() );
    p2gvv_op->setup( *fm );

    using p2gsg_scatter =
        ScatterDependencies<FieldLayout<FieldLocation::Node, NodeVector>>;
    auto p2gsg_op = createGridOperator( mesh, p2gsg_scatter() );
    p2gsg_op->setup( *fm );

    using p2gvd_scatter =
        ScatterDependencies<FieldLayout<FieldLocation::Node, NodeScalar>>;
    auto p2gvd_op = createGridOperator( mesh, p2gvd_scatter() );
    p2gvd_op->setup( *fm );

    using p2gtd_scatter =
        ScatterDependencies<FieldLayout<FieldLocation::Node, NodeVector>>;
    auto p2gtd_op = createGridOperator( mesh, p2gtd_scatter() );
    p2gtd_op->setup( *fm );

    // Create G2P operators.
    using g2psv_gather =
        GatherDependencies<FieldLayout<FieldLocation::Node, NodeScalar>>;
    auto g2psv_op = createGridOperator( mesh, g2psv_gather() );
    g2psv_op->setup( *fm );

    using g2pvv_gather =
        GatherDependencies<FieldLayout<FieldLocation::Node, NodeVector>>;
    auto g2pvv_op = createGridOperator( mesh, g2pvv_gather() );
    g2pvv_op->setup( *fm );

    using g2psg_gather =
        GatherDependencies<FieldLayout<FieldLocation::Node, NodeScalar>>;
    auto g2psg_op = createGridOperator( mesh, g2psg_gather() );
    g2psg_op->setup( *fm );

    using g2pvg_gather =
        GatherDependencies<FieldLayout<FieldLocation::Node, NodeVector>>;
    auto g2pvg_op = createGridOperator( mesh, g2pvg_gather() );
    g2pvg_op->setup( *fm );

    using g2pvd_gather =
        GatherDependencies<FieldLayout<FieldLocation::Node, NodeVector>>;
    auto g2pvd_op = createGridOperator( mesh, g2pvd_gather() );
    g2pvd_op->setup( *fm );

    // Get fields.
    auto scalar_n = fm->view( FieldLocation::Node(), NodeScalar() );
    auto vector_n = fm->view( FieldLocation::Node(), NodeVector() );

    auto scalar_n_host =
        Kokkos::create_mirror_view( Kokkos::HostSpace(), scalar_n );
    auto vector_n_host =
        Kokkos::create_mirror_view( Kokkos::HostSpace(), vector_n );

    // P2G
    // ---

    Cabana::deep_copy( scalar_p, 3.5 );
    Cabana::deep_copy( vector_p, 3.5 );
    Cabana::deep_copy( tensor_p, 3.5 );

    // Interpolate a scalar point value to the grid.
    Kokkos::deep_copy( scalar_n, 0.0 );
    p2gsv_op->apply( "p2g_scalar_val", FieldLocation::Particle(),
                     TEST_EXECSPACE(), *fm, particles, ScalarValueP2G() );
    Kokkos::deep_copy( scalar_n_host, scalar_n );
    for ( int i = node_space.min( Dim::I ); i < node_space.max( Dim::I ); ++i )
        for ( int j = node_space.min( Dim::J ); j < node_space.max( Dim::J );
              ++j )
            for ( int k = node_space.min( Dim::K );
                  k < node_space.max( Dim::K ); ++k )
                EXPECT_FLOAT_EQ( scalar_n_host( i, j, k, 0 ), 3.5 );

    // Interpolate a vector point value to the grid.
    Kokkos::deep_copy( vector_n, 0.0 );
    p2gvv_op->apply( "p2g_vector_val", FieldLocation::Particle(),
                     TEST_EXECSPACE(), *fm, particles, VectorValueP2G() );
    Kokkos::deep_copy( vector_n_host, vector_n );
    for ( int i = node_space.min( Dim::I ); i < node_space.max( Dim::I ); ++i )
        for ( int j = node_space.min( Dim::J ); j < node_space.max( Dim::J );
              ++j )
            for ( int k = node_space.min( Dim::K );
                  k < node_space.max( Dim::K ); ++k )
                for ( int d = 0; d < 3; ++d )
                    EXPECT_FLOAT_EQ( vector_n_host( i, j, k, d ), 3.5 );

    // Interpolate a scalar point gradient value to the grid.
    Kokkos::deep_copy( vector_n, 0.0 );
    p2gsg_op->apply( "p2g_scalar_grad", FieldLocation::Particle(),
                     TEST_EXECSPACE(), *fm, particles, ScalarGradientP2G() );
    Kokkos::deep_copy( vector_n_host, vector_n );
    for ( int i = node_space.min( Dim::I ); i < node_space.max( Dim::I ); ++i )
        for ( int j = node_space.min( Dim::J ); j < node_space.max( Dim::J );
              ++j )
            for ( int k = node_space.min( Dim::K );
                  k < node_space.max( Dim::K ); ++k )
                for ( int d = 0; d < 3; ++d )
                    EXPECT_FLOAT_EQ( vector_n_host( i, j, k, d ) + 1.0, 1.0 );

    // Interpolate a vector point divergence value to the grid.
    Kokkos::deep_copy( scalar_n, 0.0 );
    p2gvd_op->apply( "p2g_vector_div", FieldLocation::Particle(),
                     TEST_EXECSPACE(), *fm, particles, VectorDivergenceP2G() );
    Kokkos::deep_copy( scalar_n_host, scalar_n );
    for ( int i = node_space.min( Dim::I ); i < node_space.max( Dim::I ); ++i )
        for ( int j = node_space.min( Dim::J ); j < node_space.max( Dim::J );
              ++j )
            for ( int k = node_space.min( Dim::K );
                  k < node_space.max( Dim::K ); ++k )
                EXPECT_FLOAT_EQ( scalar_n_host( i, j, k, 0 ) + 1.0, 1.0 );

    // Interpolate a tensor point divergence value to the grid.
    Kokkos::deep_copy( vector_n, 0.0 );
    p2gtd_op->apply( "p2g_tensor_div", FieldLocation::Particle(),
                     TEST_EXECSPACE(), *fm, particles, TensorDivergenceP2G() );
    Kokkos::deep_copy( vector_n_host, vector_n );
    for ( int i = node_space.min( Dim::I ); i < node_space.max( Dim::I ); ++i )
        for ( int j = node_space.min( Dim::J ); j < node_space.max( Dim::J );
              ++j )
            for ( int k = node_space.min( Dim::K );
                  k < node_space.max( Dim::K ); ++k )
                for ( int d = 0; d < 3; ++d )
                    EXPECT_FLOAT_EQ( vector_n_host( i, j, k, d ) + 1.0, 1.0 );

    // G2P
    // ---

    Kokkos::deep_copy( scalar_n, 3.5 );
    Kokkos::deep_copy( vector_n, 3.5 );

    // Interpolate a scalar grid value to the points.
    Cabana::deep_copy( scalar_p, 0.0 );
    g2psv_op->apply( "g2p_scalar_val", FieldLocation::Particle(),
                     TEST_EXECSPACE(), *fm, particles, ScalarValueG2P() );
    Cabana::deep_copy( scalar_p_host, scalar_p );
    for ( int p = 0; p < num_particle; ++p )
        EXPECT_FLOAT_EQ( scalar_p_host( p ), 3.5 );

    // Interpolate a vector grid value to the points.
    Cabana::deep_copy( vector_p, 0.0 );
    g2pvv_op->apply( "g2p_vector_val", FieldLocation::Particle(),
                     TEST_EXECSPACE(), *fm, particles, VectorValueG2P() );
    Cabana::deep_copy( vector_p_host, vector_p );
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
            EXPECT_FLOAT_EQ( vector_p_host( p, d ), 3.5 );

    // Interpolate a scalar grid gradient to the points.
    Cabana::deep_copy( vector_p, 0.0 );
    g2psg_op->apply( "g2p_scalar_grad", FieldLocation::Particle(),
                     TEST_EXECSPACE(), *fm, particles, ScalarGradientG2P() );
    Cabana::deep_copy( vector_p_host, vector_p );
    for ( int p = 0; p < num_particle; ++p )
        for ( int d = 0; d < 3; ++d )
            EXPECT_FLOAT_EQ( vector_p_host( p, d ) + 1.0, 1.0 );

    // Interpolate a vector grid gradient to the points.
    Cabana::deep_copy( tensor_p, 0.0 );
    g2pvg_op->apply( "g2p_vector_grad", FieldLocation::Particle(),
                     TEST_EXECSPACE(), *fm, particles, VectorGradientG2P() );
    Cabana::deep_copy( tensor_p_host, tensor_p );
    for ( int p = 0; p < num_particle; ++p )
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                EXPECT_FLOAT_EQ( tensor_p_host( p, i, j ) + 1.0, 1.0 );

    // Interpolate a vector grid divergence to the points.
    Cabana::deep_copy( scalar_p, 0.0 );
    g2pvd_op->apply( "g2p_tensor_div", FieldLocation::Particle(),
                     TEST_EXECSPACE(), *fm, particles, VectorDivergenceG2P() );
    Cabana::deep_copy( scalar_p_host, scalar_p );
    for ( int p = 0; p < num_particle; ++p )
        EXPECT_FLOAT_EQ( scalar_p_host( p ) + 1.0, 1.0 );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, interpolation_test ) { interpolationTest(); }

//---------------------------------------------------------------------------//

} // end namespace Test
