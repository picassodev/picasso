#include <Picasso.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

using namespace Picasso;

// Dam break particle creation and initialization.
template <typename ParticleType, typename VelocityType>
struct ParticleInitFunc
{
    Kokkos::Array<double, 6> block;
    double density;

    KOKKOS_INLINE_FUNCTION bool
    operator()( const int, const double x[3], const double pv,
                typename ParticleType::particle_type& p ) const
    {
        if ( block[0] <= x[0] && x[0] <= block[3] && block[1] <= x[1] &&
             x[1] <= block[4] && block[2] <= x[2] && x[2] <= block[5] )
        {

            Picasso::get( p, Picasso::Field::Stress() ) = 0.0;
            Picasso::get( p, VelocityType() ) = 0.0;
            Picasso::get( p, Picasso::Field::DetDefGrad() ) = 1.0;
            Picasso::get( p, Picasso::Field::Mass() ) = pv * density;
            Picasso::get( p, Picasso::Field::Volume() ) = pv;
            Picasso::get( p, Picasso::Field::Pressure() ) = 0.0;

            for ( int d = 0; d < 3; ++d )
                Picasso::get( p, Picasso::Field::Position(), d ) = x[d];
            return true;
        }

        return false;
    }
};

// Dam break system boundaries.
struct BoundaryCondition
{
    Kokkos::Array<long int, 6> bc_index_space;
    Kokkos::Array<bool, 6> on_boundary;

    // Free slip boundary condition
    template <class ViewType>
    KOKKOS_INLINE_FUNCTION void apply( ViewType view, const int i, const int j,
                                       const int k ) const
    {
        // -x face
        if ( i == bc_index_space[0] && on_boundary[0] )
            view( i, j, k, 0 ) = 0.0;
        // +x face
        if ( i == bc_index_space[3] && on_boundary[3] )
            view( i, j, k, 0 ) = 0.0;
        // -y face
        if ( j == bc_index_space[1] && on_boundary[1] )
            view( i, j, k, 1 ) = 0.0;
        // +y face
        if ( j == bc_index_space[4] && on_boundary[4] )
            view( i, j, k, 1 ) = 0.0;
        // -z face
        if ( k == bc_index_space[2] && on_boundary[2] )
            view( i, j, k, 2 ) = 0.0;
        // +z face
        if ( k == bc_index_space[5] && on_boundary[5] )
            view( i, j, k, 2 ) = 0.0;
    }
};

// Custom field types for stress and gravity kernels.
struct DeltaUGravity : Field::Vector<double, 3>
{
    static std::string label() { return "velocity_change_from_gravity"; }
};
struct DeltaUStress : Field::Vector<double, 3>
{
    static std::string label() { return "velocity_change_from_stress"; }
};

//---------------------------------------------------------------------------//
// Update particle stress.
//---------------------------------------------------------------------------//
template <int InterpolationOrder>
struct ComputeParticlePressure
{
    double dt;
    double bulk_modulus;
    double gamma;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies&,
                ParticleViewType& particle ) const
    {
        // Get particle data.
        auto x_p = Picasso::get( particle, Picasso::Field::Position() );
        auto& J_p = Picasso::get( particle, Picasso::Field::DetDefGrad() );
        auto& p_p = Picasso::get( particle, Picasso::Field::Pressure() );
        auto s_p = Picasso::get( particle, Picasso::Field::Stress() );
        auto v_p = Picasso::get( particle, Picasso::Field::Volume() );
        auto m_p = Picasso::get( particle, Picasso::Field::Mass() );

        // Get the gather dependencies.
        auto u_i = gather_deps.get( Picasso::FieldLocation::Node(),
                                    Picasso::Field::Velocity() );

        // update strain rate
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineGradient() );
        Picasso::Mat3<double> vel_grad;
        Picasso::G2P::gradient( spline, u_i, vel_grad );

        // J_p = Kokkos::abs( !F_p );
        J_p *= Kokkos::exp( Picasso::LinearAlgebra::trace( dt * vel_grad ) );

        p_p = bulk_modulus * ( Kokkos::pow( J_p, -gamma ) - 1.0 );

        Picasso::Mat3<double> I;
        Picasso::LinearAlgebra::identity( I );

        s_p = -p_p * I;
    }
};

//---------------------------------------------------------------------------//
// Grid momentum change due to stress
//---------------------------------------------------------------------------//
template <int InterpolationOrder>
struct ComputeGridVelocityChange
{
    Picasso::Vec3<double> gravity;

    ComputeGridVelocityChange( std::array<double, 3> g )
    {
        for ( int d = 0; d < 3; ++d )
            gravity( d ) = g[d];
    }

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get particle data.
        auto m_p = Picasso::get( particle, Picasso::Field::Mass() );
        auto v_p = Picasso::get( particle, Picasso::Field::Volume() );
        auto x_p = Picasso::get( particle, Picasso::Field::Position() );
        auto s_p = Picasso::get( particle, Picasso::Field::Stress() );

        // Get the scatter dependencies.
        auto delta_u_s_i =
            scatter_deps.get( Picasso::FieldLocation::Node(), DeltaUStress() );
        auto delta_u_g_i =
            scatter_deps.get( Picasso::FieldLocation::Node(), DeltaUGravity() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineGradient() );

        // Compute velocity update from stress.
        Picasso::P2G::divergence( spline, -v_p * s_p, delta_u_s_i );
        // Compute velocity update from gravity.
        Picasso::P2G::value( spline, m_p * gravity, delta_u_g_i );
    }
};

//---------------------------------------------------------------------------//
// Update nodal momentum n+1 with stress, gravity, and BC
//---------------------------------------------------------------------------//
struct UpdateGridVelocity
{
    // Explicit time step size.
    double dt;
    BoundaryCondition bc;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType&, const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies&,
                const int i, const int j, const int k ) const
    {
        // Get the gather dependencies.
        auto m_i = gather_deps.get( Picasso::FieldLocation::Node(),
                                    Picasso::Field::Mass() );
        auto u_i = gather_deps.get( Picasso::FieldLocation::Node(),
                                    Picasso::Field::Velocity() );
        auto delta_u_s_i =
            gather_deps.get( Picasso::FieldLocation::Node(), DeltaUStress() );

        auto delta_u_g_i =
            gather_deps.get( Picasso::FieldLocation::Node(), DeltaUGravity() );

        // Compute velocity.
        Picasso::Vec3<double> zeros = { 0.0, 0.0, 0.0 };
        u_i( i, j, k ) +=
            ( m_i( i, j, k ) > 0.0 )
                ? dt * ( delta_u_s_i( i, j, k ) + delta_u_g_i( i, j, k ) ) /
                      m_i( i, j, k )
                : zeros;

        // Add boundary conditions last.
        bc.apply( u_i, i, j, k );
    }
};

//---------------------------------------------------------------------------//
// DamBreak example
template <class InterpolationType, class ParticleVelocity>
void DamBreak( std::string filename )
{
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = exec_space::memory_space;

    // Get inputs for mesh.
    auto inputs = Picasso::parse( filename );

    // Global bounding box.
    auto global_box = copy<double, 6>( inputs["global_box"] );
    int minimum_halo_size = 0;

    // Make mesh.
    using mesh_type = Picasso::UniformMesh<memory_space>;
    std::shared_ptr<mesh_type> mesh = Picasso::createUniformMesh(
        memory_space(), inputs, global_box, minimum_halo_size, MPI_COMM_WORLD );

    // Make a particle list.
    Cabana::ParticleTraits<Picasso::Field::Stress, ParticleVelocity,
                           Picasso::Field::Position, Picasso::Field::Mass,
                           Picasso::Field::Pressure, Picasso::Field::Volume,
                           Picasso::Field::DetDefGrad>
        fields;
    auto particles = Cabana::Grid::createParticleList<memory_space>(
        "test_particles", fields );

    // Initialize particles
    auto particle_box = copy<double, 6>( inputs["particle_box"] );
    double density = inputs["density"];
    ParticleInitFunc<decltype( particles ), ParticleVelocity>
        momentum_init_functor{ particle_box, density };

    double ppc = inputs["ppc"];
    auto local_grid = mesh->localGrid();
    Cabana::Grid::createParticles( Cabana::InitUniform(), exec_space{},
                                   momentum_init_functor, particles, ppc,
                                   *local_grid );

    // Boundary index space
    auto index_space = local_grid->indexSpace(
        Cabana::Grid::Own(), Cabana::Grid::Node(), Cabana::Grid::Local() );
    auto global_grid = local_grid->globalGrid();

    Kokkos::Array<long int, 6> bc_index_space{
        index_space.min( Cabana::Grid::Dim::I ),
        index_space.min( Cabana::Grid::Dim::J ),
        index_space.min( Cabana::Grid::Dim::K ),
        index_space.max( Cabana::Grid::Dim::I ) - 1,
        index_space.max( Cabana::Grid::Dim::J ) - 1,
        index_space.max( Cabana::Grid::Dim::K ) - 1 };

    Kokkos::Array<bool, 6> on_boundary{
        global_grid.onLowBoundary( Cabana::Grid::Dim::I ),
        global_grid.onLowBoundary( Cabana::Grid::Dim::J ),
        global_grid.onLowBoundary( Cabana::Grid::Dim::K ),
        global_grid.onHighBoundary( Cabana::Grid::Dim::I ),
        global_grid.onHighBoundary( Cabana::Grid::Dim::J ),
        global_grid.onHighBoundary( Cabana::Grid::Dim::K ) };

    BoundaryCondition bc{ bc_index_space, on_boundary };

    // Properties
    double gamma = inputs["gamma"];
    double bulk_modulus = inputs["bulk_modulus"];
    auto gravity = inputs["gravity"];

    // Time integragor inputs.
    double dt = inputs["dt"];
    // Only used for FLIP.
    double beta = inputs["beta"];
    const int spline_order = 1;

    auto fm = Picasso::createFieldManager( mesh );

    // Field locations and variables.
    using node_type = Picasso::FieldLocation::Node;
    using grid_type =
        Picasso::FieldLayout<node_type, Picasso::Field::PhysicalPosition<3>>;
    using particle_type = Picasso::FieldLocation::Particle;
    using mass_type = Picasso::FieldLayout<node_type, Picasso::Field::Mass>;
    using velocity_type =
        Picasso::FieldLayout<node_type, Picasso::Field::Velocity>;
    using old_u_type = Picasso::FieldLayout<node_type, Picasso::Field::OldU>;
    using delta_u_s_type = Picasso::FieldLayout<node_type, DeltaUStress>;
    using delta_u_g_type = Picasso::FieldLayout<node_type, DeltaUGravity>;

    // Define particle/grid parallel data dependencies.
    Picasso::GridOperator<mesh_type,
                          Picasso::ScatterDependencies<mass_type, old_u_type>>
        p2g( mesh );
    p2g.setup( *fm );

    Picasso::GridOperator<mesh_type, Picasso::LocalDependencies<
                                         mass_type, velocity_type, old_u_type>>
        compute_velocity( mesh );
    compute_velocity.setup( *fm );

    Picasso::GridOperator<mesh_type, Picasso::GatherDependencies<velocity_type>>
        compute_stress( mesh );
    compute_stress.setup( *fm );

    Picasso::GridOperator<
        mesh_type, Picasso::ScatterDependencies<delta_u_s_type, delta_u_g_type>>
        compute_velocity_change( mesh );
    compute_velocity_change.setup( *fm );

    Picasso::GridOperator<
        mesh_type, Picasso::GatherDependencies<mass_type, velocity_type,
                                               delta_u_s_type, delta_u_g_type>>
        update_velocity( mesh );
    update_velocity.setup( *fm );

    Picasso::GridOperator<
        mesh_type,
        Picasso::GatherDependencies<mass_type, velocity_type, old_u_type>,
        Picasso::LocalDependencies<grid_type>>
        g2p( mesh );
    g2p.setup( *fm );

    // Timestep loop.
    double time = 0.0;
    auto final_time = inputs["final_time"];
    int write_frequency = inputs["write_frequency"];
    int steps = 0;
    while ( time < final_time )
    {
        // Particle interpolation (Picasso built-in).
        Picasso::Particle2Grid<spline_order, ParticleVelocity,
                               Picasso::Field::OldU, InterpolationType>
            p2g_func{ dt };
        p2g.apply( "p2g_u", particle_type{}, exec_space{}, *fm, particles,
                   p2g_func );

        // Compute grid velocity (Picasso built-in).
        Picasso::ComputeGridVelocity compute_u_func;
        compute_velocity.apply( "grid_U", node_type{}, exec_space{}, *fm,
                                compute_u_func );

        // Update particle stress.
        ComputeParticlePressure<spline_order> compute_s_func{ dt, bulk_modulus,
                                                              gamma };
        compute_stress.apply( "particle_S", particle_type{}, exec_space{}, *fm,
                              particles, compute_s_func );

        // Compute grid velocity change due to stress
        ComputeGridVelocityChange<spline_order> compute_du_s_func( gravity );
        compute_velocity_change.apply( "div_S", particle_type{}, exec_space{},
                                       *fm, particles, compute_du_s_func );

        // Compute next grid velocity for gravity/stress.
        UpdateGridVelocity update_u_func{ dt, bc };
        update_velocity.apply( "update_U", node_type{}, exec_space{}, *fm,
                               update_u_func );

        // Grid interpolation (Picasso built-in).
        Picasso::Grid2ParticleVelocity<spline_order, InterpolationType>
            g2p_func{ dt, beta };
        g2p.apply( "g2p_U", particle_type{}, exec_space{}, *fm, particles,
                   g2p_func );

        // Redistribution of particles across MPI subdomains.
        particles.redistribute( *local_grid, Picasso::Field::Position() );

        // Write particle fields.
        Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
        if ( steps % write_frequency == 0 )
        {
            Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
                h5_config, "particles", global_grid.comm(),
                steps / write_frequency, time, particles.size(),
                particles.slice( Picasso::Field::Position() ),
                particles.slice( Picasso::Field::Pressure() ),
                particles.slice( ParticleVelocity() ),
                particles.slice( Picasso::Field::Mass() ),
                particles.slice( Picasso::Field::Volume() ) );

            // Calculate conservation sums
            double mass_particles = 0.0;
            double ke_particles = 0.0;
            Picasso::particleConservation( global_grid.comm(), exec_space(),
                                           particles, ParticleVelocity(),
                                           mass_particles, ke_particles );
            double mass_grid = 0.0;
            double ke_grid = 0.0;
            Picasso::gridConservation( global_grid.comm(), exec_space(), mesh,
                                       *fm, mass_grid, ke_grid );

            if ( global_grid.blockId() == 0 )
                std::cout << "Particle/Grid Mass: " << mass_particles << " / "
                          << mass_grid << "\n"
                          << "Particle/Grid Kinetic Energy: " << ke_particles
                          << " / " << ke_grid << "\n\n";
        }
        time += dt;
        steps++;
    }
}

int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    if ( argc < 2 )
        throw std::runtime_error( "Incorrect number of arguments. \n \
            First argument - file name for output \n \
            \n \
            Example: \n \
            $/: ./DamBreak inputs/dam_break.json\n" );
    std::string filename = argv[1];

    // Problem can run with any interpolation scheme.
    // DamBreak<PolyPicTag, Picasso::PolyPIC::Field::Velocity>( filename);
    // DamBreak<APicTag, Picasso::APIC::Field::Velocity>( filename );
    DamBreak<FlipTag, Picasso::Field::Velocity>( filename );

    Kokkos::finalize();
    MPI_Finalize();

    return 0;
}
