#include <Harlow_DenseLinearAlgebra.hpp>
#include <Harlow_ParticleCommunication.hpp>
#include <Harlow_SiloParticleWriter.hpp>
#include <Harlow_VelocityInterpolation.hpp>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <mpi.h>

#include <cstdlib>
#include <array>
#include <cmath>

//---------------------------------------------------------------------------//
struct ParticleField
{
    enum Values
    {
        x = 0, // position
        m = 1  // mass
    };
};

struct TracerField
{
    enum Values
    {
        x = 0,    // position
        c = 1     // color
    };
};

//---------------------------------------------------------------------------//
void solve( const int num_cell,
            const int ppc,
            const int tpc,
            const double t_final,
            const double delta_t )
{
    // Fluid density.
    double density = 1.0;

    // PI
    double pi = 4.0 * atan( 1.0 );

    // Types.
    using execution_space = Kokkos::Serial;
    using memory_space = Kokkos::HostSpace;
    using device_type = Kokkos::Device<execution_space,memory_space>;
    using Cajita::Dim;

    // Spline discretization.
    const int spline_order = 1;
    using sd_type_i = Cajita::SplineData<double,spline_order,Cajita::Face<Dim::I>>;
    using sd_type_j = Cajita::SplineData<double,spline_order,Cajita::Face<Dim::J>>;
    using sd_type_k = Cajita::SplineData<double,spline_order,Cajita::Face<Dim::K>>;

    // Number of polypic modes
    const int num_mode = 8;

    // Build the global mesh.
    const double cell_size = 2.0 / num_cell;
    const int halo_width = 1;
    const std::array<double,3> global_low_corner = { -1.0, -1.0, 0.0 };
    const std::array<double,3> global_high_corner = { 1.0, 1.0, cell_size };
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Global grid.
    const std::array<bool,3> periodic = {false,false,true};
    auto global_grid = Cajita::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, periodic, Cajita::UniformDimPartitioner() );

    // Local grid.
    auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );

    // Local mesh.
    auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

    // Index spaces.
    auto cell_space =
        local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );
    auto face_i_space =
        local_grid->indexSpace( Cajita::Own(), Cajita::Face<Dim::I>(), Cajita::Local() );
    auto face_j_space =
        local_grid->indexSpace( Cajita::Own(), Cajita::Face<Dim::J>(), Cajita::Local() );
    auto face_k_space =
        local_grid->indexSpace( Cajita::Own(), Cajita::Face<Dim::K>(), Cajita::Local() );

    // A reduced space neglecting the outer-most cell.
    Cajita::IndexSpace<3> reduced_cell_space(
        {cell_space.min(Dim::I)+1,cell_space.min(Dim::J)+1,cell_space.min(Dim::K)},
        {cell_space.max(Dim::I)-1,cell_space.max(Dim::J)-1,cell_space.max(Dim::K)} );

    // Face fields.
    // 0: mass
    // 1: momentum
    // 2: velocity
    // 3: pressure gradient
    auto face_i_layout =
        Cajita::createArrayLayout( local_grid, 4, Cajita::Face<Dim::I>() );
    auto face_i_fields =
        Cajita::createArray<double,device_type>( "face_i_fields", face_i_layout );
    auto m_i = Cajita::createSubarray( *face_i_fields, 0, 1 );
    auto mu_i = Cajita::createSubarray( *face_i_fields, 1, 2 );
    auto u_i = Cajita::createSubarray( *face_i_fields, 2, 3 );
    auto pg_i = Cajita::createSubarray( *face_i_fields, 3, 4 );

    auto scatter_i = Cajita::createSubarray( *face_i_fields, 0, 2 );
    auto scatter_halo_i = Cajita::createHalo( *scatter_i, Cajita::FullHaloPattern() );

    auto velocity_halo_i = Cajita::createHalo( *u_i, Cajita::FullHaloPattern() );

    auto face_j_layout =
        Cajita::createArrayLayout( local_grid, 4, Cajita::Face<Dim::J>() );
    auto face_j_fields =
        Cajita::createArray<double,device_type>( "face_j_fields", face_j_layout );
    auto m_j = Cajita::createSubarray( *face_j_fields, 0, 1 );
    auto mu_j = Cajita::createSubarray( *face_j_fields, 1, 2 );
    auto u_j = Cajita::createSubarray( *face_j_fields, 2, 3 );
    auto pg_j = Cajita::createSubarray( *face_j_fields, 3, 4 );

    auto scatter_j = Cajita::createSubarray( *face_j_fields, 0, 2 );
    auto scatter_halo_j = Cajita::createHalo( *scatter_j, Cajita::FullHaloPattern() );

    auto velocity_halo_j = Cajita::createHalo( *u_j, Cajita::FullHaloPattern() );

    auto face_k_layout =
        Cajita::createArrayLayout( local_grid, 4, Cajita::Face<Dim::K>() );
    auto face_k_fields =
        Cajita::createArray<double,device_type>( "face_k_fields", face_k_layout );
    auto m_k = Cajita::createSubarray( *face_k_fields, 0, 1 );
    auto mu_k = Cajita::createSubarray( *face_k_fields, 1, 2 );
    auto u_k = Cajita::createSubarray( *face_k_fields, 2, 3 );
    auto pg_k = Cajita::createSubarray( *face_k_fields, 3, 4 );

    auto scatter_k = Cajita::createSubarray( *face_k_fields, 0, 2 );
    auto scatter_halo_k = Cajita::createHalo( *scatter_k, Cajita::FullHaloPattern() );

    auto velocity_halo_k = Cajita::createHalo( *u_k, Cajita::FullHaloPattern() );

    // Cell fields.
    // 0: velocity divergence
    // 1: pressure
    auto cell_layout =
        Cajita::createArrayLayout( local_grid, 1, Cajita::Cell() );
    auto p_c =
        Cajita::createArray<double,device_type>( "pressure", cell_layout );
    auto du_c =
        Cajita::createArray<double,device_type>( "velocity_divergence", cell_layout );

    auto cell_halo = Cajita::createHalo( *p_c, Cajita::FullHaloPattern() );

    // Views.
    auto m_i_view = m_i->view();
    auto mu_i_view = mu_i->view();
    auto u_i_view = u_i->view();
    auto pg_i_view = pg_i->view();

    auto m_j_view = m_j->view();
    auto mu_j_view = mu_j->view();
    auto u_j_view = u_j->view();
    auto pg_j_view = pg_j->view();

    auto m_k_view = m_k->view();
    auto mu_k_view = mu_k->view();
    auto u_k_view = u_k->view();
    auto pg_k_view = pg_k->view();

    auto p_c_view = p_c->view();
    auto du_c_view = du_c->view();

    // Initialize fields to zero.
    Cajita::ArrayOp::assign( *face_i_fields, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *face_j_fields, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *face_k_fields, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *p_c, 0.0, Cajita::Ghost() );
    Cajita::ArrayOp::assign( *du_c, 0.0, Cajita::Ghost() );

    // Particle data types.
    using ParticleTypes =
        Cabana::MemberTypes<double[3], // position
                            double>;    // mass

    // Particles.
    auto num_particle = reduced_cell_space.size() * ppc;
    Cabana::AoSoA<ParticleTypes,device_type> particles( "particles", num_particle );
    auto x_p = Cabana::slice<ParticleField::x>( particles, "position" );
    auto m_p = Cabana::slice<ParticleField::m>( particles, "mass" );

    // Tracer data types.
    using TracerTypes =
        Cabana::MemberTypes<double[3], // position
                            int>;      // color

    // Tracers.
    auto num_tracer = reduced_cell_space.size() * tpc;
    Cabana::AoSoA<TracerTypes,device_type> tracers( "tracers", num_tracer );
    auto x_t = Cabana::slice<TracerField::x>( tracers, "position" );
    auto c_t = Cabana::slice<TracerField::c>( tracers, "color" );

    // Initialize particle positions and tracers. Initialize them randomly in the cell.
    // Also initialize particle masses and grid velocity.
    {
        auto m_i_sv = Kokkos::Experimental::create_scatter_view( m_i_view );
        auto mu_i_sv = Kokkos::Experimental::create_scatter_view( mu_i_view );
        auto m_j_sv = Kokkos::Experimental::create_scatter_view( m_j_view );
        auto mu_j_sv = Kokkos::Experimental::create_scatter_view( mu_j_view );
        auto m_k_sv = Kokkos::Experimental::create_scatter_view( m_k_view );
        auto mu_k_sv = Kokkos::Experimental::create_scatter_view( mu_k_view );
        uint64_t seed = global_grid->blockId() + ( 19383747 % (global_grid->blockId() + 1) );
        using rnd_type = Kokkos::Random_XorShift64_Pool<device_type>;
        rnd_type pool;
        pool.init( seed, reduced_cell_space.size() );
        Kokkos::parallel_for(
            "init_particle_location",
            Cajita::createExecutionPolicy( reduced_cell_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){
                // Compute the owned local cell id.
                int i_own = i - reduced_cell_space.min(Dim::I);
                int j_own = j - reduced_cell_space.min(Dim::J);
                int k_own = k - reduced_cell_space.min(Dim::K);
                int cell_id = i_own + reduced_cell_space.extent(Dim::I) * (
                    j_own + k_own * reduced_cell_space.extent(Dim::J) );

                // Get the coordinates of the low cell node.
                int low_node[3] = {i,j,k};
                double low_coords[3];
                local_mesh.coordinates( Cajita::Node(), low_node, low_coords );

                // Get the coordinates of the high cell node.
                int high_node[3] = {i+1,j+1,k+1};
                double high_coords[3];
                local_mesh.coordinates( Cajita::Node(), high_node, high_coords );

                // Particle volume.
                double p_vol = local_mesh.measure( Cajita::Cell(), low_node ) / ppc;

                // Particle mass.
                double p_mass = density * p_vol;

                // Random number generator.
                auto rand = pool.get_state(cell_id);

                // Create particles.
                for ( int p = 0; p < ppc; ++p )
                {
                    // Particle id.
                    int pid = cell_id * ppc + p;

                    // Put particles randomly in the cell.
                    for ( int d = 0; d < 3; ++d )
                    {
                        x_p(pid,d) =
                            Kokkos::rand<decltype(rand),double>::draw( rand, low_coords[d], high_coords[d] );
                    }
                    double px[3] = { x_p(pid,Dim::I), x_p(pid,Dim::J), x_p(pid,Dim::K) };

                    // Particle mass.
                    m_p(pid) = p_mass;

                    // Particle momentum. Initial velocity is a vortex sheet
                    // moving at 1 radian per second.
                    double mu_p[3] = {0.0,0.0,0.0};
                    double r = px[Dim::I] * px[Dim::I] + px[Dim::J] * px[Dim::J];
                    if ( r <= 0.25 )
                    {
                        mu_p[Dim::I] = -p_mass * px[Dim::J] * pi;
                        mu_p[Dim::J] = p_mass * px[Dim::I] * pi;
                    }

                    // Interpolate initial momentum and mass to faces.
                    sd_type_i sd_i;
                    Cajita::evaluateSpline( local_mesh, px, sd_i );
                    Cajita::P2G::value( mu_p[Dim::I], sd_i, mu_i_sv );
                    Cajita::P2G::value( p_mass, sd_i, m_i_sv );

                    sd_type_j sd_j;
                    Cajita::evaluateSpline( local_mesh, px, sd_j );
                    Cajita::P2G::value( mu_p[Dim::J], sd_j, mu_j_sv );
                    Cajita::P2G::value( p_mass, sd_j, m_j_sv );

                    sd_type_k sd_k;
                    Cajita::evaluateSpline( local_mesh, px, sd_k );
                    Cajita::P2G::value( mu_p[Dim::K], sd_k, mu_k_sv );
                    Cajita::P2G::value( p_mass, sd_k, m_k_sv );
                }

                // Create tracers.
                for ( int t = 0; t < tpc; ++t )
                {
                    // Tracer id.
                    int tid = cell_id * tpc + t;

                    // Put tracers randomly in the cell.
                    for ( int d = 0; d < 3; ++d )
                    {
                        x_t(tid,d) =
                            Kokkos::rand<decltype(rand),double>::draw( rand, low_coords[d], high_coords[d] );
                    }
                    double tx[3] = { x_t(tid,Dim::I), x_t(tid,Dim::J), x_t(tid,Dim::K) };

                    // Tracer color.
                    if ( (tx[Dim::I] > 0.0 && tx[Dim::J] > 0.0) ||
                         (tx[Dim::I] < 0.0 && tx[Dim::J] < 0.0) )
                        c_t(tid) = 1;
                    else
                        c_t(tid) = 2;
                }
            });

        // Complete the particle-grid scatter to the faces.
        Kokkos::Experimental::contribute( m_i_view, m_i_sv );
        Kokkos::Experimental::contribute( mu_i_view, mu_i_sv );
        Kokkos::Experimental::contribute( m_j_view, m_j_sv );
        Kokkos::Experimental::contribute( mu_j_view, mu_j_sv );
        Kokkos::Experimental::contribute( m_k_view, m_k_sv );
        Kokkos::Experimental::contribute( mu_k_view, mu_k_sv );
        scatter_halo_i->scatter( *scatter_i );
        scatter_halo_j->scatter( *scatter_j );
        scatter_halo_k->scatter( *scatter_k );

        // Compute grid initial velocity.
        Kokkos::parallel_for(
            "compute_initial_grid_velocity_i",
            Cajita::createExecutionPolicy( face_i_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){

                // Compute face velocity
                u_i_view(i,j,k,0) =
                    ( m_i_view(i,j,k,0) > 0.0 ) ? mu_i_view(i,j,k,0) / m_i_view(i,j,k,0) : 0.0;

                // Apply velocity boundary condition.
                if ( i < 2 || face_i_space.max(Dim::I) - 2 <= i )
                {
                    u_i_view(i,j,k,0) = 0.0;
                }
            });
        Kokkos::parallel_for(
            "compute_initial_grid_velocity_j",
            Cajita::createExecutionPolicy( face_j_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){

                // Compute face velocity
                u_j_view(i,j,k,0) =
                    ( m_j_view(i,j,k,0) > 0.0 ) ? mu_j_view(i,j,k,0) / m_j_view(i,j,k,0) : 0.0;

                // Apply velocity boundary condition.
                if ( j < 2 || face_j_space.max(Dim::J) - 2 <= j )
                {
                    u_j_view(i,j,k,0) = 0.0;
                }
            });
        Kokkos::parallel_for(
            "compute_initial_grid_velocity_k",
            Cajita::createExecutionPolicy( face_k_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){

                // Compute face velocity
                u_k_view(i,j,k,0) =
                    ( m_k_view(i,j,k,0) > 0.0 ) ? mu_k_view(i,j,k,0) / m_k_view(i,j,k,0) : 0.0;
            });
    }

    // Create a grid solver.
    auto solver =
        Cajita::createStructuredSolver<double,device_type>( "PCG", *cell_layout );

    // Create a grid preconditioner.
    auto preconditioner =
        Cajita::createStructuredSolver<double,device_type>( "Diagonal", *cell_layout, true );
    solver->setPreconditioner( preconditioner );

    // Create a 7-point 3d laplacian stencil (symmetric).
    std::vector<std::array<int,3> > stencil =
        { {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1} };
    solver->setMatrixStencil( stencil, true );

    // Cajita::Create the laplacian matrix entries. The stencil is defined
    // over cells.
    int rdx_2 = 1.0 / (cell_size * cell_size);
    auto matrix_entry_layout = Cajita::createArrayLayout( local_grid, 4, Cajita::Cell() );
    auto matrix_entries = Cajita::createArray<double,device_type>(
        "matrix_entries", matrix_entry_layout );
    auto entry_view = matrix_entries->view();
    Kokkos::parallel_for(
        "fill_matrix_entries",
        createExecutionPolicy( cell_space, execution_space() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k ){
            entry_view(i,j,k,0) = -6.0 * rdx_2;
            entry_view(i,j,k,1) = rdx_2;
            entry_view(i,j,k,2) = rdx_2;
            entry_view(i,j,k,3) = rdx_2;
        } );

    solver->setMatrixValues( *matrix_entries );

    // Set the tolerance.
    solver->setTolerance( 1.0e-9 );

    // Set the maximum iterations.
    solver->setMaxIter( 1000 );

    // Set the print level.
    solver->setPrintLevel( 0 );

    // Setup the problem.
    solver->setup();

    // Time step.
    int num_step = t_final / delta_t;
    double time = 0.0;
    for ( int t = 0; t < num_step; ++t )
    {
        std::cout << "Step " << t+1 << "/" << num_step << " - time " << time << std::endl;

        // Output particles
        Harlow::SiloParticleWriter::writeTimeStep( *global_grid, t, time, x_t, c_t );

        // Gather grid values.
        velocity_halo_i->gather( *u_i );
        velocity_halo_j->gather( *u_j );
        velocity_halo_k->gather( *u_k );

        // Move Tracers.
        Kokkos::parallel_for(
            "move_tracers",
            Kokkos::RangePolicy<execution_space>(0,num_tracer),
            KOKKOS_LAMBDA( const int t ){

                // Get the tracer position.
                double tx[3] = { x_t(t,Dim::I), x_t(t,Dim::J), x_t(t,Dim::K) };

                // Evaluate tracer splines.
                sd_type_i sd_i;
                Cajita::evaluateSpline( local_mesh, tx, sd_i );
                sd_type_j sd_j;
                Cajita::evaluateSpline( local_mesh, tx, sd_j );
                sd_type_k sd_k;
                Cajita::evaluateSpline( local_mesh, tx, sd_k );

                // Interpolate grid velocity to tracer
                double u_t[3];
                Cajita::G2P::value( u_i_view, sd_i, u_t[Dim::I] );
                Cajita::G2P::value( u_j_view, sd_j, u_t[Dim::J] );
                Cajita::G2P::value( u_k_view, sd_k, u_t[Dim::K] );

                // Move tracer.
                for ( int d = 0; d < 3; ++d )
                    x_t(t,d) += delta_t * u_t[d];
            });

        // Mass and momentum scatter views.
        auto m_i_sv = Kokkos::Experimental::create_scatter_view( m_i_view );
        auto mu_i_sv = Kokkos::Experimental::create_scatter_view( mu_i_view );
        auto m_j_sv = Kokkos::Experimental::create_scatter_view( m_j_view );
        auto mu_j_sv = Kokkos::Experimental::create_scatter_view( mu_j_view );
        auto m_k_sv = Kokkos::Experimental::create_scatter_view( m_k_view );
        auto mu_k_sv = Kokkos::Experimental::create_scatter_view( mu_k_view );

        // Reset grid mass and momentum.
        Cajita::ArrayOp::assign( *scatter_i, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *scatter_j, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *scatter_k, 0.0, Cajita::Ghost() );

        // Update particles and predict grid momentum.
        Kokkos::parallel_for(
            "particle_step",
            Kokkos::RangePolicy<execution_space>(0,num_particle),
            KOKKOS_LAMBDA( const int p ){

                // Get the particle position.
                double px[3] = { x_p(p,Dim::I), x_p(p,Dim::J), x_p(p,Dim::K) };

                // Evaluate particle spline.
                sd_type_i sd_i;
                Cajita::evaluateSpline( local_mesh, px, sd_i );
                sd_type_j sd_j;
                Cajita::evaluateSpline( local_mesh, px, sd_j );
                sd_type_k sd_k;
                Cajita::evaluateSpline( local_mesh, px, sd_k );

                // Interpolate grid velocity to particle with PolyPIC
                double u_p[num_mode][3];
                Harlow::PolyPIC::g2p<num_mode>( u_i_view, u_j_view, u_k_view,
                                                sd_i, sd_j, sd_k,
                                                u_p );

                // Move the particle.
                for ( int d = 0; d < 3; ++d )
                {
                    px[d] += delta_t * u_p[0][d];
                    x_p(p,d) = px[d];
                }

                // Re-evaluate the particle splines after the particles have moved.
                Cajita::evaluateSpline( local_mesh, px, sd_i );
                Cajita::evaluateSpline( local_mesh, px, sd_j );
                Cajita::evaluateSpline( local_mesh, px, sd_k );

                // Interpolate particle mass to the grid.
                Cajita::P2G::value( m_p(p), sd_i, m_i_sv );
                Cajita::P2G::value( m_p(p), sd_j, m_j_sv );
                Cajita::P2G::value( m_p(p), sd_k, m_k_sv );

                // Interpolate particle momentum to the grid.
                Harlow::PolyPIC::p2g<num_mode>( m_p(p), u_p,
                                                sd_i, sd_j, sd_k, delta_t,
                                                mu_i_sv, mu_j_sv, mu_k_sv );
            });

        // Complete the particle-grid scatter to the faces.
        Kokkos::Experimental::contribute( m_i_view, m_i_sv );
        Kokkos::Experimental::contribute( mu_i_view, mu_i_sv );
        Kokkos::Experimental::contribute( m_j_view, m_j_sv );
        Kokkos::Experimental::contribute( mu_j_view, mu_j_sv );
        Kokkos::Experimental::contribute( m_k_view, m_k_sv );
        Kokkos::Experimental::contribute( mu_k_view, mu_k_sv );
        scatter_halo_i->scatter( *scatter_i );
        scatter_halo_j->scatter( *scatter_j );
        scatter_halo_k->scatter( *scatter_k );

        // Compute grid velocity.
        Kokkos::parallel_for(
            "compute_initial_grid_velocity_i",
            Cajita::createExecutionPolicy( face_i_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){

                // Compute face velocity
                u_i_view(i,j,k,0) =
                    ( m_i_view(i,j,k,0) > 0.0 ) ? mu_i_view(i,j,k,0) / m_i_view(i,j,k,0) : 0.0;

                // Apply velocity boundary condition.
                if ( i < 2 || face_i_space.max(Dim::I) - 2 <= i )
                {
                    u_i_view(i,j,k,0) = 0.0;
                }
            });
        Kokkos::parallel_for(
            "compute_initial_grid_velocity_j",
            Cajita::createExecutionPolicy( face_j_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){

                // Compute face velocity
                u_j_view(i,j,k,0) =
                    ( m_j_view(i,j,k,0) > 0.0 ) ? mu_j_view(i,j,k,0) / m_j_view(i,j,k,0) : 0.0;

                // Apply velocity boundary condition.
                if ( j < 2 || face_j_space.max(Dim::J) - 2 <= j )
                {
                    u_j_view(i,j,k,0) = 0.0;
                }
            });
        Kokkos::parallel_for(
            "compute_initial_grid_velocity_k",
            Cajita::createExecutionPolicy( face_k_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){

                // Compute face velocity
                u_k_view(i,j,k,0) =
                    ( m_k_view(i,j,k,0) > 0.0 ) ? mu_k_view(i,j,k,0) / m_k_view(i,j,k,0) : 0.0;
            });
        velocity_halo_i->gather( *u_i );
        velocity_halo_j->gather( *u_j );
        velocity_halo_k->gather( *u_k );

        // Compute pressure solve rhs.
        Kokkos::parallel_for(
            "pressure_rhs",
            Cajita::createExecutionPolicy( cell_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){

                // Divergence.
                du_c_view(i,j,k,0) = ( ( u_i_view(i+1,j,k,0) - u_i_view(i,j,k,0) ) +
                                       ( u_j_view(i,j+1,k,0) - u_j_view(i,j,k,0) ) +
                                       ( u_k_view(i,j,k+1,0) - u_k_view(i,j,k,0) ) ) / cell_size;

                // Scale by density and time.
                du_c_view(i,j,k,0) *= density / delta_t;
            });

        // Solve for the pressure.
        solver->solve( *du_c, *p_c );

        // Compute pressure gradient.
        Cajita::ArrayOp::assign( *pg_i, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *pg_j, 0.0, Cajita::Ghost() );
        Cajita::ArrayOp::assign( *pg_k, 0.0, Cajita::Ghost() );
        auto pg_i_sv = Kokkos::Experimental::create_scatter_view( pg_i_view );
        auto pg_j_sv = Kokkos::Experimental::create_scatter_view( pg_j_view );
        auto pg_k_sv = Kokkos::Experimental::create_scatter_view( pg_k_view );
        Kokkos::parallel_for(
            "pressure_gradient",
            Cajita::createExecutionPolicy( cell_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){

                auto result = p_c_view(i,j,k,0) / cell_size;

                auto pg_i_access = pg_i_sv.access();
                pg_i_access(i,j,k,0) += result;
                pg_i_access(i+1,j,k,0) += -result;

                auto pg_j_access = pg_j_sv.access();
                pg_j_access(i,j,k,0) += result;
                pg_j_access(i,j+1,k,0) += -result;

                auto pg_k_access = pg_k_sv.access();
                pg_k_access(i,j,k,0) += result;
                pg_k_access(i,j,k+1,0) += -result;
            });
        Kokkos::Experimental::contribute( pg_i_view, pg_i_sv );
        Kokkos::Experimental::contribute( pg_j_view, pg_j_sv );
        Kokkos::Experimental::contribute( pg_k_view, pg_k_sv );
        velocity_halo_i->scatter( *pg_i );
        velocity_halo_j->scatter( *pg_j );
        velocity_halo_k->scatter( *pg_k );

        // Update velocity with pressure gradient.
        Cajita::ArrayOp::update( *u_i, 1.0, *pg_i, -delta_t / density, Cajita::Own() );
        Cajita::ArrayOp::update( *u_j, 1.0, *pg_j, -delta_t / density, Cajita::Own() );
        Cajita::ArrayOp::update( *u_k, 1.0, *pg_k, -delta_t / density, Cajita::Own() );

        // Redistribute particles.
        Harlow::ParticleCommunication::redistribute(
            *local_grid, particles, std::integral_constant<std::size_t,ParticleField::x>() );
        num_particle = particles.size();

        // Redistribute tracers.
        Harlow::ParticleCommunication::redistribute(
            *local_grid, tracers, std::integral_constant<std::size_t,TracerField::x>() );
        num_tracer = tracers.size();

        // Compute divergence.
        // velocity_halo_i->gather( *u_i );
        // velocity_halo_j->gather( *u_j );
        // velocity_halo_k->gather( *u_k );
        // Kokkos::parallel_for(
        //     "pressure_rhs",
        //     Cajita::createExecutionPolicy( cell_space, execution_space() ),
        //     KOKKOS_LAMBDA( const int i, const int j, const int k ){
        //         du_c_view(i,j,k,0) = ( ( u_i_view(i+1,j,k,0) - u_i_view(i,j,k,0) ) +
        //                                ( u_j_view(i,j+1,k,0) - u_j_view(i,j,k,0) ) +
        //                                ( u_k_view(i,j,k+1,0) - u_k_view(i,j,k,0) ) ) / cell_size;
        //     });
        // Cajita::BovWriter::writeTimeStep( t, time, *du_c );

        // Update time
        time += delta_t;
    }
}

//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    Kokkos::initialize( argc, argv );

    // number of cells
    int num_cell = std::atoi( argv[1] );

    // material points per cell
    int ppc = std::atoi( argv[2] );

    // tracers per cell.
    int tpc = std::atoi( argv[3] );

    // end time.
    double t_final = std::atof( argv[4] );

    // time step size
    double delta_t = std::atof( argv[5] );

    // run the problem.
    solve( num_cell, ppc, tpc, t_final, delta_t );

    Kokkos::finalize();

    MPI_Finalize();

    return 0;
}

//---------------------------------------------------------------------------//
