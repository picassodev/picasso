#include <Harlow_DenseLinearAlgebra.hpp>
#include <Harlow_ParticleCommunication.hpp>
#include <Harlow_SiloParticleWriter.hpp>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <cstdlib>
#include <array>

//---------------------------------------------------------------------------//
void nohProblem( const double cell_size,
                 const int ppc,
                 const double theta )
{
    // Types.
    using execution_space = Kokkos::Serial;
    using memory_space = Kokkos::HostSpace;
    using device_type = Kokkos::Device<execution_space,memory_space>;
    using Cajita::Dim;

    // Global mesh.
    double radius = 1.2;
    const int halo_width = 1;
    double radius_plus = radius + cell_size * halo_width;
    const std::array<double,3> global_low_corner =
        {-radius_plus,-radius_plus,-radius_plus};
    const std::array<double,3> global_high_corner =
        {radius_plus,radius_plus,radius_plus};
    auto global_mesh = Cajita::createUniformGlobalMesh(
        global_low_corner, global_high_corner, cell_size );

    // Global grid.
    const std::array<bool,3> periodic = {false,false,false};
    auto global_grid = Cajita::createGlobalGrid(
        MPI_COMM_WORLD, global_mesh, periodic, Cajita::UniformDimPartitioner() );

    // Local grid.
    auto local_grid = Cajita::createLocalGrid( global_grid, halo_width );

    // Local mesh.
    auto local_mesh = Cajita::createLocalMesh<device_type>( *local_grid );

    // Index spaces.
    auto node_space =
        local_grid->indexSpace( Cajita::Own(), Cajita::Node(), Cajita::Local() );
    auto cell_space =
        local_grid->indexSpace( Cajita::Own(), Cajita::Cell(), Cajita::Local() );

    // Grid fields. Pack nodal data such that the dofs are:
    // 0: mass
    // 1-3: particle momentum
    // 4-6: particle force
    // 7-9: particle energy
    // 10: particle pdv work
    // 11-13: old node velocity
    // 14-16: theta node velocity
    // 17-19: new node velocity
    // 20-22: old node specific internal energy
    // 23-25: theta node specific internal energy
    // 27-29: new node specific internal energy
    auto node_layout =
        Cajita::createArrayLayout( local_grid, 29, Cajita::Node() );
    auto node_fields =
        Cajita::createArray<double,device_type>( "node_fields", node_layout );
    auto m_v = Cajita::createSubarray( *node_fields, 0, 1 );
    auto mu_v = Cajita::createSubarray( *node_fields, 1, 4 );
    auto f_v = Cajita::createSubarray( *node_fields, 4, 7 );
    auto me_v = Cajita::createSubarray( *node_fields, 7, 10 );
    auto pdv_v = Cajita::createSubarray( *node_fields, 10, 11 );
    auto u_v_old = Cajita::createSubarray( *node_fields, 11, 14 );
    auto u_v_theta = Cajita::createSubarray( *node_fields, 14, 17 );
    auto u_v_new = Cajita::createSubarray( *node_fields, 17, 20 );
    auto e_v_old = Cajita::createSubarray( *node_fields, 20, 23 );
    auto e_v_theta = Cajita::createSubarray( *node_fields, 23, 26 );
    auto e_v_new = Cajita::createSubarray( *node_fields, 26, 29 );

    // Scatter communication plan. Mass, momentum, impulse, and PdV work get scattered
    auto scatter_fields = Cajita::createSubarray( *node_fields, 0, 11 );
    auto scatter_halo = Cajita::createHalo( *scatter_fields, Cajita::FullHaloPattern() );

    // Gather communication plan. Velocities and internal energies are gathered.
    auto gather_fields = Cajita::createSubarray( *node_fields, 11, 29 );
    auto gather_halo = Cajita::createHalo( *gather_fields, Cajita::FullHaloPattern() );

    // Particle data types.
    enum Field
    {
        u = 0,     // velocity
        x = 1,     // position
        e = 2,     // specific internal energy
        m = 3,     // mass
        v = 4,     // volume
        p = 5,     // pressure
        J = 6,     // deformation gradient determinant
        d = 7      // density
    };
    using ParticleTypes =
        Cabana::MemberTypes<double[3], // velocity
                            double[3], // position
                            double,    // specific internal energy
                            double,    // mass
                            double,    // volume
                            double,    // pressure
                            double,    // deformation gradient determinant
                            double>;   // density

    // Geometry query. This problem is in a sphere centered at the origin.
    auto inside_sphere =
        KOKKOS_LAMBDA( const double x, const double y, const double z )
        {
            return ( (x*x + y*y + z*z) <= radius * radius );
        };

    // Create the particle positions.
    Kokkos::Array<double,3> low_node;
    std::array<int,3> low_idx = {0,0,0};
    local_mesh.coordinates( Cajita::Node(), low_idx.data(), low_node.data() );
    double subcell_size = cell_size / ppc;
    std::vector<double> x_coords, y_coords, z_coords;
    for ( int ic = cell_space.min(Dim::I);
          ic < cell_space.max(Dim::I); ++ic )
        for ( int jc = cell_space.min(Dim::J);
              jc < cell_space.max(Dim::J); ++jc )
            for ( int kc = cell_space.min(Dim::K);
                  kc < cell_space.max(Dim::K); ++kc )
            {
                for ( int ip = 0; ip < ppc; ++ip )
                {
                    double xp = (0.5 + ip) * subcell_size + ic * cell_size + low_node[Dim::I];
                    for ( int jp = 0; jp < ppc; ++jp )
                    {
                        double yp = (0.5 + jp) * subcell_size + jc * cell_size + low_node[Dim::J];
                        for ( int kp = 0; kp < ppc; ++kp )
                        {
                            double zp = (0.5 + kp) * subcell_size + kc * cell_size +low_node[Dim::K];
                            if ( inside_sphere(xp,yp,zp) )
                            {
                                x_coords.push_back(xp);
                                y_coords.push_back(yp);
                                z_coords.push_back(zp);
                            }
                        }
                    }
                }
            }

    // Allocate the particles.
    int local_num_p = x_coords.size();
    Cabana::AoSoA<ParticleTypes,device_type>
        particles( "particles", local_num_p );

    // Host mirror.
    auto host_particles =
        Cabana::create_mirror_view( Kokkos::HostSpace(), particles );
    auto u_p_host = Cabana::slice<Field::u>( host_particles, "u_p_host" );
    auto x_p_host = Cabana::slice<Field::x>( host_particles, "x_p_host" );
    auto e_p_host = Cabana::slice<Field::e>( host_particles, "e_p_host" );
    auto m_p_host = Cabana::slice<Field::m>( host_particles, "m_p_host" );
    auto v_p_host = Cabana::slice<Field::v>( host_particles, "v_p_host" );
    auto p_p_host = Cabana::slice<Field::p>( host_particles, "p_p_host" );
    auto J_p_host = Cabana::slice<Field::J>( host_particles, "J_p_host" );
    auto d_p_host = Cabana::slice<Field::d>( host_particles, "d_p_host" );

    // Initialize particles.
    double u_max = 0.0;
    for ( int p = 0; p < local_num_p; ++p )
    {
        // Position
        x_p_host(p,Dim::I) = x_coords[p];
        x_p_host(p,Dim::J) = y_coords[p];
        x_p_host(p,Dim::K) = z_coords[p];

        // Velocity
        double r = std::sqrt( x_coords[p] * x_coords[p] +
                              y_coords[p] * y_coords[p] +
                              z_coords[p] * z_coords[p] );
        u_p_host(p,Dim::I) = - x_coords[p] / r;
        u_p_host(p,Dim::J) = - y_coords[p] / r;
        u_p_host(p,Dim::K) = - z_coords[p] / r;

        // Compute maximum.
        double u_mag = u_p_host(p,Dim::I) * u_p_host(p,Dim::I) +
                       u_p_host(p,Dim::J) * u_p_host(p,Dim::J) +
                       u_p_host(p,Dim::K) * u_p_host(p,Dim::K);
        if ( u_mag > u_max ) u_max = u_mag;
    }

    double p_volume = subcell_size * subcell_size * subcell_size;
    Cabana::deep_copy( v_p_host, p_volume );

    double density_0 = 1.0;
    Cabana::deep_copy( m_p_host, density_0 * p_volume );
    Cabana::deep_copy( d_p_host, density_0 );

    double energy_0 = 1.0e-12;
    Cabana::deep_copy( e_p_host, energy_0 / (density_0 * p_volume) );

    double pressure_0 = (2.0/3.0) * 1.0e-12;
    Cabana::deep_copy( p_p_host, pressure_0 / density_0 );

    Cabana::deep_copy( J_p_host, 1.0 );

    // Copy to device.
    Cabana::deep_copy( particles, host_particles );

    // Slice.
    auto u_p = Cabana::slice<Field::u>( particles, "velocity" );
    auto x_p = Cabana::slice<Field::x>( particles, "position" );
    auto e_p = Cabana::slice<Field::e>( particles, "energy" );
    auto m_p = Cabana::slice<Field::m>( particles, "mass" );
    auto v_p = Cabana::slice<Field::v>( particles, "volume" );
    auto p_p = Cabana::slice<Field::p>( particles, "pressure" );
    auto J_p = Cabana::slice<Field::J>( particles, "J" );
    auto d_p = Cabana::slice<Field::d>( particles, "density" );

    // Equation of state.
    double gamma = 5.0 / 3.0;
    auto eos =
        KOKKOS_LAMBDA( const double density, const double energy )
        {
            return (gamma - 1.0) * energy * density;
        };

    // Get data for particle/grid operations.
    double rdx = 1.0 / cell_size;
    using Basis = Cajita::Spline<2>;

    // Get grid views.
    auto m_v_view = m_v->view();
    auto mu_v_view = mu_v->view();
    auto f_v_view = f_v->view();
    auto me_v_view = me_v->view();
    auto pdv_v_view = pdv_v->view();
    auto u_v_old_view = u_v_old->view();
    auto u_v_theta_view = u_v_theta->view();
    auto u_v_new_view = u_v_new->view();
    auto e_v_old_view = e_v_old->view();
    auto e_v_theta_view = e_v_theta->view();
    auto e_v_new_view = e_v_new->view();

    // Time step parameters.
    double t_final = 0.6;
    double delta_t = 0.5 * cell_size / std::sqrt(u_max);
    int num_step = t_final / delta_t;
    double time = 0;

    // Initialize grid data.
    {
        // Reset deposit views.
        Kokkos::deep_copy( m_v_view, 0.0 );
        Kokkos::deep_copy( mu_v_view, 0.0 );
        Kokkos::deep_copy( f_v_view, 0.0 );
        Kokkos::deep_copy( me_v_view, 0.0 );
        Kokkos::deep_copy( pdv_v_view, 0.0 );

        // Create scatter views.
        auto m_v_sv =
            Kokkos::Experimental::create_scatter_view( m_v_view );
        auto mu_v_sv =
            Kokkos::Experimental::create_scatter_view( mu_v_view );
        auto me_v_sv =
            Kokkos::Experimental::create_scatter_view( me_v_view );

        // Do particle loop.
        Kokkos::parallel_for(
            "particle_loop",
            Kokkos::RangePolicy<execution_space>(0,local_num_p),
            KOKKOS_LAMBDA( const int p ){

                // Update the logical position, stencil, and weights.
                double lx[3];
                int stencil[3][Basis::num_knot];
                double w[3][Basis::num_knot];
                for ( int d = 0; d < 3; ++d )
                {
                    lx[d] = Basis::mapToLogicalGrid( x_p(p,d), rdx, low_node[d] );
                    Basis::stencil( lx[d], stencil[d] );
                    Basis::value( lx[d], w[d] );
                }

                // Access scatter view data.
                auto m_v_sv_access = m_v_sv.access();
                auto mu_v_sv_access = mu_v_sv.access();
                auto me_v_sv_access = me_v_sv.access();

                // Update grid values (P2G).
                double wm;
                for ( int i = 0; i < Basis::num_knot; ++i )
                    for ( int j = 0; j < Basis::num_knot; ++j )
                        for ( int k = 0; k < Basis::num_knot; ++k )
                        {
                            // Node weight times mass
                            wm = m_p(p) * w[Dim::I][i] * w[Dim::J][j] * w[Dim::K][k];

                            // Deposit mass
                            m_v_sv_access(stencil[Dim::I][i],
                                          stencil[Dim::J][j],
                                          stencil[Dim::K][k],
                                          0 ) += wm;

                            // Deposit energy.
                            me_v_sv_access(stencil[Dim::I][i],
                                           stencil[Dim::J][j],
                                           stencil[Dim::K][k],
                                           0) += wm * e_p(p);

                            // Deposit momentum
                            for ( int d = 0; d < 3; ++d )
                                mu_v_sv_access(stencil[Dim::I][i],
                                               stencil[Dim::J][j],
                                               stencil[Dim::K][k],
                                               d) +=
                                    wm * u_p(p,d);
                        }
            });

        // Scatter deposited values
        Kokkos::Experimental::contribute( m_v_view, m_v_sv );
        Kokkos::Experimental::contribute( mu_v_view, mu_v_sv );
        scatter_halo->scatter( *scatter_fields );

        // Compute grid data.
        Kokkos::parallel_for(
            "compute_init_grid_data",
            Cajita::createExecutionPolicy( node_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){
                if ( m_v_view(i,j,k,0) > 0.0 )
                {
                    // velocity
                    for ( int d = 0; d < 3; ++d )
                    {
                        u_v_old_view(i,j,k,d) = mu_v_view(i,j,k,d) / m_v_view(i,j,k,0);
                        u_v_new_view(i,j,k,d) = u_v_old_view(i,j,k,d);
                        u_v_theta_view(i,j,k,d) = u_v_old_view(i,j,k,d);
                    }

                    // specific internal energy
                    e_v_old_view(i,j,k,0) = me_v_view(i,j,k,0) / m_v_view(i,j,k,0);
                    e_v_new_view(i,j,k,0) = e_v_old_view(i,j,k,0);
                    e_v_theta_view(i,j,k,0) = e_v_old_view(i,j,k,0);
                }
                else
                {
                    // velocity
                    for ( int d = 0; d < 3; ++d )
                    {
                        u_v_old_view(i,j,k,d) = 0.0;
                        u_v_theta_view(i,j,k,d) = 0.0;
                        u_v_new_view(i,j,k,d) = 0.0;
                    }

                    // specific internal energy
                    e_v_old_view(i,j,k,0) = 0.0;
                    e_v_theta_view(i,j,k,0) = 0.0;
                    e_v_new_view(i,j,k,0) = 0.0;
                }
            });
    }

    // Initial output.
    Harlow::SiloParticleWriter::writeTimeStep(
        *global_grid, 0, time, x_p,
        u_p, e_p, J_p, p_p, d_p, v_p );
    Cajita::BovWriter::writeTimeStep( 0, time, *u_v_new );

    // Time step.
    for ( double t = 0; t < num_step; ++t )
    {
        std::cout << "Step " << t << " of " << num_step << std::endl;
        // Gather grid velocities.
        gather_halo->gather( *gather_fields );

        // Reset deposit views.
        Kokkos::deep_copy( m_v_view, 0.0 );
        Kokkos::deep_copy( mu_v_view, 0.0 );
        Kokkos::deep_copy( f_v_view, 0.0 );
        Kokkos::deep_copy( me_v_view, 0.0 );
        Kokkos::deep_copy( pdv_v_view, 0.0 );

        // Create scatter views.
        auto m_v_sv =
            Kokkos::Experimental::create_scatter_view( m_v_view );
        auto mu_v_sv =
            Kokkos::Experimental::create_scatter_view( mu_v_view );
        auto f_v_sv =
            Kokkos::Experimental::create_scatter_view( f_v_view );
        auto me_v_sv =
            Kokkos::Experimental::create_scatter_view( me_v_view );
        auto pdv_v_sv =
            Kokkos::Experimental::create_scatter_view( pdv_v_view );

        // Do particle loop.
        local_num_p = particles.size();
        Kokkos::parallel_for(
            "particle_loop",
            Kokkos::RangePolicy<execution_space>(0,local_num_p),
            KOKKOS_LAMBDA( const int p ){

                // Update the logical position, stencil, and weights.
                double lx[3];
                int stencil[3][Basis::num_knot];
                double w[3][Basis::num_knot];
                double gw[3][Basis::num_knot];
                for ( int d = 0; d < 3; ++d )
                {
                    lx[d] = Basis::mapToLogicalGrid( x_p(p,d), rdx, low_node[d] );
                    Basis::stencil( lx[d], stencil[d] );
                    Basis::value( lx[d], w[d] );
                    Basis::gradient( lx[d], rdx, gw[d] );
                }

                // Update particle values (G2P).
                double weight;
                double grad_weight[3];
                double div_u = 0;
                double grad_u[3][3] = {{0.0,0.0,0.0},
                                       {0.0,0.0,0.0},
                                       {0.0,0.0,0.0}};
                for ( int i = 0; i < Basis::num_knot; ++i )
                    for ( int j = 0; j < Basis::num_knot; ++j )
                        for ( int k = 0; k < Basis::num_knot; ++k )
                        {
                            // Node weight.
                            weight =
                                w[Dim::I][i] * w[Dim::J][j] * w[Dim::K][k];

                            // Node gradient.
                            grad_weight[Dim::I] =
                                gw[Dim::I][i] * w[Dim::J][j] * w[Dim::K][k];
                            grad_weight[Dim::J] =
                                w[Dim::I][i] * gw[Dim::J][j] * w[Dim::K][k];
                            grad_weight[Dim::K] =
                                w[Dim::I][i] * w[Dim::J][j] * gw[Dim::K][k];

                            for ( int d = 0; d < 3; ++d )
                            {
                                // Update position
                                x_p(p,d) += delta_t *
                                            weight *
                                            u_v_theta_view(stencil[Dim::I][i],
                                                           stencil[Dim::J][j],
                                                           stencil[Dim::K][k],
                                                           d );

                                // Update velocity.
                                u_p(p,d) += weight * ( u_v_new_view(stencil[Dim::I][i],
                                                                    stencil[Dim::J][j],
                                                                    stencil[Dim::K][k],
                                                                    d ) -
                                                       u_v_old_view(stencil[Dim::I][i],
                                                                    stencil[Dim::J][j],
                                                                    stencil[Dim::K][k],
                                                                    d) );

                                // Update specific internal energy.
                                e_p(p) += weight * ( e_v_new_view(stencil[Dim::I][i],
                                                                  stencil[Dim::J][j],
                                                                  stencil[Dim::K][k],
                                                                  d ) -
                                                     e_v_old_view(stencil[Dim::I][i],
                                                                  stencil[Dim::J][j],
                                                                  stencil[Dim::K][k],
                                                                  d) );

                                // Velocity gradient
                                for ( int d1 = 0; d1 < 3; ++d1 )
                                    grad_u[d][d1] += grad_weight[d] *
                                                     u_v_theta_view(stencil[Dim::I][i],
                                                                    stencil[Dim::J][j],
                                                                    stencil[Dim::K][k],
                                                                    d1 );

                                // Velocity divergence.
                                div_u += grad_weight[d] *
                                         u_v_theta_view(stencil[Dim::I][i],
                                                        stencil[Dim::J][j],
                                                        stencil[Dim::K][k],
                                                        d );
                            }
                        }

                // Update particle deformation gradient determinant.
                J_p(p) *= exp( delta_t * Harlow::DenseLinearAlgebra::determinant(grad_u) );

                // Store the current particle density.
                d_p(p) = m_p(p) / ( v_p(p) * J_p(p) );

                // Update particle pressure
                p_p(p) = eos( d_p(p), e_p(p) );

                // Volume weighted pressure.
                double pv = p_p(p) * J_p(p) * v_p(p);

                // Update the logical position, stencil, and weights.
                for ( int d = 0; d < 3; ++d )
                {
                    lx[d] = Basis::mapToLogicalGrid( x_p(p,d), rdx, low_node[d] );
                    Basis::stencil( lx[d], stencil[d] );
                    Basis::value( lx[d], w[d] );
                    Basis::gradient( lx[d], rdx, gw[d] );
                }

                // Access scatter view data.
                auto m_v_sv_access = m_v_sv.access();
                auto mu_v_sv_access = mu_v_sv.access();
                auto f_v_sv_access = f_v_sv.access();
                auto me_v_sv_access = me_v_sv.access();
                auto pdv_v_sv_access = pdv_v_sv.access();

                // Update grid values (P2G).
                double wm;
                for ( int i = 0; i < Basis::num_knot; ++i )
                    for ( int j = 0; j < Basis::num_knot; ++j )
                        for ( int k = 0; k < Basis::num_knot; ++k )
                        {
                            // Node weight.
                            weight = w[Dim::I][i] * w[Dim::J][j] * w[Dim::K][k];

                            // Node weight times mass
                            wm = m_p(p) * weight;

                            // Node gradient.
                            grad_weight[Dim::I] =
                                gw[Dim::I][i] * w[Dim::J][j] * w[Dim::K][k];
                            grad_weight[Dim::J] =
                                w[Dim::I][i] * gw[Dim::J][j] * w[Dim::K][k];
                            grad_weight[Dim::K] =
                                w[Dim::I][i] * w[Dim::J][j] * gw[Dim::K][k];

                            // Deposit mass
                            m_v_sv_access(stencil[Dim::I][i],
                                          stencil[Dim::J][j],
                                          stencil[Dim::K][k],
                                          0 ) += wm;

                            // Deposit energy.
                            me_v_sv_access(stencil[Dim::I][i],
                                           stencil[Dim::J][j],
                                           stencil[Dim::K][k],
                                           0 ) += wm * e_p(p);

                            for ( int d = 0; d < 3; ++d )
                            {
                                // Deposit momentum
                                mu_v_sv_access(stencil[Dim::I][i],
                                               stencil[Dim::J][j],
                                               stencil[Dim::K][k],
                                               d) +=
                                    wm * u_p(p,d);

                                // Deposit force
                                f_v_sv_access(stencil[Dim::I][i],
                                              stencil[Dim::J][j],
                                              stencil[Dim::K][k],
                                              d) +=
                                    grad_weight[d] * pv;

                                // Deposit PdV work.
                                pdv_v_sv_access(stencil[Dim::I][i],
                                                stencil[Dim::J][j],
                                                stencil[Dim::K][k],
                                                0 ) += weight * pv * div_u;

                            }
                        }
            });

        // Scatter deposited values
        Kokkos::Experimental::contribute( m_v_view, m_v_sv );
        Kokkos::Experimental::contribute( mu_v_view, mu_v_sv );
        Kokkos::Experimental::contribute( f_v_view, f_v_sv );
        Kokkos::Experimental::contribute( me_v_view, me_v_sv );
        Kokkos::Experimental::contribute( pdv_v_view, pdv_v_sv );
        scatter_halo->scatter( *scatter_fields );

        // Compute grid quantities.
        Kokkos::parallel_for(
            "update_grid.",
            Cajita::createExecutionPolicy( node_space, execution_space() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){
                if ( m_v_view(i,j,k,0) > 0.0 )
                {
                    // velocity
                    for ( int d = 0; d < 3; ++d )
                    {
                        u_v_old_view(i,j,k,d) = mu_v_view(i,j,k,d) / m_v_view(i,j,k,0);
                        u_v_theta_view(i,j,k,d) = u_v_old_view(i,j,k,d) +
                                                  theta * delta_t * f_v_view(i,j,k,d) /
                                                  m_v_view(i,j,k,0);
                        u_v_new_view(i,j,k,d) =
                            (u_v_theta_view(i,j,k,d) + (theta-1.0)*u_v_old_view(i,j,k,d)) / theta;
                    }

                    // Specific internal energy
                    e_v_old_view(i,j,k,0) = me_v_view(i,j,k,0) / m_v_view(i,j,k,0);
                    e_v_theta_view(i,j,k,0) = e_v_old_view(i,j,k,0) -
                                              theta * delta_t * pdv_v_view(i,j,k,0) /
                                              m_v_view(i,j,k,0);
                    e_v_new_view(i,j,k,0) =
                        (e_v_theta_view(i,j,k,0) + (theta-1.0)*e_v_old_view(i,j,k,0)) / theta;
                }
                else
                {
                    // velocity
                    for ( int d = 0; d < 3; ++d )
                    {
                        u_v_old_view(i,j,k,d) = 0.0;
                        u_v_theta_view(i,j,k,d) = 0.0;
                        u_v_new_view(i,j,k,d) = 0.0;
                    }

                    // specific internal energy
                    e_v_old_view(i,j,k,0) = 0.0;
                    e_v_theta_view(i,j,k,0) = 0.0;
                    e_v_new_view(i,j,k,0) = 0.0;
                }
            });

        // Compute energies.
        double total_ke = 0.0;
        Kokkos::parallel_reduce(
            "kinetic_energy",
            Kokkos::RangePolicy<execution_space>(0,local_num_p),
            KOKKOS_LAMBDA( const int p, double& ke ){
                ke += 0.5 * m_p(p) * ( u_p(p,Dim::I) * u_p(p,Dim::I) +
                                       u_p(p,Dim::J) * u_p(p,Dim::J) +
                                       u_p(p,Dim::K) * u_p(p,Dim::K) );
            },
            total_ke );
        double total_ie = 0.0;
        Kokkos::parallel_reduce(
            "internal_energy",
            Kokkos::RangePolicy<execution_space>(0,local_num_p),
            KOKKOS_LAMBDA( const int p, double& ie ){
                ie +=  m_p(p) * e_p(p);
            },
            total_ie );
        std::cout << "Total Energy: " << total_ke + total_ie << std::endl;
        std::cout << "Kinetic Energy: " << total_ke << std::endl;
        std::cout << "Internal Energy: " << total_ie << std::endl;

        // Communicate particles.
        Harlow::ParticleCommunication::redistribute(
            *local_grid, particles,
            std::integral_constant<std::size_t,Field::x>() );

        // Update time.
        time += delta_t;

        // Write results.
        Harlow::SiloParticleWriter::writeTimeStep(
            *global_grid, t+1, time, x_p,
            u_p, e_p, J_p, p_p, d_p, v_p );
        Cajita::BovWriter::writeTimeStep( t+1, time, *u_v_new );
    }
}

//---------------------------------------------------------------------------//
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );

    Kokkos::initialize( argc, argv );

    // cell size (cm)
    double cell_size = std::atof( argv[1] );

    // particles per cell in a dimension
    int ppc = std::atoi( argv[2] );

    // time integration theta.
    double theta = std::atof( argv[3] );

    // run the problem.
    nohProblem( cell_size, ppc, theta );

    Kokkos::finalize();

    MPI_Finalize();

    return 0;

}

//---------------------------------------------------------------------------//
