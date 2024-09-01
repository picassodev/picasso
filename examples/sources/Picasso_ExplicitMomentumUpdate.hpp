#include <Picasso.hpp>

#include "Picasso_BoundaryCondition.hpp"

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <memory>

namespace Picasso
{

//---------------------------------------------------------------------------//
// Compute nodal velocity from mass-weighted momentum.
//---------------------------------------------------------------------------//
struct ComputeGridVelocity
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType&, const GatherDependencies&,
                const ScatterDependencies&, const LocalDependencies& local_deps,
                const int i, const int j, const int k ) const
    {
        // Get the local dependencies.
        auto m_i =
            local_deps.get( Picasso::FieldLocation::Node(), Field::Mass() );
        auto u_i =
            local_deps.get( Picasso::FieldLocation::Node(), Field::Velocity() );
        auto old_u_i =
            local_deps.get( Picasso::FieldLocation::Node(), Field::OldU() );

        // Compute velocity.
        for ( int d = 0; d < 3; ++d )
        {
            old_u_i( i, j, k, d ) = ( m_i( i, j, k ) > 0.0 )
                                        ? old_u_i( i, j, k, d ) / m_i( i, j, k )
                                        : 0.0;
            u_i( i, j, k, d ) = old_u_i( i, j, k, d );
        }
    }
};

//---------------------------------------------------------------------------//
// Update particle stress.
//---------------------------------------------------------------------------//
template <int InterpolationOrder, class StressProperties>
struct ComputeParticlePressure
{
    double dt;

    // Stress property functions
    StressProperties properties;

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
        auto x_p = Picasso::get( particle, Field::Position() );
        auto& J_p = Picasso::get( particle, Field::DetDefGrad() );
        auto& p_p = Picasso::get( particle, Field::Pressure() );
        auto s_p = Picasso::get( particle, Field::Stress() );
        auto v_p = Picasso::get( particle, Field::Volume() );
        auto m_p = Picasso::get( particle, Field::Mass() );

        // Get the gather dependencies.
        auto u_i = gather_deps.get( Picasso::FieldLocation::Node(),
                                    Field::Velocity() );

        // update strain rate
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineGradient() );
        Picasso::Mat3<double> vel_grad;
        Picasso::G2P::gradient( spline, u_i, vel_grad );

        // J_p = Kokkos::abs( !F_p );
        J_p *= Kokkos::exp( Picasso::LinearAlgebra::trace( dt * vel_grad ) );

        p_p = properties.bulk_modulus *
              ( Kokkos::pow( J_p, -properties.gamma ) - 1.0 );

        Picasso::Mat3<double> I;
        Picasso::LinearAlgebra::identity( I );

        s_p = -p_p * I;

        printf( "particle %8.5e %8.5e %8.5e %8.5e %8.5e %8.5e %8.5e\n", J_p,
                p_p, v_p, m_p, x_p( 0 ), x_p( 1 ), x_p( 2 ) );
    }
};

//---------------------------------------------------------------------------//
// Grid momentum change due to stress
//---------------------------------------------------------------------------//
template <int InterpolationOrder>
struct ComputeGridVelocityChangeStress
{
    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get particle data.
        auto v_p = Picasso::get( particle, Field::Volume() );
        auto x_p = Picasso::get( particle, Field::Position() );
        auto s_p = Picasso::get( particle, Field::Stress() );

        // Get the scatter dependencies.
        auto delta_u_s_i = scatter_deps.get( Picasso::FieldLocation::Node(),
                                             Field::DeltaUStress() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineGradient() );

        // Compute velocity update.
        Picasso::P2G::divergence( spline, -v_p * s_p, delta_u_s_i );
    }
};

//---------------------------------------------------------------------------//
// Grid momentum change due to gravity
//---------------------------------------------------------------------------//
template <int InterpolationOrder, class StressProperties>
struct ComputeGridVelocityChangeGravity
{
    // Stress property functions
    StressProperties properties;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies,
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh, const GatherDependencies&,
                const ScatterDependencies& scatter_deps,
                const LocalDependencies&, ParticleViewType& particle ) const
    {
        // Get particle data.
        auto m_p = Picasso::get( particle, Field::Mass() );
        auto x_p = Picasso::get( particle, Field::Position() );

        // Get the scatter dependencies.
        auto delta_u_g_i = scatter_deps.get( Picasso::FieldLocation::Node(),
                                             Field::DeltaUGravity() );

        // Node interpolant.
        auto spline = Picasso::createSpline(
            Picasso::FieldLocation::Node(),
            Picasso::InterpolationOrder<InterpolationOrder>(), local_mesh, x_p,
            Picasso::SplineValue(), Picasso::SplineGradient() );

        // Compute velocity update.
        Picasso::Vec3<double> g_p = { properties.gravity[0],
                                      properties.gravity[1],
                                      properties.gravity[2] };

        Picasso::P2G::value( spline, m_p * g_p, delta_u_g_i );
    }
};

//---------------------------------------------------------------------------//
// Update nodal momentum n+1 with stress, gravity
//---------------------------------------------------------------------------//
struct UpdateGridVelocity
{
    // Explicit time step size.
    double dt;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType&, const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies&,
                const int i, const int j, const int k ) const
    {
        // Get the local dependencies.
        auto m_i =
            gather_deps.get( Picasso::FieldLocation::Node(), Field::Mass() );
        auto u_i = gather_deps.get( Picasso::FieldLocation::Node(),
                                    Field::Velocity() );
        auto delta_u_s_i = gather_deps.get( Picasso::FieldLocation::Node(),
                                            Field::DeltaUStress() );

        auto delta_u_g_i = gather_deps.get( Picasso::FieldLocation::Node(),
                                            Field::DeltaUGravity() );

        // Compute velocity.
        Picasso::Vec3<double> zeros = { 0.0, 0.0, 0.0 };
        u_i( i, j, k ) +=
            ( m_i( i, j, k ) > 0.0 )
                ? dt * ( delta_u_s_i( i, j, k ) + delta_u_g_i( i, j, k ) ) /
                      m_i( i, j, k )
                : zeros;

        printf( "%d %d %d %8.5e %8.5e %8.5e %8.5e\n", i, j, k, m_i( i, j, k ),
                delta_u_s_i( i, j, k, 2 ), delta_u_g_i( i, j, k, 2 ),
                u_i( i, j, k, 2 ) );
    }
};

template <int InterpolationOrder, class InterpolationType>
struct Grid2ParticleVelocity;

//---------------------------------------------------------------------------//
// Update particle state. APIC variant.
//---------------------------------------------------------------------------//
template <int InterpolationOrder>
struct Grid2ParticleVelocity<InterpolationOrder, APIC::APicTag>
{
    // FIXME: only needed for FLIP/PIC
    double beta;

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

        // Update particle velocity using a PolyPIC update.
        Picasso::APIC::g2p( u_i, f_p, spline );

        // Update particle position.
        auto x_i_updated =
            [=]( const int i, const int j, const int k, const int d )
        {
            printf( "g2p %d %d %d %8.5e %8.5e\n", i, j, k, x_i( i, j, k, 2 ),
                    u_i( i, j, k, 2 ) );
            return x_i( i, j, k, d ) + dt * u_i( i, j, k, d );
        };
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
              class ParticleViewType>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType& local_mesh,
                const GatherDependencies& gather_deps,
                const ScatterDependencies&, const LocalDependencies& local_deps,
                ParticleViewType& particle ) const
    {
        // Get particle data.
        auto u_p = Picasso::get( particle, Field::Velocity() );
        auto x_p = Picasso::get( particle, Field::Position() );

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

//---------------------------------------------------------------------------//
// Compute boundary condition on grid.
//---------------------------------------------------------------------------//
template <class BCType, class FieldType>
struct ApplyBoundaryCondition
{
    BCType bc;

    template <class LocalMeshType, class GatherDependencies,
              class ScatterDependencies, class LocalDependencies>
    KOKKOS_INLINE_FUNCTION void
    operator()( const LocalMeshType&, const GatherDependencies&,
                const ScatterDependencies&, const LocalDependencies& local_deps,
                const int i, const int j, const int k ) const
    {
        // Get the local dependencies.
        auto f_i =
            local_deps.get( Picasso::FieldLocation::Node(), FieldType() );

        bc.apply( f_i, i, j, k );
    }
};

template <int InterpolationOrder, class ParticleFieldType, class OldFieldType,
          class InterpolationType>
struct Particle2Grid;

//---------------------------------------------------------------------------//
// Project particle enthalpy/momentum to grid. APIC variant
//---------------------------------------------------------------------------//
template <int InterpolationOrder, class ParticleFieldType, class OldFieldType>
struct Particle2Grid<InterpolationOrder, ParticleFieldType, OldFieldType,
                     APIC::APicTag>
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

//---------------------------------------------------------------------------//
// Explicit momentum time stepper
//---------------------------------------------------------------------------//

template <class Mesh, class InterpolationType, class ParticleVelocity,
          class StressProperties>
class ExplicitMomentumIntegrator
{
  public:
    using mass_type =
        Picasso::FieldLayout<Picasso::FieldLocation::Node, Field::Mass>;
    using velocity_type =
        Picasso::FieldLayout<Picasso::FieldLocation::Node, Field::Velocity>;
    using old_u_type =
        Picasso::FieldLayout<Picasso::FieldLocation::Node, Field::OldU>;
    using delta_u_s_type =
        Picasso::FieldLayout<Picasso::FieldLocation::Node, Field::DeltaUStress>;
    using delta_u_g_type = Picasso::FieldLayout<Picasso::FieldLocation::Node,
                                                Field::DeltaUGravity>;
    using grid_type = Picasso::FieldLayout<Picasso::FieldLocation::Node,
                                           Picasso::Field::PhysicalPosition<3>>;

    // Momentum P2G operator types.
    using p2g_scatter_deps =
        Picasso::ScatterDependencies<mass_type, old_u_type>;
    using p2g_u_op = Picasso::GridOperator<Mesh, p2g_scatter_deps>;

    // Grid velocity operator types.
    using compute_u_local_deps =
        Picasso::LocalDependencies<mass_type, velocity_type, old_u_type>;
    using compute_u_op = Picasso::GridOperator<Mesh, compute_u_local_deps>;

    // Update particle stress operator types
    using compute_s_gather_deps = Picasso::GatherDependencies<velocity_type>;
    using compute_s_op = Picasso::GridOperator<Mesh, compute_s_gather_deps>;

    // Grid boundary condition operator types.
    using bc_u_local_deps =
        Picasso::LocalDependencies<velocity_type, delta_u_s_type,
                                   delta_u_g_type, old_u_type, mass_type>;
    using bc_u_op = Picasso::GridOperator<Mesh, bc_u_local_deps>;

    // Compute velocity change due to stress operator types.
    using compute_du_s_scatter_deps =
        Picasso::ScatterDependencies<delta_u_s_type>;
    using compute_du_s_op =
        Picasso::GridOperator<Mesh, compute_du_s_scatter_deps>;

    // Compute velocity change due to gravity operator types.
    using compute_du_g_scatter_deps =
        Picasso::ScatterDependencies<delta_u_g_type>;
    using compute_du_g_op =
        Picasso::GridOperator<Mesh, compute_du_g_scatter_deps>;

    // Grid velocity time update operator types.
    using update_u_gather_deps =
        Picasso::GatherDependencies<mass_type, velocity_type, delta_u_s_type,
                                    delta_u_g_type>;
    using update_u_op = Picasso::GridOperator<Mesh, update_u_gather_deps>;

    // Particle update G2P types.
    using g2p_u_gather_deps =
        Picasso::GatherDependencies<mass_type, velocity_type, old_u_type>;
    using g2p_u_local_deps = Picasso::LocalDependencies<grid_type>;
    using g2p_u_op =
        Picasso::GridOperator<Mesh, g2p_u_gather_deps, g2p_u_local_deps>;

    using property_type = StressProperties;
    using interpolation_type = InterpolationType;
    using interpolation_variable = ParticleVelocity;

  public:
    // Constructor.
    ExplicitMomentumIntegrator( const std::shared_ptr<Mesh>& mesh,
                                StressProperties properties,
                                const double max_dt, const double cfl_number,
                                const double beta, const bool fixed_dt )
        : _max_dt( max_dt )
        , _cfl_number( cfl_number )
        , _beta( beta )
        , _fixed_dt( fixed_dt )
        , _p2g_momentum( mesh )
        , _compute_velocity( mesh )
        , _compute_stress( mesh )
        , _compute_delta_u_s( mesh )
        , _compute_delta_u_g( mesh )
        , _apply_bc_momentum( mesh )
        , _update_velocity( mesh )
        , _g2p_momentum( mesh )
        , _properties( properties )
    {
        _total_steps = 0;
        _total_time = 0.0;
        _dt = _max_dt;
        _cell_size = mesh->cellSize();
    }

    // Populate the field manager.
    void setup( Picasso::FieldManager<Mesh>& fm )
    {
        _p2g_momentum.setup( fm );
        _compute_velocity.setup( fm );
        _compute_stress.setup( fm );
        _compute_delta_u_s.setup( fm );
        _compute_delta_u_g.setup( fm );
        _apply_bc_momentum.setup( fm );
        _update_velocity.setup( fm );
        _g2p_momentum.setup( fm );
    }

    // Do a time step.
    // Note that the boundary condition is passed here rather than saved as a
    // member so that it can be initialized with the FieldManager.
    template <class ExecutionSpace, class ParticleList, class LocalGridType,
              class BCType>
    void step( const ExecutionSpace& exec_space,
               const Picasso::FieldManager<Mesh>& fm, ParticleList& particles,
               const LocalGridType& local_grid, const BCType bc )
    {
        // Spline interpolation order.
        const int spline_order = 1;

        // P2G
        Particle2Grid<spline_order, interpolation_variable, Field::OldU,
                      interpolation_type>
            p2g_func{ _dt };
        _p2g_momentum.apply( "Picasso::p2g_U",
                             Picasso::FieldLocation::Particle(), exec_space, fm,
                             particles, p2g_func );

        // Compute grid velocity.
        ComputeGridVelocity compute_u_func;
        _compute_velocity.apply( "Picasso::grid_U",
                                 Picasso::FieldLocation::Node(), exec_space, fm,
                                 compute_u_func );

        // Update particle stress.
        ComputeParticlePressure<spline_order, property_type> compute_s_func{
            _dt, _properties };
        _compute_stress.apply( "Picasso::particle_S",
                               Picasso::FieldLocation::Particle(), exec_space,
                               fm, particles, compute_s_func );

        // Compute grid velocity change due to stress
        ComputeGridVelocityChangeStress<spline_order> compute_du_s_func;
        _compute_delta_u_s.apply(
            "Picasso::div_S", Picasso::FieldLocation::Particle(), exec_space,
            fm, particles, compute_du_s_func );

        // Compute grid velocity change due to gravity
        ComputeGridVelocityChangeGravity<spline_order, property_type>
            compute_du_g_func{ _properties };
        _compute_delta_u_g.apply(
            "Picasso::rhoG", Picasso::FieldLocation::Particle(), exec_space, fm,
            particles, compute_du_g_func );

        // Compute next grid velocity.
        UpdateGridVelocity update_u_func{ _dt };
        _update_velocity.apply( "Picasso::update_U",
                                Picasso::FieldLocation::Node(), exec_space, fm,
                                update_u_func );

        // Apply boundary condition.
        ApplyBoundaryCondition<BCType, Field::Velocity> apply_bc{ bc };
        _apply_bc_momentum.apply( "Picasso::BC_U",
                                  Picasso::FieldLocation::Node(), exec_space,
                                  fm, apply_bc );

        // G2P
        Grid2ParticleVelocity<spline_order, interpolation_type> g2p_func{ _beta,
                                                                          _dt };
        _g2p_momentum.apply( "Picasso::g2p_U",
                             Picasso::FieldLocation::Particle(), exec_space, fm,
                             particles, g2p_func );

        // Do not force particles redistribution
        particles.redistribute( local_grid, Field::Position() );

        _total_time += _dt;
        _total_steps += 1;
    }

    double dt() { return _dt; }
    double totalTime() { return _total_time; }
    int totalSteps() { return _total_steps; }

  protected:
    double _max_dt;
    double _cfl_number;
    double _beta;
    bool _fixed_dt;
    int _total_steps;
    double _total_time;
    double _dt;
    double _cell_size;

    p2g_u_op _p2g_momentum;
    compute_u_op _compute_velocity;
    compute_s_op _compute_stress;
    compute_du_s_op _compute_delta_u_s;
    compute_du_g_op _compute_delta_u_g;
    bc_u_op _apply_bc_momentum;
    update_u_op _update_velocity;
    g2p_u_op _g2p_momentum;

    property_type _properties;
};

template <class Mesh, class InterpolationType, class ParticleVelocity,
          class StressProperties>
auto createExplicitMomentumIntegrator(
    const std::shared_ptr<Mesh>& mesh, InterpolationType, ParticleVelocity,
    StressProperties properties, const double max_dt, const double cfl = 0.5,
    const double beta = 1.0, const bool fixed_dt = false )
{
    return ExplicitMomentumIntegrator<Mesh, InterpolationType, ParticleVelocity,
                                      StressProperties>(
        mesh, properties, max_dt, cfl, beta, fixed_dt );
}
//---------------------------------------------------------------------------//

} // namespace Picasso
