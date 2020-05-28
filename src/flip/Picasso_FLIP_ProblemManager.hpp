#ifndef PICASSO_FLIP_PROBLEMMANAGER_HPP
#define PICASSO_FLIP_PROBLEMMANAGER_HPP

#include <Picasso_FLIP_AuxiliaryFields.hpp>
#include <Picasso_FLIP_EquationOfState.hpp>
#include <Picasso_FLIP_BoundaryCondition.hpp>

#include <Picasso_UniformMesh.hpp>
#include <Picasso_ParticleList.hpp>
#include <Picasso_FieldManager.hpp>
#include <Picasso_FacetGeometry.hpp>
#include <Picasso_ParticleInit.hpp>
#include <Picasso_SiloParticleWriter.hpp>

#include <Kokkos_Core.hpp>

#include <boost/property_tree/ptree.hpp>

#include <memory>

namespace Picasso
{
namespace FLIP
{
//---------------------------------------------------------------------------//
template<class MemorySpace>
class ProblemManager
{
  public:

    // Memory space.
    using memory_space = MemorySpace;

    // Mesh type.
    using mesh_type = UniformMesh<memory_space>;

    // Particle list.
    using particle_list = ParticleList<mesh_type,
                                       Field::LogicalPosition,
                                       Field::Velocity,
                                       Field::Mass,
                                       Field::InternalEnergy>;

    // Particle type.
    using particle_type = typename particle_list::particle_type;

    // Field manager.
    using field_manager = FieldManager<mesh_type>;

  public:

    // Constructor.
    template<class ExecutionSpace>
    ProblemManager( const ExecutionSpace& exec_space,
                    const boost::property_tree::ptree& ptree,
                    MPI_Comm comm )
    {
        // Get problem parameters.
        const auto& flip_params = ptree.get_child("flip");
        _halo_min =
            std::max(flip_params.get<int>("minimum_halo_cell_width"),3);
        auto ppc = flip_params.get<int>("particle_per_cell_dim");
        _delta_t = flip_params.get<double>("time_step_size");
        _theta = flip_params.get<double>("time_integration_theta");

        // Create material model.
        double gamma = flip_params.get<double>("gamma");
        _eos = IdealGas( gamma );

        // Load the geometry.
        FacetGeometry<MemorySpace> geom( ptree, exec_space );
        const auto& geom_data = geom.data();

        // Create the mesh.
        _mesh = std::make_shared<mesh_type>( ptree,
                                             geom.globalBoundingBox(),
                                             _halo_min,
                                             comm );

        // Create boundary condition.
        _bc = DomainNoSlipBoundary( *_mesh );

        // Create particles.
        _particles = std::make_shared<particle_list>( "flip_particles", _mesh );


        // Create the field manager.
        _fields = std::make_shared<field_manager>( _mesh );

        // Cell fields.
        _fields->add( FieldLocation::Cell(), Field::Density() );
        _fields->add( FieldLocation::Cell(), Field::Pressure() );
        _fields->add( FieldLocation::Cell(), Field::InternalEnergy() );
        _fields->add( FieldLocation::Cell(), VelocityThetaDivergence() );
        _fields->add( FieldLocation::Cell(), CompressionTerm() );

        // Node fields.
        _fields->add( FieldLocation::Node(), Field::Mass() );
        _fields->add( FieldLocation::Node(), Field::Velocity() );
        _fields->add( FieldLocation::Node(), VelocityOld() );
        _fields->add( FieldLocation::Node(), VelocityTheta() );
        _fields->add( FieldLocation::Node(), AccelerationTheta() );
        _fields->add( FieldLocation::Node(), AccelerationSquared() );

        // Particle initialization function.
        auto init_func =
            KOKKOS_LAMBDA( const double x_ref[3],
                           const double volume,
                           particle_type& p )
            {
                // Put points in single precision.
                float xf[3] = {float(x_ref[0]),float(x_ref[1]),float(x_ref[2])};

                // Locate the point in the geometry.
                // FIXME: This only works for uniform grids. We get logical
                // particle coordinates through this function so this location
                // will only work for uniform grids where these are also the
                // physical coordinates. So either we need to map the particle
                // coordinate back to the physical frame (and provide both in
                // the interface to the particle init functions, which is
                // probably the best idea) or do something like generate the
                // geometry as a level set although we would need a level set
                // then for each geometric object to do the setup right.
                auto volume_id = FacetGeometryOps::locatePoint(xf,geom_data);

                // Hard code for shock-tube for now.

                // Left and right.
                if ( 1 == volume_id || 2 == volume_id )
                {
                    // Assign position.
                    for ( int d = 0; d < 3; ++d )
                        ParticleAccess::get( p, Field::LogicalPosition(), d ) =
                            x_ref[d];

                    // Assign velocity.
                    for ( int d = 0; d < 3; ++d )
                        ParticleAccess::get( p, Field::Velocity(), d ) =
                            0.0;

                    // Left side.
                    if ( 1 == volume_id )
                    {
                        // Initial conditions.
                        double pressure_left = 1.0;
                        double density_left = 1.0;

                        // Assign mass.
                        double mass_left = density_left * volume;
                        ParticleAccess::get( p, Field::Mass() ) = mass_left;

                        // Assign internal energy.
                        ParticleAccess::get( p, Field::InternalEnergy() ) =
                            _eos( Field::InternalEnergy(),
                                  pressure_left, density_left ) * mass_left;
                    }

                    // Right side.
                    else if ( 2 == volume_id )
                    {
                        // Initial conditions.
                        double pressure_right = 0.1;
                        double density_right = 0.125;

                        // Assign mass.
                        double mass_right = density_right * volume;
                        ParticleAccess::get( p, Field::Mass() ) = mass_right;

                        // Assign internal energy.
                        ParticleAccess::get( p, Field::InternalEnergy() ) =
                            _eos( Field::InternalEnergy(),
                                  pressure_right, density_right ) * mass_right;
                    }

                    // Particle was created.
                    return true;
                }

                // Outside domain case.
                else
                {
                    return false;
                }
            };

        // Initialize particle data.
        initializeParticles(
            InitUniform(), exec_space, ppc, init_func, *_particles );
    }

    // Get the mesh.
    std::shared_ptr<mesh_type> mesh() const { return _mesh; }

    // Get the particles.
    std::shared_ptr<particle_list> particleList() const { return _particles; }

    // Get the grid field manager.
    std::shared_ptr<field_manager> fields() const
    { return _fields; }

    // Communicate particles.
    template<class ExecutionSpace>
    void communicateParticles( const ExecutionSpace& )
    {
        _particles->redistribute();
    }

    // Write particle fields.
    void writeParticleFields( const int step, const double time ) const
    {
        SiloParticleWriter::writeTimeStep(
            _mesh->localGrid()->globalGrid(),
            step,
            time,
            _particles->slice(Field::LogicalPosition()),
            _particles->slice(Field::InternalEnergy()),
            _particles->slice(Field::Mass()),
            _particles->slice(Field::Velocity()) );
    }

    // Time step size.
    double timeStepSize() const { return _delta_t; }

    // Time integration parameter.
    double theta() const { return _theta; }

    // Equation of state.
    IdealGas eos() const
    { return _eos; }

    // Boundary condition.
    DomainNoSlipBoundary boundaryCondition() const
    { return _bc; }

  private:

    // Mesh.
    std::shared_ptr<mesh_type> _mesh;

    // Particles.
    std::shared_ptr<particle_list> _particles;

    // Grid fields.
    std::shared_ptr<field_manager> _fields;

    // Minimum halo width.
    int _halo_min;

    // Time step size.
    double _delta_t;

    // Time integration theta.
    double _theta;

    // Equation of state.
    IdealGas _eos;

    // Boundary condition.
    DomainNoSlipBoundary _bc;
};

//---------------------------------------------------------------------------//

} // end namespace FLIP
} // end namespace Picasso

#endif // end PICASSO_FLIP_PROBLEMMANAGER_HPP
