#ifndef PICASSO_LPBF_PROBLEMMANAGER_HPP
#define PICASSO_LPBF_PROBLEMMANAGER_HPP

#include <Picasso_LPBF_LaserSource.hpp>
#include <Picasso_LPBF_AuxiliaryFieldTypes.hpp>

#include <Picasso_ParticleList.hpp>
#include <Picasso_UniformMesh.hpp>
#include <Picasso_FieldManager.hpp>
#include <Picasso_FieldTypes.hpp>
#include <Picasso_ParticleLevelSet.hpp>
#include <Picasso_FacetGeometry.hpp>
#include <Picasso_ParticleInit.hpp>
#include <Picasso_SiloParticleWriter.hpp>

#include <Kokkos_Core.hpp>

#include <boost/property_tree/ptree.hpp>

#include <memory>

namespace Picasso
{
namespace LPBF
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
                                       Field::Mass,
                                       Field::Volume,
                                       Field::InternalEnergy,
                                       Field::Color>;

    // Particle type
    using particle_type = typename particle_list::particle_type;

    // Field manager.
    using field_manager = FieldManager<mesh_type>;

    // Level set.
    using level_set = ParticleLevelSet<particle_list,FieldLocation::Node>;

  public:

    // Constructor.
    template<class ExecutionSpace>
    ProblemManager( const ExecutionSpace& exec_space,
                    const boost::property_tree::ptree& ptree,
                    MPI_Comm comm )
    {
        // Get the problem parameters.
        const auto& lpbf_params = ptree.get_child("lpbf");
        _halo_min =
            std::max(lpbf_params.get<int>("minimum_halo_cell_width"),3);
        auto ppc = lpbf_params.get<int>("particle_per_cell_dim");
        _delta_t = lpbf_params.get<double>("time_step_size");

        // Initial conditions.
        auto init_temp = lpbf_params.get<double>("initial_temperature");

        // Get the material parameters.
        const auto& mat_params = ptree.get_child("material");
        _density = mat_params.get<double>("density");
        _specific_heat_capacity =
            mat_params.get<double>("specific_heat_capacity");
        _thermal_conductivity =
            mat_params.get<double>("thermal_conductivity");

        // Load the geometry.
        FacetGeometry<MemorySpace> geom( ptree, exec_space );
        const auto& geom_data = geom.data();

        // Create the mesh.
        _mesh = std::make_shared<mesh_type>( ptree,
                                             geom.globalBoundingBox(),
                                             _halo_min,
                                             comm );

        // Create particles.
        _particles = std::make_shared<particle_list>( "lpbf_particles", _mesh );

        // Create the field manager.
        _state_manager = std::make_shared<field_manager>( _mesh );
        _state_manager->add( FieldLocation::Node(), Field::Mass() );
        _state_manager->add( FieldLocation::Node(), Field::InternalEnergy() );

        // Create auxiliary fields for implementation.
        _aux_manager = std::make_shared<field_manager>( _mesh );
        _aux_manager->add( FieldLocation::Node(),
                           UpdatedInternalEnergy() );
        _aux_manager->add( FieldLocation::Node(),
                           Field::SignedDistance() );

        // Create particle level set. We will create the level set over all
        // particles.
        int level_set_color = -1;
        _level_set =
            std::make_shared<level_set>( ptree, _particles, level_set_color );

        // Make the laser source term.
        _laser_source.setup( ptree );

        // Initialize particles.
        createParticles( exec_space, ppc, init_temp, geom_data );
    }

    // Particle initialization.
    template<class ExecutionSpace, class GeomData>
    void createParticles( const ExecutionSpace& exec_space,
                          const int ppc,
                          const double init_temp,
                          const GeomData& geom_data )
    {
        // Particle initialization function.
        auto init_func =
            KOKKOS_LAMBDA( const double x[3],
                           const double volume,
                           particle_type& p )
            {
                // Put points in single precision.
                float xf[3] = {float(x[0]),float(x[1]),float(x[2])};

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

                // If the particle ends up in any volume other than the
                // implicit complement create a new particle.
                if ( volume_id >= 0 )
                {
                    // Assign position.
                    for ( int d = 0; d < 3; ++d )
                        Access::get( p, Field::LogicalPosition(), d ) =
                            x[d];

                    // Assign mass.
                    Access::get( p, Field::Mass() ) = _density * volume;

                    // Assign volume.
                    Access::get( p, Field::Volume() ) = volume;

                    // Start with an internal energy at the initial
                    // temperature using a simple equation of state.
                    Access::get( p, Field::InternalEnergy() ) =
                        _specific_heat_capacity * init_temp;

                    // Color
                    Access::get( p, Field::Color() ) = 0;

                    // Return true for created particle.
                    return true;
                }

                // Otherwise we don't create this particle.
                else
                {
                    return false;
                }
            };

        // Initialize particle data.
        initializeParticles(
            InitUniform(), exec_space, ppc, init_func, *_particles );

        // Compute initial level set color indices.
        _level_set->updateParticleIndices( exec_space );
    }

    // Get the mesh.
    std::shared_ptr<mesh_type> mesh() const { return _mesh; }

    // Get the particles.
    std::shared_ptr<particle_list> particleList() const { return _particles; }

    // Get the primary state manager.
    std::shared_ptr<field_manager> state() const
    { return _state_manager; }

    // Get the auxiliary field manager. (Time step implementation details).
    std::shared_ptr<field_manager> auxiliaryFields() const
    { return _aux_manager; }

    // Get the free surface level set.
    std::shared_ptr<level_set> levelSet() const { return _level_set; }

    // Get the laser source term.
    const LaserSource& laserSource() const { return _laser_source; }

    // Communicate particles.
    template<class ExecutionSpace>
    void communicateParticles( const ExecutionSpace& exec_space )
    {
        auto did_comm = _particles->redistribute();
        if ( did_comm )
        {
            _level_set->updateParticleIndices( exec_space );
        }
    }

    // Write particle fields.
    void writeParticleFields( const int step, const double time ) const
    {
        SiloParticleWriter::writeTimeStep(
            _mesh->localGrid()->globalGrid(),
            step,
            time,
            _particles->slice(Field::LogicalPosition()),
            _particles->slice(Field::InternalEnergy()) );
    }

    // Time step size.
    double timeStepSize() const { return _delta_t; }

    // Material parameters.
    double density() const { return _density; }
    double thermalConductivity() const { return _thermal_conductivity; }
    double specificHeatCapacity() const { return _specific_heat_capacity; }

  private:

    // Mesh
    std::shared_ptr<mesh_type> _mesh;

    // Particles.
    std::shared_ptr<particle_list> _particles;

    // Primary state variables.
    std::shared_ptr<field_manager> _state_manager;

    // Auxiliary field variables.
    std::shared_ptr<field_manager> _aux_manager;

    // Particle level set.
    std::shared_ptr<level_set> _level_set;

    // Minimum halo width.
    int _halo_min;

    // Laser source term.
    LaserSource _laser_source;

    // Time step size.
    double _delta_t;

    // Material density.
    double _density;

    // Material specific heat capactiy.
    double _specific_heat_capacity;

    // Material thermal conductivity.
    double _thermal_conductivity;
};

//---------------------------------------------------------------------------//

} // end namespace LPBF
} // end namespace Picasso

#endif // end PICASSO_LPBF_PROBLEMMANAGER_HPP
