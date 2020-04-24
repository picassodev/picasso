#ifndef HARLOW_PARTICLELEVELSET_HPP
#define HARLOW_PARTICLELEVELSET_HPP

#include <Harlow_LevelSetRedistance.hpp>
#include <Harlow_FieldManager.hpp>
#include <Harlow_Types.hpp>

#include <Cajita.hpp>

#include <ArborX.hpp>

#include <boost/property_tree/ptree.hpp>

#include <cfloat>
#include <cmath>

//---------------------------------------------------------------------------//
namespace Harlow
{
template<class CoordinateSlice>
struct ParticleLevelSetPrimitiveData
{
    using memory_space = typename CoordinateSlice::memory_space;
    using size_type = typename CoordinateSlice::size_type;
    CoordinateSlice x;
    Kokkos::View<int*,memory_space> c;
    int num_color;
};

template<class IndexType>
struct ParticleLevelSetPredicateIndex
{
    using size_type = IndexType;
    size_type i;
    size_type j;
    size_type k;
};

template<class LocalMesh, class EntityType>
struct ParticleLevelSetPredicateData
{
    using memory_space = typename LocalMesh::memory_space;
    using size_type = int;
    using entity_type = EntityType;
    LocalMesh local_mesh;
    size_type size;
    size_type i_size;
    size_type ij_size;

    template<class LocalGrid>
    ParticleLevelSetPredicateData( const LocalMesh& lm,
                                   const LocalGrid& local_grid )
        : local_mesh(lm)
    {
        auto ghost_entities = local_grid.indexSpace(
            Cajita::Ghost(), entity_type(), Cajita::Local() );
        size = ghost_entities.size();
        i_size = ghost_entities.extent(Dim::I);
        ij_size = i_size * ghost_entities.extent(Dim::J);
    }

    KOKKOS_FUNCTION void convertIndexTo3d(
        size_type i, ParticleLevelSetPredicateIndex<size_type>& ijk ) const
    {
        ijk.k = i / ij_size;
        size_type k_size = ijk.k * ij_size;
        ijk.j = (i - k_size) / i_size;
        ijk.i = i - ijk.j * i_size - k_size;
    }
};

template<class DistanceEstimateView>
struct ParticleLevelSetCallback
{
    DistanceEstimateView distance_estimate;
    float radius;
    float dx;
    using tag = ArborX::Details::InlineCallbackTag;
    template <typename Predicate, typename OutputFunctor>
    KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                    int,
                                    float distance,
                                    OutputFunctor const &) const
    {
        // Get the entity index.
        auto ijk = getData( predicate );

        // Compute the distance from the grid entity to the particle sphere.
        auto estimate = distance - radius;

        // If a distance is negative we are going to correct it so clamp all
        // negative distances to the cell size. This will allow for the
        // projected iterations to converge quickly when the entire region in
        // the ball is negative as we will need to extend the ball anyway.
        distance_estimate( ijk.i, ijk.j, ijk.k, 0 ) =
            ( estimate > 0.0 ) ? estimate : -dx;
    }
};

} // end namespace Harlow

//---------------------------------------------------------------------------//
namespace ArborX
{
namespace Traits
{
template<class PrimitiveData>
struct Access<PrimitiveData,PrimitivesTag>
{
    using memory_space = typename PrimitiveData::memory_space;
    using size_type = typename PrimitiveData::size_type;
    static size_type size( const PrimitiveData& data )
    {
        return data.num_color;
    }
    static KOKKOS_FUNCTION Point get( const PrimitiveData& data, size_type i )
    {
        // Get the actual index of the particle.
        auto p = data.c(i);

        // Return a point made from the particles.
        return { static_cast<float>(data.x(p,0)),
                 static_cast<float>(data.x(p,1)),
                 static_cast<float>(data.x(p,2)) };
    }
};

template<class PredicateData>
struct Access<PredicateData,PredicatesTag>
{
    using entity_type = typename PredicateData::entity_type;
    using memory_space = typename PredicateData::memory_space;
    using size_type = typename PredicateData::size_type;
    static size_type size( const PredicateData& data )
    {
        return data.size;
    }
    static KOKKOS_FUNCTION auto get( const PredicateData& data, size_type i )
    {
        // Get the entity index.
        Harlow::ParticleLevelSetPredicateIndex<size_type> ijk;
        data.convertIndexTo3d( i, ijk );
        int index[3] = { ijk.i, ijk.j, ijk.k };

        // Get the coordinates of the entity.
        double x[3];
        data.local_mesh.coordinates( entity_type(), index, x );

        // Find the nearest particle to the entity. Attach the entity index to
        // use in the callback.
        return attach(
            nearest(Point{static_cast<float>(x[0]),
                          static_cast<float>(x[1]),
                          static_cast<float>(x[2])}, 1),
            ijk );
    }
};

} // end namespace Traits
} // end namespace ArborX

//---------------------------------------------------------------------------//
namespace Harlow
{
//---------------------------------------------------------------------------//
template<class ParticleListType, class SignedDistanceLocation>
class ParticleLevelSet
{
  public:

    using mesh_type = typename ParticleListType::mesh_type;
    using memory_space = typename ParticleListType::memory_space;
    using location_type = SignedDistanceLocation;
    using entity_type = typename location_type::entity_type;
    using array_type = Cajita::Array<double,
                                     entity_type,
                                     Cajita::UniformMesh<double>,
                                     memory_space>;
    using halo_type = Cajita::Halo<double,memory_space>;

    // Construct the level set with particles of a given color. If the input
    // color is negative then all particles will be used in the level set
    // regardless of color.
    ParticleLevelSet( const boost::property_tree::ptree& ptree,
                      const std::shared_ptr<ParticleListType>& particles,
                      const int color )
        : _particles( particles )
        , _color( color )
    {
        // Create array data.
        const auto& mesh = _particles->mesh();
        _distance_estimate = createArray(
            mesh, location_type(), Field::Color() );
        _signed_distance = createArray(
            mesh, location_type(), Field::SignedDistance() );
        _halo = Cajita::createHalo<double,memory_space>(
            *(_signed_distance->layout()), Cajita::FullHaloPattern() );

        // Extract parameters.
        const auto& params = ptree.get_child("particle_level_set");

        // Particles have an analytic spherical level set. Get the radius as a
        // fraction of the cell size.
        _dx = mesh.localGrid()->globalGrid().globalMesh().uniformCellSize();
        _radius = _dx * params.get<double>("particle_radius",0.5);

        // Get the Hopf-Lax redistancing parameters.
        _redistance_secant_tol =
            params.get<double>("redistance_secant_tol",0.01);
        _redistance_max_secant_iter =
            params.get<int>("redistance_max_secant_iter",50);
        _redistance_num_random_guess =
            params.get<int>("redistance_num_random_guess",10);
        _redistance_projection_tol =
            params.get<double>("redistance_projection_tol",0.05);
        _redistance_max_projection_iter =
            params.get<int>("redistance_max_projection_iter",200);
    }

    // Update the set of particle indices for the color we are building the
    // set for. This operation is needed any time the particle population is
    // updated (e.g. after a redistribution).
    template<class ExecutionSpace>
    void updateParticleIndices( const ExecutionSpace& exec_space )
    {
        // Get the colors.
        auto c_p = _particles->slice( Field::Color() );

        // Initialize color indices.
        _color_indices = Kokkos::View<int*,memory_space>(
            Kokkos::ViewAllocateWithoutInitializing("color_indices"),
            c_p.size() );

        // If the color is not negative then count how many particles have the
        // color we want and compute the indices.
        if ( _color >= 0 )
        {
            _color_count = 0;
            Kokkos::parallel_for(
                "count_color",
                Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, c_p.size() ),
                KOKKOS_LAMBDA( const int p ){
                    if ( _color == c_p(p) )
                    {
                        _color_indices(
                            Kokkos::atomic_fetch_add(&_color_count,1) ) = p;
                    }
                });
        }
        // Otherwise a negative color means all particles are included in the
        // level set so we can more efficiently generate the array.
        else
        {
            _color_count = c_p.size();
            Kokkos::parallel_for(
                "fill_color_indices",
                Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, c_p.size() ),
                KOKKOS_LAMBDA( const int p ){
                    _color_indices(p) = p;
                });
        }
    }

    // Compute the signed distance function from the current particle
    // locations.
    template<class ExecutionSpace>
    void updateSignedDistance( const ExecutionSpace& exec_space )
    {
        using device_type = Kokkos::Device<ExecutionSpace,memory_space>;

        // Build a tree from the particles of the given color.
        auto x_p = _particles->slice( Field::LogicalPosition() );
        ParticleLevelSetPrimitiveData<decltype(x_p)> primitive_data;
        primitive_data.x = x_p;
        primitive_data.c = _color_indices;
        primitive_data.num_color = _color_count;
        ArborX::BVH<device_type> bvh( primitive_data );

        // Make the search predicates.
        auto local_mesh = Cajita::createLocalMesh<memory_space>(
            *(_particles->mesh().localGrid()) );
        ParticleLevelSetPredicateData<decltype(local_mesh),entity_type>
            predicate_data( local_mesh, *(_particles->mesh().localGrid()) );

        // Make the distance callback.
        auto estimate_view = _distance_estimate->view();
        ParticleLevelSetCallback<decltype(estimate_view)> distance_callback;
        distance_callback.distance_estimate = estimate_view;
        distance_callback.radius = static_cast<float>(_radius);
        distance_callback.dx = static_cast<float>(_dx);

        // Query the particle tree with the mesh entities to find the closest
        // particle and compute the initial signed distance estimate. Dummy
        // arguments are needed even though we don't care about the output.
        Kokkos::View<int *, device_type> indices(
            Kokkos::view_alloc( "indices", Kokkos::WithoutInitializing ), 0 );
        Kokkos::View<int *, device_type> offset(
            Kokkos::view_alloc( "offset", Kokkos::WithoutInitializing ), 0 );
        bvh.query( predicate_data, distance_callback, indices, offset );

        // Do a reduction to get the global minimum.
//        _halo->scatter( *_distance_estimate, Cajita::ReduceMin() );

        // Gather to get updated ghost values.
        _halo->gather( *_distance_estimate );

        // The positive values are correct but the negative values are
        // wrong. Correct the negative values with redistancing.
        auto distance_view = _signed_distance->view();
        auto own_entities = _particles->mesh().localGrid()->indexSpace(
            Cajita::Own(), entity_type(), Cajita::Local() );
        Kokkos::parallel_for(
            "redistance",
            Cajita::createExecutionPolicy(own_entities,exec_space),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){
                if ( estimate_view(i,j,k,0) < 0.0 )
                {
                    int entity_index[3] = {i,j,k};
                    distance_view(i,j,k,0) =
                        LevelSet::redistanceEntity(
                            entity_type(),
                            estimate_view,
                            local_mesh,
                            entity_index,
                            _redistance_secant_tol,
                            _redistance_max_secant_iter,
                            _redistance_num_random_guess,
                            _redistance_projection_tol,
                            _redistance_max_projection_iter );
                }
                else
                {
                    distance_view(i,j,k,0) = estimate_view(i,j,k,0);
                }
            });
    }

    // Get the signed distance initial estimate.
    std::shared_ptr<array_type> getDistanceEstimate() const
    {
        return _distance_estimate;
    }

    // Get the signed distance function.
    std::shared_ptr<array_type> getSignedDistance() const
    {
        return _signed_distance;
    }

  private:

    std::shared_ptr<ParticleListType> _particles;
    int _color;
    std::shared_ptr<array_type> _distance_estimate;
    std::shared_ptr<array_type> _signed_distance;
    std::shared_ptr<halo_type> _halo;
    double _radius;
    double _dx;
    double _redistance_secant_tol;
    int _redistance_max_secant_iter;
    int _redistance_num_random_guess;
    double _redistance_projection_tol;
    int _redistance_max_projection_iter;
    Kokkos::View<int*,memory_space> _color_indices;
    int _color_count;
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_PARTICLELEVELSET_HPP
