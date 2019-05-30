#ifndef HARLOW_PARTICLEWORKSET_HPP
#define HARLOW_PARTICLEWORKSET_HPP

#include <Cajita_GridBlock.hpp>

#include <Harlow_Splines.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

namespace Harlow
{
//---------------------------------------------------------------------------//
/*!
  \class ParticleWorkset
  \brief Particle work set.

  This class defines the workset over particles needed for particle/grid
  operations for equation evaluations.

  Each particle is centered in a grid stencil defined by the B-spline used for
  interpolation. The workset contains the particle position along with the
  nodes in the stencil and their distances from the particles, basis weights,
  and basis gradients.

  A workset should be regenerated every time particles are redistributed and
  updated every time a new time step is started.
*/
//---------------------------------------------------------------------------//
template<class DeviceType>
struct ParticleWorkset
{
    // Types.
    using device_type = DeviceType;
    using execution_space = typename device_type::execution_space;
    using memory_space = typename device_type::memory_space;

    // Number of particles.
    std::size_t _num_particle;

    // Number of basis values in each dimension.
    int _ns;

    // Particle logical position. (particle,dim)
    Kokkos::View<double*[3],device_type> _logical;

    // Particle adjacent node index. (particle,ns,dim)
    Kokkos::View<int**[3],device_type> _nodes;

    // Particle distance from nodes in stencil. (particle,ns,dim)
    Kokkos::View<double**[3],device_type> _distance;

    // Particle basis at nodes in stencil. (particle,ns,dim)
    Kokkos::View<double**[3],device_type> _basis;

    // Particle basis gradient values at nodes in stencil
    // (particle,ni,nj,nk,dim)
    Kokkos::View<double****[3],device_type> _basis_grad;

    // Grid cell size.
    double _dx;

    // Inverse grid cell size.
    double _rdx;

    // Low corner of the local grid.
    double _low_x;
    double _low_y;
    double _low_z;

    // Time step size.
    double _dt;
};

//---------------------------------------------------------------------------//
// Update the workset without reallocating. An update is required after
// the particles have moved.
template<int SplineOrder, class PositionSlice, class Workset>
void updateParticleWorkset( const PositionSlice& position,
                            Workset& workset,
                            const double dt = -1.0 )
{
    // Device parameters.
    using execution_space = typename Workset::execution_space;

    // Basis parameters.
    using Basis = Spline<SplineOrder>;
    static constexpr int ns = Basis::num_knot;

    // Update the timestep if needed.
    if ( dt > 0.0 ) workset._dt = dt;

    // Update particle quantities.
    Kokkos::parallel_for(
        "updateParticleWorkset",
        Kokkos::RangePolicy<execution_space>(0,workset._num_particle),
        KOKKOS_LAMBDA( const int p )
        {
            // map logical coordinates
            workset._logical(p,Dim::I) =
                Basis::mapToLogicalGrid(
                    position(p,Dim::I), workset._rdx, workset._low_x );
            workset._logical(p,Dim::J) =
                Basis::mapToLogicalGrid(
                    position(p,Dim::J), workset._rdx, workset._low_y );
            workset._logical(p,Dim::K) =
                Basis::mapToLogicalGrid(
                    position(p,Dim::K), workset._rdx, workset._low_z );

            // Get the logical index of the particle.
            int pli[3] = { int(workset._logical(p,Dim::I)),
                           int(workset._logical(p,Dim::J)),
                           int(workset._logical(p,Dim::K)) };

            // Create the interpolation stencil.
            int offsets[ns];
            Basis::stencil( offsets );

            // Evaluate the basis.
            double wi[ns];
            Basis::value( workset._logical(p,Dim::I), wi );
            double wj[ns];
            Basis::value( workset._logical(p,Dim::J), wj );
            double wk[ns];
            Basis::value( workset._logical(p,Dim::K), wk );

            // Cache the stencil node ids.
            for ( int n = 0; n < ns; ++n )
                for ( int d = 0; d < 3; ++d )
                    workset._nodes(p,n,d) = pli[d] + offsets[n];

            // Cache the node distance values.
            for ( int n = 0; n < ns; ++n )
            {
                workset._distance(p,n,Dim::I) =
                    (workset._low_x + workset._nodes(p,n,Dim::I) * workset._dx) -
                    position(p,Dim::I);
                workset._distance(p,n,Dim::J) =
                    (workset._low_y + workset._nodes(p,n,Dim::J) * workset._dx) -
                    position(p,Dim::J);
                workset._distance(p,n,Dim::K) =
                    (workset._low_z + workset._nodes(p,n,Dim::K) * workset._dx) -
                    position(p,Dim::K);
            }

            // Cache the basis values.
            for ( int n = 0; n < ns; ++n )
            {
                workset._basis(p,n,Dim::I) = wi[n];
                workset._basis(p,n,Dim::J) = wj[n];
                workset._basis(p,n,Dim::K) = wk[n];
            }

            // Compute nodal gradients.
            double dist[3];
            double grad[3];
            for ( int i = 0; i < ns; ++i )
                for ( int j = 0; j < ns; ++j )
                    for ( int k = 0; k < ns; ++k )
                    {
                        dist[Dim::I] = workset._distance(p,i,Dim::I);
                        dist[Dim::J] = workset._distance(p,j,Dim::J);
                        dist[Dim::K] = workset._distance(p,k,Dim::K);

                        Basis::gradient(
                            wi[i]*wj[j]*wk[k], dist, workset._rdx, grad );

                        workset._basis_grad(p,i,j,k,Dim::I) = grad[Dim::I];
                        workset._basis_grad(p,i,j,k,Dim::J) = grad[Dim::J];
                        workset._basis_grad(p,i,j,k,Dim::K) = grad[Dim::K];
                    }
        } );
}

//---------------------------------------------------------------------------//
// Allocate the workset.
template<int SplineOrder, class Workset>
void allocateParticleWorkset( const Cajita::GridBlock& block,
                              const std::size_t num_particle,
                              Workset& workset )
{
    // Device parameters.
    using device_type = typename Workset::device_type;

    // Basis parameters.
    using Basis = Spline<SplineOrder>;
    static constexpr int ns = Basis::num_knot;

    // Allocate
    workset._num_particle = num_particle;

    workset._ns = ns;

    workset._logical =
        Kokkos::View<double*[3],device_type>(
            Kokkos::ViewAllocateWithoutInitializing("workset._logical"),
            workset._num_particle );

    workset._nodes =
        Kokkos::View<int**[3],device_type>(
            Kokkos::ViewAllocateWithoutInitializing("workset._nodes"),
            workset._num_particle,ns);

    workset._distance =
        Kokkos::View<double**[3],device_type>(
            Kokkos::ViewAllocateWithoutInitializing("workset._distance"),
            workset._num_particle,ns);

    workset._basis =
        Kokkos::View<double**[3],device_type>(
            Kokkos::ViewAllocateWithoutInitializing("workset._basis"),
            workset._num_particle,ns);

    workset._basis_grad =
        Kokkos::View<double****[3],device_type>(
            Kokkos::ViewAllocateWithoutInitializing("workset._basis_grad"),
            workset._num_particle,ns,ns,ns);

    workset._dx = block.cellSize();

    workset._rdx = block.inverseCellSize();

    workset._low_x = block.lowCorner(Dim::I);

    workset._low_y = block.lowCorner(Dim::J);

    workset._low_z = block.lowCorner(Dim::K);
}

//---------------------------------------------------------------------------//
// Create the workset.
template<int SplineOrder,class DeviceType>
std::shared_ptr<ParticleWorkset<DeviceType> >
createParticleWorkset( const Cajita::GridBlock& block,
                       const std::size_t num_particle )
{
    auto workset = std::make_shared<ParticleWorkset<DeviceType> >();

    allocateParticleWorkset<SplineOrder>( block, num_particle, *workset );

    return workset;
}

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_PARTICLEWORKSET_HPP
