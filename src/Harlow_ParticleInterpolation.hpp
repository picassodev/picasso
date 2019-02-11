#ifndef HARLOW_PARTICLEINTERPOLATION_HPP
#define HARLOW_PARTICLEINTERPOLATION_HPP

#include <Harlow_Splines.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <type_traits>
#include <cmath>

namespace Harlow
{
namespace ParticleGrid
{
//---------------------------------------------------------------------------//
// Particle field accessor type checker. All user-defined accessors must
// satisfy this type checker by declaring the particle_accessor type. See the
// ParticleViewAccessor below as an example.
//---------------------------------------------------------------------------//
template<class T, class Enable = void>
struct is_particle_accessor : public std::false_type {};

template<class T>
struct is_particle_accessor<
    T,typename std::enable_if<
          std::is_same<typename std::remove_cv<T>::type,
                       typename std::remove_cv<
                           typename T::particle_accessor>::type>::value
          >::type> : public std::true_type {};

//---------------------------------------------------------------------------//
// General particle view accessor.
//
// Note this also defines the interface for particle accessors.
//---------------------------------------------------------------------------//
template<class ViewType>
class ParticleViewAccessor
{
  public:

    using particle_accessor = ParticleViewAccessor<ViewType>;

    static constexpr unsigned field_rank =
        ViewType::traits::dimension::rank - 1;

    using reference_type = typename ViewType::reference_type;

    ParticleViewAccessor( const ViewType& view )
        : _view( view ) {}

    KOKKOS_INLINE_FUNCTION
    int extent( const int dim ) const
    { return _view.extent(dim+1); }

    template<int R = field_rank>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==R),reference_type>::type
    operator()( const int p ) const
    { return _view(p); }

    template<int R = field_rank>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==R),reference_type>::type
    operator()( const int p, const int d0 ) const
    { return _view(p,d0); }

    template<int R = field_rank>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==R),reference_type>::type
    operator()( const int p, const int d0, const int d1 ) const
    { return _view(p,d0,d1); }

  private:
    ViewType _view;
};

// Helper function.
template<class ViewType>
ParticleViewAccessor<ViewType>
createParticleViewAccessor( const ViewType& view )
{
    return ParticleViewAccessor<ViewType>( view );
}

//---------------------------------------------------------------------------//
// Grid field accessor type checker. All user-defined accessors must
// satisfy this type checker by diclearing the grid_accessor type. See the
// GridViewAccessor below as an example.
//---------------------------------------------------------------------------//
template<class T, class Enable = void>
struct is_grid_accessor : public std::false_type {};

template<class T>
struct is_grid_accessor<
    T,typename std::enable_if<
          std::is_same<typename std::remove_cv<T>::type,
                       typename std::remove_cv<
                           typename T::grid_accessor>::type>::value
          >::type> : public std::true_type {};

//---------------------------------------------------------------------------//
// General grid view accessor.
//
// This also declares an interface for grid accessors.
//---------------------------------------------------------------------------//
template<class ViewType>
class GridViewAccessor
{
  public:

    using grid_accessor = GridViewAccessor<ViewType>;

    static constexpr unsigned field_rank =
        ViewType::traits::dimension::rank - 3;

    using reference_type = typename ViewType::reference_type;

    GridViewAccessor( const ViewType& view )
        : _view( view ) {}

    KOKKOS_INLINE_FUNCTION
    int extent( const int dim ) const
    { return _view.extent(dim+3); }

    template<int R = field_rank>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(0==R),reference_type>::type
    operator()( const int i, const int j, const int k ) const
    { return _view(i,j,k); }

    template<int R = field_rank>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(1==R),reference_type>::type
    operator()( const int i, const int j, const int k, const int d0 ) const
    { return _view(i,j,k,d0); }

    template<int R = field_rank>
    KOKKOS_FORCEINLINE_FUNCTION
    typename std::enable_if<(2==R),reference_type>::type
    operator()( const int i, const int j, const int k, const int d0, const int d1 ) const
    { return _view(i,j,k,d0,d1); }

  private:
    ViewType _view;
};

// Helper function.
template<class ViewType>
GridViewAccessor<ViewType>
createGridViewAccessor( const ViewType& view )
{
    return GridViewAccessor<ViewType>( view );
}

//---------------------------------------------------------------------------//
// Rasterize a single particle to the grid.
//---------------------------------------------------------------------------//
// rank 0 field.
template<class Real, class FieldAccessor, class ViewType>
KOKKOS_INLINE_FUNCTION
void interpolateParticleValue(
    const int pid,
    const int ns,
    const int* offsets,
    const int* pli,
    const Real* wi,
    const Real* wj,
    const Real* wk,
    const FieldAccessor& src,
    ViewType& dst,
    typename std::enable_if<
    (0 == FieldAccessor::field_rank && is_particle_accessor<FieldAccessor>::value),
    int*>::type = 0 )
{
    // Get the value this particle will contribute to the grid.
    auto value = src( pid );

    // Loop over the neighboring nodes and add the field
    // contribution.
    for ( int i = 0; i < ns; ++i )
        for ( int j = 0; j < ns; ++j )
            for ( int k = 0; k < ns; ++k )
            {
                // Add the contribution of the particle to the node.
                dst( pli[Dim::I] + offsets[i],
                     pli[Dim::J] + offsets[j],
                     pli[Dim::K] + offsets[k] ) +=
                    wi[i] * wj[j] * wk[k] * value;
            }
}

//---------------------------------------------------------------------------//
// rank 1 field.
template<class Real, class FieldAccessor, class ViewType>
KOKKOS_INLINE_FUNCTION
void interpolateParticleValue(
    const int pid,
    const int ns,
    const int* offsets,
    const int* pli,
    const Real* wi,
    const Real* wj,
    const Real* wk,
    const FieldAccessor& src,
    ViewType& dst,
    typename std::enable_if<
    (1 == FieldAccessor::field_rank && is_particle_accessor<FieldAccessor>::value),
    int*>::type = 0 )
{
    // Loop over the field dimensions.
    for ( int d0 = 0; d0 < src.extent(0); ++d0 )
    {
        // Get the value this particle will contribute to the grid.
        auto value = src( pid, d0 );

        // Loop over the neighboring nodes and add the field
        // contribution.
        for ( int i = 0; i < ns; ++i )
            for ( int j = 0; j < ns; ++j )
                for ( int k = 0; k < ns; ++k )
                {
                    // Add the contribution of the particle to the node.
                    dst( pli[Dim::I] + offsets[i],
                         pli[Dim::J] + offsets[j],
                         pli[Dim::K] + offsets[k],
                         d0 ) +=
                        wi[i] * wj[j] * wk[k] * value;
                }
    }
}

//---------------------------------------------------------------------------//
// rank 2 field.
template<class Real, class FieldAccessor, class ViewType>
KOKKOS_INLINE_FUNCTION
void interpolateParticleValue(
    const int pid,
    const int ns,
    const int* offsets,
    const int* pli,
    const Real* wi,
    const Real* wj,
    const Real* wk,
    const FieldAccessor& src,
    ViewType& dst,
    typename std::enable_if<
    (2 == FieldAccessor::field_rank && is_particle_accessor<FieldAccessor>::value),
    int*>::type = 0 )
{
    // Loop over the field dimensions.
    for ( int d0 = 0; d0 < src.extent(0); ++d0 )
        for ( int d1 = 0; d1 < src.extent(1); ++d1 )
        {
            // Get the value this particle will contribute to the grid.
            auto value = src( pid, d0, d1 );

            // Loop over the neighboring nodes and add the field
            // contribution.
            for ( int i = 0; i < ns; ++i )
                for ( int j = 0; j < ns; ++j )
                    for ( int k = 0; k < ns; ++k )
                    {
                        // Add the contribution of the particle to the node.
                        dst( pli[Dim::I] + offsets[i],
                             pli[Dim::J] + offsets[j],
                             pli[Dim::K] + offsets[k],
                             d0,
                             d1 ) +=
                            wi[i] * wj[j] * wk[k] * value;
                    }
        }
}

//---------------------------------------------------------------------------//
// Interpolate from the grid to a single particle.
//---------------------------------------------------------------------------//
// rank 0 field.
template<class Real, class FieldAccessor, class ViewType>
KOKKOS_INLINE_FUNCTION
void interpolateParticleValue(
    const int pid,
    const int ns,
    const int* offsets,
    const int* pli,
    const Real* wi,
    const Real* wj,
    const Real* wk,
    const FieldAccessor& src,
    ViewType& dst,
    typename std::enable_if<
    (0 == FieldAccessor::field_rank && is_grid_accessor<FieldAccessor>::value),
    int*>::type = 0 )
{
    // Loop over the neighboring nodes and add the field
    // contribution.
    for ( int i = 0; i < ns; ++i )
        for ( int j = 0; j < ns; ++j )
            for ( int k = 0; k < ns; ++k )
                dst( pid ) +=
                    wi[i] * wj[j] * wk[k] *
                    src( pli[Dim::I] + offsets[i],
                         pli[Dim::J] + offsets[j],
                         pli[Dim::K] + offsets[k] );
}

//---------------------------------------------------------------------------//
// rank 1 field.
template<class Real, class FieldAccessor, class ViewType>
KOKKOS_INLINE_FUNCTION
void interpolateParticleValue(
    const int pid,
    const int ns,
    const int* offsets,
    const int* pli,
    const Real* wi,
    const Real* wj,
    const Real* wk,
    const FieldAccessor& src,
    ViewType& dst,
    typename std::enable_if<
    (1 == FieldAccessor::field_rank && is_grid_accessor<FieldAccessor>::value),
    int*>::type = 0 )
{
    // Loop over the neighboring nodes and add the field
    // contribution.
    for ( int i = 0; i < ns; ++i )
        for ( int j = 0; j < ns; ++j )
            for ( int k = 0; k < ns; ++k )
            {
                // Compute the weight for this node.
                Real w = wi[i] * wj[j] * wk[k];

                // Loop over the field dimensions and add the contribution of
                // the node to the particle.
                for ( int d0 = 0; d0 < src.extent(0); ++d0 )
                    dst( pid, d0 ) +=
                        w * src( pli[Dim::I] + offsets[i],
                                 pli[Dim::J] + offsets[j],
                                 pli[Dim::K] + offsets[k],
                                 d0 );
            }
}

//---------------------------------------------------------------------------//
// rank 2 field.
template<class Real, class FieldAccessor, class ViewType>
KOKKOS_INLINE_FUNCTION
void interpolateParticleValue(
    const int pid,
    const int ns,
    const int* offsets,
    const int* pli,
    const Real* wi,
    const Real* wj,
    const Real* wk,
    const FieldAccessor& src,
    ViewType& dst,
    typename std::enable_if<
    (2 == FieldAccessor::field_rank && is_grid_accessor<FieldAccessor>::value),
    int*>::type = 0 )
{
    // Loop over the neighboring nodes and add the field
    // contribution.
    for ( int i = 0; i < ns; ++i )
        for ( int j = 0; j < ns; ++j )
            for ( int k = 0; k < ns; ++k )
            {
                // Compute the weight for this node.
                Real w = wi[i] * wj[j] * wk[k];

                // Loop over the field dimensions. Add the contribution of the
                // node to the particle.
                for ( int d0 = 0; d0 < src.extent(0); ++d0 )
                    for ( int d1 = 0; d1 < src.extent(1); ++d1 )
                        dst( pid, d0, d1 ) +=
                            w * src( pli[Dim::I] + offsets[i],
                                     pli[Dim::J] + offsets[j],
                                     pli[Dim::K] + offsets[k],
                                     d0,
                                     d1 );
            }
}

//---------------------------------------------------------------------------//
// Value interpolation.
//---------------------------------------------------------------------------//
/*!
  \brief Interpolate the value of a field from a field accessor to a
  view. This will do particle-to-grid or grid-to-particle depending on the
  accessor type.

  \param particle_position view of particle positions.

  \param low_corner The physical location of the low corner of the primal grid
  used for the interpolation. For a node field use the low corner of the grid
  including the halo. For a cell field use the center of the cell in the low
  corner of the grid including the halo.

  \param rdx The inverse of the physical distance between grid locations.

  \param src Field accessor providing the data source.

  \param dst Destination view - where the data will be interpolated to.

  \param reset_dst_to_zero If true the destination view will be reset to
  zero. If false then the interpolation results will be added to the existing
  values in dst. Default to true.

  \note This function does both particle-to-grid and grid-to-particle. If a
  particle accessor is provided as the source, a grid view must be provided as
  a destination. Analogously, if a grid accessor is provided as the source, a
  particle view must be provided as the destination.
*/
template<int SplineOrder,
         class PositionViewType,
         class FieldAccessor,
         class DataViewType>
void interpolate(
    const PositionViewType& particle_position,
    const std::vector<typename PositionViewType::value_type>& low_corner,
    const typename PositionViewType::value_type rdx,
    const FieldAccessor& src,
    DataViewType& dst,
    const bool reset_dst_to_zero = true )
{
    // Get the spatial coordinate value type.
    using Real = typename PositionViewType::value_type;

    // Get the spline.
    using Basis = Spline<SplineOrder>;

    // Reset the destination view if necessary.
    if ( reset_dst_to_zero ) Kokkos::deep_copy( dst, 0.0 );

    // Extract the low corner of the grid.
    auto low_x = low_corner[Dim::I];
    auto low_y = low_corner[Dim::J];
    auto low_z = low_corner[Dim::K];

    // Get the stencil size.
    const int ns = Basis::num_knot;

    // Create a scatter-view of the destination data.
    auto dst_sv = Kokkos::Experimental::create_scatter_view( dst );

    // Loop over particles and interpolate.
    Kokkos::parallel_for(
        "interpolate",
        Kokkos::RangePolicy<typename DataViewType::execution_space>(
            0, particle_position.extent(0) ),
        KOKKOS_LAMBDA( const int p )
        {
            // Create the interpolation stencil.
            int offsets[ns];
            Basis::stencil( offsets );

            // Compute the logical space coordinates of the particle.
            Real plx[3] =
                { Basis::mapToLogicalGrid( particle_position(p,Dim::I), rdx, low_x ),
                  Basis::mapToLogicalGrid( particle_position(p,Dim::J), rdx, low_y ),
                  Basis::mapToLogicalGrid( particle_position(p,Dim::K), rdx, low_z ) };

            // Get the logical index of the particle.
            int pli[3] = { int(plx[0]), int(plx[1]), int(plx[2]) };

            // Get the particle weights.
            Real wi[ns];
            Basis::value( plx[Dim::I], wi );
            Real wj[ns];
            Basis::value( plx[Dim::J], wj );
            Real wk[ns];
            Basis::value( plx[Dim::K], wk );

            // Access the scatter view.
            auto dst_sv_data = dst_sv.access();

            // Do the individual interpolation on the particle.
            interpolateParticleValue(
                p, ns, offsets, pli, wi, wj, wk, src, dst_sv_data );
        } );

    // Apply the contribution of the scatter view.
    Kokkos::Experimental::contribute( dst, dst_sv );
}

//---------------------------------------------------------------------------//

} // end namespace ParticleGrid
} // end namespace Harlow

#endif // end HARLOW_PARTICLEINTERPOLATION_HPP
