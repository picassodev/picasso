#ifndef HARLOW_SPLINES_HPP
#define HARLOW_SPLINES_HPP

#include <Harlow_Types.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>

namespace Harlow
{
//---------------------------------------------------------------------------//
// B-Spline interface.
//---------------------------------------------------------------------------//
template<int Order>
struct Spline;

//---------------------------------------------------------------------------//
// Linear. Defined on the primal grid.
template<>
struct Spline<FunctionOrder::Linear>
{
    // The number of non-zero knots in the spline.
    static constexpr int num_knot = 2;

    /*!
      \brief Map a particle to the logical space of the primal grid in a
      single dimension.
      \param xp The coordinate to map to the logical space.
      \param rdx The inverse of the physical distance between grid locations.
      \param low_x The physical location of the low corner of the primal
      grid.
      \return The coordinate in the logical primal grid space.

      \note Casting this result to an integer yields the cell id.
      \note A linear spline uses the primal grid.
    */
    template<class Real>
    KOKKOS_INLINE_FUNCTION
    static Real
    mapToLogicalGrid( const Real xp, const Real rdx, const Real low_x )
    {
        return (xp - low_x) * rdx;
    }

    /*
      \brief Get the logical space stencil of the spline. The stencil defines
      the offsets into a grid field about a logical coordinate.
      \param stencil The stencil offsets.
    */
    KOKKOS_INLINE_FUNCTION
    static void stencil( int offsets[2] )
    {
        offsets[0] = 0;
        offsets[1] = 1;
    }

    /*!
      \brief Calculate the value of the spline at all knots.
      \param x0 The coordinate at which to evaluate the spline in the logical
      grid space.
      \param values Basis values at the knots. Ordered from lowest to highest
      in terms of knot location.
    */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static void value( const Real x0, Real values[2] )
    {
        // Knot at i
        Real xn = x0 - int(x0);
        values[0] = 1.0 - xn;

        // Knot at i + 1
        xn -= 1.0;
        values[1] = 1.0 + xn;
    }

    /*!
      \brief Calculate the value of the gradient of the spline at given ijk
      index in the stencil.
      \param weight The node interpolation weight.
      \param distance The physical distance between the particle and the
      node.
      \param rdx The inverse physical grid cell size.
      \param gradient Basis gradient at the given node.
    */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static void gradient( const Real /* weight */,
                          const Real* distance,
                          const Real rdx,
                          Real gradient[3] )
    {
        gradient[Dim::I] = ( distance[Dim::I] > 0.0 ) ? rdx : -rdx;
        gradient[Dim::J] = ( distance[Dim::J] > 0.0 ) ? rdx : -rdx;
        gradient[Dim::K] = ( distance[Dim::K] > 0.0 ) ? rdx : -rdx;
    }
};

//---------------------------------------------------------------------------//
// Quadratic. Defined on the dual grid.
template<>
struct Spline<FunctionOrder::Quadratic>
{
    // The number of non-zero knots in the spline.
    static constexpr int num_knot = 3;

    /*!
      \brief Map a particle to the logical space of the dual grid in a
      single dimension.
      \param xp The coordinate to map to the logical space.
      \param rdx The inverse of the physical distance between grid locations.
      \param low_x The physical location of the low corner of the dual grid.
      \return The coordinate in the logical dual grid space.

      \note Casting this result to an integer yields the cell id.
      \note A quadratic spline uses the dual grid.
    */
    template<class Real>
    KOKKOS_INLINE_FUNCTION
    static Real
    mapToLogicalGrid( const Real xp, const Real rdx, const Real low_x )
    {
        return (xp - low_x) * rdx + 0.5;
    }

    /*
      \brief Get the logical space stencil of the spline. The stencil defines
      the offsets into a grid field about a logical coordinate.
      \param stencil The stencil offsets.
    */
    KOKKOS_INLINE_FUNCTION
    static void stencil( int offsets[3] )
    {
        offsets[0] = -1;
        offsets[1] = 0;
        offsets[2] = 1;
    }

    /*!
       \brief Calculate the value of the spline at all knots.
       \param x0 The coordinate at which to evaluate the spline in the logical
       grid space.
       \param values Basis values at the knots. Ordered from lowest to highest
       in terms of knot location.
    */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static void value( const Real x0, Real values[3] )
    {
        // Knot at i - 1
        Real xn = x0 - int(x0) + 0.5;
        values[0] = 0.5 * xn * xn - 1.5 * xn + 9.0 / 8.0;

        // Knot at i
        xn -= 1.0;
        values[1] = -xn * xn + 0.75;

        // Knot at i + 1
        xn -= 1.0;
        values[2] = 0.5 * xn * xn + 1.5 * xn + 9.0 / 8.0;
    }

    /*!
      \brief Calculate the value of the gradient of the spline at given ijk
      index in the stencil.
      \param weight The node interpolation weight.
      \param distance The physical distance between the particle and the
      node.
      \param rdx The inverse physical grid cell size.
      \param gradient Basis gradient at the given node.
    */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static void gradient( const Real weight,
                          const Real* distance,
                          const Real rdx,
                          Real gradient[3] )
    {
        Real w_dp = weight * 4.0 * rdx * rdx;
        gradient[Dim::I] = w_dp * distance[Dim::I];
        gradient[Dim::J] = w_dp * distance[Dim::J];
        gradient[Dim::K] = w_dp * distance[Dim::K];
    }
};

//---------------------------------------------------------------------------//
// Cubic. Defined on the primal grid.
template<>
struct Spline<FunctionOrder::Cubic>
{
    // The number of non-zero knots in the spline.
    static constexpr int num_knot = 4;

    /*!
      \brief Map a particle to the logical space of the primal grid in a
      single dimension.
      \param xp The coordinate to map to the logical space.
      \param rdx The inverse of the physical distance between grid locations.
      \param low_x The physical location of the low corner of the primal
      grid.
      \return The coordinate in the logical primal grid space.

      \note Casting this result to an integer yields the cell id.
      \note A cubic spline uses the primal grid.
    */
    template<class Real>
    KOKKOS_INLINE_FUNCTION
    static Real
    mapToLogicalGrid( const Real xp, const Real rdx, const Real low_x )
    {
        return (xp - low_x) * rdx;
    }

    /*
      \brief Get the logical space stencil of the spline. The stencil defines
      the offsets into a grid field about a logical coordinate.
      \param stencil The stencil offsets.
    */
    KOKKOS_INLINE_FUNCTION
    static void stencil( int offsets[4] )
    {
        offsets[0] = -1;
        offsets[1] = 0;
        offsets[2] = 1;
        offsets[3] = 2;
    }

    /*!
       \brief Calculate the value of the spline at all knots.
       \param x0 The coordinate at which to evaluate the spline in the logical
       grid space.
       \param values Basis values at the knots. Ordered from lowest to highest
       in terms of knot location.
    */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static void value( const Real x0, Real values[4] )
    {
        // Knot at i - 1
        Real xn = x0 - int(x0) + 1.0;
        Real xn2 = xn * xn;
        values[0] = -xn * xn2 / 6.0 + xn2 - 2.0 * xn + 4.0 / 3.0;

        // Knot at i
        xn -= 1.0;
        xn2 = xn * xn;
        values[1] = 0.5 * xn * xn2 - xn2 + 2.0 / 3.0;

        // Knot at i + 1
        xn -= 1.0;
        xn2 = xn * xn;
        values[2] = - 0.5 * xn * xn2 - xn2 + 2.0 / 3.0;

        // Knot at i + 2
        xn -= 1.0;
        xn2 = xn * xn;
        values[3] = xn * xn2 / 6.0 + xn2 + 2.0 * xn + 4.0 / 3.0;
    }

    /*!
      \brief Calculate the value of the gradient of the spline at given ijk
      index in the stencil.
      \param weight The node interpolation weight.
      \param distance The physical distance between the particle and the
      node.
      \param rdx The inverse physical grid cell size.
      \param gradient Basis gradient at the given node.
    */
    template<typename Real>
    KOKKOS_INLINE_FUNCTION
    static void gradient( const Real weight,
                          const Real* distance,
                          const Real rdx,
                          Real gradient[3] )
    {
        Real w_dp = weight * 3.0 * rdx * rdx;
        gradient[Dim::I] = w_dp * distance[Dim::I];
        gradient[Dim::J] = w_dp * distance[Dim::J];
        gradient[Dim::K] = w_dp * distance[Dim::K];
    }
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_SPLINES_HPP
