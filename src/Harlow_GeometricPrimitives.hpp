#ifndef HARLOW_GEOMETRICPRIMITIVES_HPP
#define HARLOW_GEOMETRICPRIMITIVES_HPP

#include <Kokkos_Core.hpp>

#include <utility>
#include <cmath>
#include <limits>

namespace Harlow
{
namespace Geometry
{
namespace Primitives
{
//---------------------------------------------------------------------------//
// Quadric surface.
//
// Creates a quadric surface defined by the equation:
// f(x,y,z) = a*x^2 + b*x*y + c*x*z + d*x + e*y^2 + f*y*z + g*y +
//            h*z^2 + i*z + j
//          = 0
//---------------------------------------------------------------------------//
template<class Scalar>
struct Surface
{
    // Given a point compute the closest orthogonal projection to the
    // surface given by the unit normal vector and signed distance.
    KOKKOS_INLINE_FUNCTION
    Scalar closestProjection( const Scalar x[3],
                              Scalar n[3],
                              Scalar& distance ) const
    {
        // Use the tetrahedron height intersection approximation. Start by
        // computing the base of the tetrahedron (r,y,z), (x,s,z), (x,y,t).
        Scalar ra = _a;
        Scalar rb = _b*x[1] + _c*x[2] + _d;
        Scalar rc = _e*x[1]*x[1] + _f*x[1]*x[2] + _g*x[1] + _h*x[2]*x[2] +
                    +_i*x[2] + j;
        Scalar rd = sqrt(rb*rb-4.0*ra*rc);
        Scalar r[3] = { fmin(-rb+rd,-rb-rd), x[1], x[2]};
        r[0] /= 2.0 * ra;

        Scalar sa = _e;
        Scalar sb = _b*x[0] + _f*x[2] + _g;
        Scalar sc = _a*x[0]x[0] + _c*x[0]*x[2] + _d*x[0] + _h*x[2]*x[2] +
                    +_i*x[2] + j;
        Scalar sd = sqrt(sb*sb-4.0*sa*sc);
        Scalar s[3] = { x[0], fmin(-sb+sd,-sb-sd), x[2] };
        s[1] /= 2.0 * sa;

        Scalar ta = _h;
        Scalar tb = _c*x[0] + _f*x[1] + _i;
        Scalar tc = _a*x[0]x[0] + _b*x[0]*x[1] + _d*x[0] + _e*x[1]*x[1] +
                    _g*x[1] + j;
        Scalar td = sqrt(tb*tb-4.0*ta*tc);
        Scalar t[3] = { x[0], x[1], fmin(-tb+td,-tb-td) };
        t[2] /= 2.0 * ta;

        // Compute the unit normal of the plane formed by the tetrahedron
        // base.
        Scalar sr[3] = { s[0]-r[0], s[1]-r[1], s[2]-r[2] };
        Scalar tr[3] = { t[0]-r[0], t[1]-r[1], t[2]-r[2] };
        n[0] = sr[1]*tr[2] - sr[2]*tr[1];
        n[1] = sr[2]*tr[0] - sr[0]*tr[2];
        n[2] = sr[0]*tr[1] - sr[1]*tr[0];
        Scalar nmag_inv = 1.0 / sqrt( n[0]*n[0] + n[1]*n[1] + n[2]*n[2] );
        for ( int d = 0; d < 3; ++d )
            n[d] *= nmag_inv;

        // Compute the signed distance.
        distance = (x[0]-r[0])*n[0] + (x[1]-r[1])*n[1] + (x[2]-r[2])*n[2];
    }

    Scalar _a;
    Scalar _b;
    Scalar _c;
    Scalar _d;
    Scalar _e;
    Scalar _f;
    Scalar _g;
    Scalar _h;
    Scalar _i;
    Scalar _j;
};

//---------------------------------------------------------------------------//
// Primitive surfaces.
//---------------------------------------------------------------------------//
// Create a general plane: ax + by + cz + d = 0
template<class Scalar>
void createPlane( const Scalar a, const Scalar b, const Scalar c,
                  const Scalar d, Surface<Scalar>& s )
{
    s._a = 0.0;
    s._b = 0.0;
    s._c = 0.0;
    s._d = a;
    s._e = 0.0;
    s._f = 0.0;
    s._g = b;
    s._h = 0.0;
    s._i = c;
    s._j = d;
}

//---------------------------------------------------------------------------//
// Create a plane in XY at the given coordinate: z = constant
template<class Scalar>
void createPlaneXY( const Scalar c, Surface<Scalar>& s )
{
    createPlane( 0.0, 0.0, 1.0, c, s );
}

//---------------------------------------------------------------------------//
// Create a plane in XZ at the given coordinate: y = constant
template<class Scalar>
void createPlaneXZ( const Scalar c, Surface<Scalar>& s )
{
    createPlane( 0.0, 1.0, 0.0, c, s );
}

//---------------------------------------------------------------------------//
// Create a plane in YZ at the given coordinate: x = constant
template<class Scalar>
void createPlaneYZ( const Scalar c, Surface<Scalar>& s )
{
    createPlane( 1.0, 0.0, 0.0, c, s );
}

//---------------------------------------------------------------------------//
// Create a general ellipsoid: x^2/a^2 + y^2/b^2 + z^2/c^2 = 1
template<class Scalar>
void createEllipsoid( const Scalar a, const Scalar b, const Scalar c,
                      Surface<Scalar>& s )
{
    s._a = 1.0 / (a*a);
    s._b = 0.0;
    s._c = 0.0;
    s._d = 0.0;
    s._e = 1.0 / (b*b);
    s._f = 0.0;
    s._g = 0.0;
    s._h = 1.0 / (c*c);
    s._i = 0.0;
    s._j = -1.0;
}

//---------------------------------------------------------------------------//
// Create a sphere: x^2/r^2 + y^2/r^2 + z^2/r^2 = 1
template<class Scalar>
void createSphere( const Scalar r, Surface<Scalar>& s )
{
    createEllipsoid( r, r, r, s );
}

//---------------------------------------------------------------------------//
// Create an elliptic cylinder:

//---------------------------------------------------------------------------//
// Base class
//---------------------------------------------------------------------------//
template<class MemorySpace>
struct ObjectBase
{
    using memory_space = MemorySpace;

    // Note that using "= default" caused a CUDA segfault so we explictly put
    // the "{}" here.
    KOKKOS_FUNCTION
    virtual ~ObjectBase() {}

    // Determine if a point is inside the primitive
    KOKKOS_FUNCTION
    virtual bool inside( const double x[3] ) const = 0;

    // Get the axis-aligned bounding box of the primitive. Organized as:
    // {-x,-y,-z,+x,+y,+z}
    KOKKOS_FUNCTION
    virtual void boundingBox( double box[6] ) const = 0;
};

//---------------------------------------------------------------------------//
// Interface
//---------------------------------------------------------------------------//
template<class MemorySpace>
struct Object
{
    using memory_space = MemorySpace;

    // Determine if a point is inside the primitive
    KOKKOS_FUNCTION
    bool inside( const double x[3] ) const
    {
        return _impl->inside( x );
    }

    // Get the axis-aligned bounding box of the primitive. Organized as:
    // {-x,-y,-z,+x,+y,+z}
    KOKKOS_FUNCTION
    void boundingBox( double box[6] ) const
    {
        _impl->boundingBox( box );
    }

    ObjectBase<MemorySpace>* _impl;
};

//---------------------------------------------------------------------------//
// Creation and destruction
//---------------------------------------------------------------------------//
template<class MemorySpace, class Builder, class ... Args>
void create( Object<MemorySpace>& obj, Builder, Args&& ... args )
{
    obj._impl = Builder::create(std::forward<Args>(args)...);
}

template<class MemorySpace>
void destroy( Object<MemorySpace>& obj )
{
    Kokkos::parallel_for(
        "destroy_primitive",
        Kokkos::RangePolicy<typename MemorySpace::execution_space>(0,1),
        KOKKOS_LAMBDA( const int ){
            obj._impl->~ObjectBase<MemorySpace>();
        });

    Kokkos::kokkos_free<MemorySpace>( obj._impl );
}

//---------------------------------------------------------------------------//
// Boolean operations
//---------------------------------------------------------------------------//
template<class MemorySpace>
struct Union : public ObjectBase<MemorySpace>
{
    using memory_space = MemorySpace;

    // Construct the union of A and B.
    KOKKOS_FUNCTION
    Union( const Object<MemorySpace>& a,
           const Object<MemorySpace>& b )
        : _a( a )
        , _b( b )
    {}

    // Determine if a point is inside the primitive
    KOKKOS_FUNCTION
    bool inside( const double x[3] ) const override
    {
        return (_a.inside(x) || _b.inside(x));
    }

    // Get the axis-aligned bounding box of the primitive
    KOKKOS_FUNCTION
    void boundingBox( double box[6] ) const override
    {
        double box_a[6];
        double box_b[6];
        _a.boundingBox( box_a );
        _b.boundingBox( box_b );

        for ( int d = 0; d < 3; ++d )
        {
            box[d] = fmin( box_a[d], box_b[d] );
            box[d+3] = fmax( box_a[d+3], box_b[d+3] );
        }
    }

    Object<MemorySpace> _a;
    Object<MemorySpace> _b;
};

template<class MemorySpace>
struct UnionBuilder
{
    using type = Union<MemorySpace>;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    static
    type* create( const Object<MemorySpace>& a, const Object<MemorySpace>& b )
    {
        auto obj = (type*) Kokkos::kokkos_malloc<MemorySpace>(sizeof(type));
        Kokkos::parallel_for(
            "create_union",
            Kokkos::RangePolicy<execution_space>(0,1),
            KOKKOS_LAMBDA( const int ){
                new ((type*)obj) type(a,b);
            });
        return obj;
    }
};

//---------------------------------------------------------------------------//
template<class MemorySpace>
struct Difference : public ObjectBase<MemorySpace>
{
    using memory_space = MemorySpace;

    // Construct the difference of A and B. Order if operations is (A - B). In
    // other words, the new object is whatever was in A but not in B.
    KOKKOS_FUNCTION
    Difference( const Object<MemorySpace>& a,
                const Object<MemorySpace>& b )
        : _a( a )
        , _b( b )
    {}

    // Determine if a point is inside the primitive
    KOKKOS_FUNCTION
    bool inside( const double x[3] ) const override
    {
        return (_a.inside(x) && !_b.inside(x));
    }

    // Get the axis-aligned bounding box of the primitive
    KOKKOS_FUNCTION
    void boundingBox( double box[6] ) const override
    {
        // This implementation is not an optimal one but it is "safe" such
        // that the bounding box of (A-B) can't be larger than the bounding
        // box of A.
        _a.boundingBox( box );
    }

    Object<MemorySpace> _a;
    Object<MemorySpace> _b;
};

template<class MemorySpace>
struct DifferenceBuilder
{
    using type = Difference<MemorySpace>;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    static
    type* create( const Object<MemorySpace>& a, const Object<MemorySpace>& b )
    {
        auto obj = (type*) Kokkos::kokkos_malloc<MemorySpace>(sizeof(type));
        Kokkos::parallel_for(
            "create_union",
            Kokkos::RangePolicy<execution_space>(0,1),
            KOKKOS_LAMBDA( const int ){
                new ((type*)obj) type(a,b);
            });
        return obj;
    }
};

//---------------------------------------------------------------------------//
template<class MemorySpace>
struct Intersection : public ObjectBase<MemorySpace>
{
    using memory_space = MemorySpace;

    // Construct the intersection of A and B.
    KOKKOS_FUNCTION
    Intersection( const Object<MemorySpace>& a,
                  const Object<MemorySpace>& b )
        : _a( a )
        , _b( b )
    {}

    // Determine if a point is inside the primitive
    KOKKOS_FUNCTION
    bool inside( const double x[3] ) const override
    {
        return (_a.inside(x) && _b.inside(x));
    }

    // Get the axis-aligned bounding box of the primitive
    KOKKOS_FUNCTION
    void boundingBox( double box[6] ) const override
    {
        double box_a[6];
        double box_b[6];
        _a.boundingBox( box_a );
        _b.boundingBox( box_b );

        for ( int d = 0; d < 3; ++d )
        {
            box[d] = fmax( box_a[d], box_b[d] );
            box[d+3] = fmin( box_a[d+3], box_b[d+3] );
        }
    }

    Object<MemorySpace> _a;
    Object<MemorySpace> _b;
};

template<class MemorySpace>
struct IntersectionBuilder
{
    using type = Intersection<MemorySpace>;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    static
    type* create( const Object<MemorySpace>& a, const Object<MemorySpace>& b )
    {
        auto obj = (type*) Kokkos::kokkos_malloc<MemorySpace>(sizeof(type));
        Kokkos::parallel_for(
            "create_union",
            Kokkos::RangePolicy<execution_space>(0,1),
            KOKKOS_LAMBDA( const int ){
                new ((type*)obj) type(a,b);
            });
        return obj;
    }
};

//---------------------------------------------------------------------------//
// Transformations
//---------------------------------------------------------------------------//
template<class MemorySpace>
struct Move : public ObjectBase<MemorySpace>
{
    using memory_space = MemorySpace;

    // Move A a given distance in x, y, and z.
    KOKKOS_FUNCTION
    Move( const Object<MemorySpace>& a,
          const Kokkos::Array<double,3>& distance )
        : _a( a )
        , _dist( distance )
    {}

    // Determine if a point is inside the primitive.
    KOKKOS_FUNCTION
    bool inside( const double x[3] ) const override
    {
        // Note we move the point back by the distance vector to achieve the
        // affect of the underlying object actually being moved.
        double xd[3] = { x[0]-_dist[0], x[1]-_dist[1], x[2]-_dist[2] };
        return _a.inside(xd);
    }

    // Get the axis-aligned bounding box of the primitive
    KOKKOS_FUNCTION
    void boundingBox( double box[6] ) const override
    {
        _a.boundingBox( box );
        for ( int d = 0; d < 3; ++d )
        {
            box[d] += _dist[d];
            box[d+3] += _dist[d];
        }
    }

    Object<MemorySpace> _a;
    Kokkos::Array<double,3> _dist;
};

template<class MemorySpace>
struct MoveBuilder
{
    using type = Move<MemorySpace>;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    static
    type* create( const Object<MemorySpace>& a,
                  const Kokkos::Array<double,3>& distance )
    {
        auto obj = (type*) Kokkos::kokkos_malloc<MemorySpace>(sizeof(type));
        Kokkos::parallel_for(
            "create_union",
            Kokkos::RangePolicy<execution_space>(0,1),
            KOKKOS_LAMBDA( const int ){
                new ((type*)obj) type(a,distance);
            });
        return obj;
    }
};

//---------------------------------------------------------------------------//
template<class MemorySpace>
struct Rotate : public ObjectBase<MemorySpace>
{
    using memory_space = MemorySpace;

    // Rotate A a given angle about the given axis which passes through the
    // origin of the coordinate system (0,0,0). Angle is in
    // radians (0-2*PI).
    KOKKOS_FUNCTION
    Rotate( const Object<MemorySpace>& a,
            const double angle,
            const Kokkos::Array<double,3>& axis )
        : _a( a )
    {
        // Create a unit axis vector.
        double al = sqrt(axis[0]*axis[0]+axis[1]*axis[1]+axis[2]*axis[2]);
        double u[3] = { axis[0] / al, axis[1] / al, axis[2] / al };

        // Forward Cosine/Sine
        double caf = cos(angle);
        double saf = sin(angle);

        // Build the forward rotation matrix.
        // (0,0)
        _rmf[0] = u[0]*u[0]*(1.0-caf) + caf;

        // (0,1)
        _rmf[1] = u[0]*u[1]*(1.0-caf) - u[2]*saf;

        // (0,2)
        _rmf[2] = u[2]*u[0]*(1.0-caf) + u[1]*saf;

        // (1,0)
        _rmf[3] = u[0]*u[1]*(1.0-caf) + u[2]*saf;

        // (1,1)
        _rmf[4] = u[1]*u[1]*(1.0-caf) + caf;

        // (1,2)
        _rmf[5] = u[1]*u[2]*(1.0-caf) - u[0]*saf;

        // (2,0)
        _rmf[6] = u[2]*u[0]*(1.0-caf) - u[1]*saf;

        // (2,1)
        _rmf[7] = u[1]*u[2]*(1.0-caf) + u[0]*saf;

        // (2,2)
        _rmf[8] = u[2]*u[2]*(1.0-caf) + caf;

        // Reverse Cosine/Sine
        double car = cos(-angle);
        double sar = sin(-angle);

        // Build the reverse rotation matrix.
        // (0,0)
        _rmr[0] = u[0]*u[0]*(1.0-car) + car;

        // (0,1)
        _rmr[1] = u[0]*u[1]*(1.0-car) - u[2]*sar;

        // (0,2)
        _rmr[2] = u[2]*u[0]*(1.0-car) + u[1]*sar;

        // (1,0)
        _rmr[3] = u[0]*u[1]*(1.0-car) + u[2]*sar;

        // (1,1)
        _rmr[4] = u[1]*u[1]*(1.0-car) + car;

        // (1,2)
        _rmr[5] = u[1]*u[2]*(1.0-car) - u[0]*sar;

        // (2,0)
        _rmr[6] = u[2]*u[0]*(1.0-car) - u[1]*sar;

        // (2,1)
        _rmr[7] = u[1]*u[2]*(1.0-car) + u[0]*sar;

        // (2,2)
        _rmr[8] = u[2]*u[2]*(1.0-car) + car;
    }

    // Rotate a point about the axis in the direction of the given angle.
    KOKKOS_FUNCTION
    void rotateForward( const double x[3], double xr[3] ) const
    {
        xr[0] = _rmf[0]*x[0] + _rmf[1]*x[1] + _rmf[2]*x[2];
        xr[1] = _rmf[3]*x[0] + _rmf[4]*x[1] + _rmf[5]*x[2];
        xr[2] = _rmf[6]*x[0] + _rmf[7]*x[1] + _rmf[8]*x[2];
    }

    // Rotate a point about the axis in the opposite direction of the given
    // angle.
    KOKKOS_FUNCTION
    void rotateReverse( const double x[3], double xr[3] ) const
    {
        xr[0] = _rmr[0]*x[0] + _rmr[1]*x[1] + _rmr[2]*x[2];
        xr[1] = _rmr[3]*x[0] + _rmr[4]*x[1] + _rmr[5]*x[2];
        xr[2] = _rmr[6]*x[0] + _rmr[7]*x[1] + _rmr[8]*x[2];
    }

    // Determine if a point is inside the primitive
    KOKKOS_FUNCTION
    bool inside( const double x[3] ) const override
    {
        // Note we rotate the point in the reverse direction to achieve the
        // effect of the underlying object being rotated.
        double xr[3];
        rotateReverse( x, xr );
        return (_a.inside(xr));
    }

    // Get the axis-aligned bounding box of the primitive
    KOKKOS_FUNCTION
    void boundingBox( double box[6] ) const override
    {
        // Get the original box.
        _a.boundingBox( box );

        // Rotate it.
        double boxr[6];
        rotateForward( &box[0], &boxr[0] );
        rotateForward( &box[3], &boxr[3] );

        // Determine the bounding box of the rotated box. This isn't optimal
        // in terms of the smallest possible box but it is safe.
        for ( int d = 0; d < 3; ++d )
        {
            box[d] = fmin( boxr[d], boxr[d+3] );
            box[d+3] = fmax( boxr[d], boxr[d+3] );
        }
    }

    Object<MemorySpace> _a;
    Kokkos::Array<double,9> _rmf;
    Kokkos::Array<double,9> _rmr;
};

template<class MemorySpace>
struct RotateBuilder
{
    using type = Rotate<MemorySpace>;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    static
    type* create( const Object<MemorySpace>& a,
                  const double angle,
                  const Kokkos::Array<double,3>& axis )
    {
        auto obj = (type*) Kokkos::kokkos_malloc<MemorySpace>(sizeof(type));
        Kokkos::parallel_for(
            "create_union",
            Kokkos::RangePolicy<execution_space>(0,1),
            KOKKOS_LAMBDA( const int ){
                new ((type*)obj) type(a,angle,axis);
            });
        return obj;
    }
};

//---------------------------------------------------------------------------//
// Shapes. All primitive shapes have their center of mass at the origin.
//---------------------------------------------------------------------------//
template<class MemorySpace>
struct Brick : public ObjectBase<MemorySpace>
{
    using memory_space = MemorySpace;

    // Axis-aligned brick with the given extents.
    KOKKOS_FUNCTION
    Brick( const Kokkos::Array<double,3> extent )
    {
        for ( int d = 0; d < 3; ++d )
        {
            _brick[d] = -extent[d] * 0.5;
            _brick[d+3] = extent[d] * 0.5;
        }
    }

    // Determine if a point is inside the primitive
    KOKKOS_FUNCTION
    bool inside( const double x[3] ) const override
    {
        return ( x[0] >= _brick[0] && x[0] <= _brick[3] &&
                 x[1] >= _brick[1] && x[1] <= _brick[4] &&
                 x[2] >= _brick[2] && x[2] <= _brick[5] );
    }

    // Get the axis-aligned bounding box of the primitive
    KOKKOS_FUNCTION
    void boundingBox( double box[6] ) const override
    {
        for ( int i = 0; i < 6; ++i )
            box[i] = _brick[i];
    }

  private:

    Kokkos::Array<double,6> _brick;
};

template<class MemorySpace>
struct BrickBuilder
{
    using type = Brick<MemorySpace>;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    static
    type* create( const Kokkos::Array<double,3> extent )
    {
        auto obj = (type*) Kokkos::kokkos_malloc<MemorySpace>(sizeof(type));
        Kokkos::parallel_for(
            "create_union",
            Kokkos::RangePolicy<execution_space>(0,1),
            KOKKOS_LAMBDA( const int ){
                new ((type*)obj) type(extent);
            });
        return obj;
    }
};

//---------------------------------------------------------------------------//
template<class MemorySpace>
struct Sphere : public ObjectBase<MemorySpace>
{
    using memory_space = MemorySpace;

    // Sphere of the given radius.
    KOKKOS_FUNCTION
    Sphere( const double radius )
        : _radius( radius )
        , _r2( radius*radius )
    {}

    // Determine if a point is inside the primitive
    KOKKOS_FUNCTION
    bool inside( const double x[3] ) const override
    {
        double d2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
        return ( d2 <= _r2 );
    }

    // Get the axis-aligned bounding box of the primitive
    KOKKOS_FUNCTION
    void boundingBox( double box[6] ) const override
    {
        for ( int d = 0; d < 3; ++d )
        {
            box[d] = -_radius;
            box[d+3] = _radius;
        }
    }

  private:

    double _radius;
    double _r2;
    Kokkos::Array<double,3> _origin;
};

template<class MemorySpace>
struct SphereBuilder
{
    using type = Sphere<MemorySpace>;
    using memory_space = MemorySpace;
    using execution_space = typename memory_space::execution_space;

    static
    type* create( const double radius )
    {
        auto obj = (type*) Kokkos::kokkos_malloc<MemorySpace>(sizeof(type));
        Kokkos::parallel_for(
            "create_union",
            Kokkos::RangePolicy<execution_space>(0,1),
            KOKKOS_LAMBDA( const int ){
                new ((type*)obj) type(radius);
            });
        return obj;
    }
};

//---------------------------------------------------------------------------//

} // end namespace Primitives
} // end namespace Geometry
} // end namespace Harlow

#endif // end HARLOW_GEOMETRICPRIMITIVES_HPP
