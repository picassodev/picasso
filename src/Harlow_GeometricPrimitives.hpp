#ifndef HARLOW_GEOMETRICPRIMITIVES_HPP
#define HARLOW_GEOMETRICPRIMITIVES_HPP

#include <Kokkos_Core.hpp>

#include <utility>

namespace Harlow
{
namespace Geometry
{
namespace Primitives
{
//---------------------------------------------------------------------------//
// Base class.
//---------------------------------------------------------------------------//
template<class MemorySpace>
struct ObjectBase
{
    using memory_space = MemorySpace;

    KOKKOS_FUNCTION
    virtual ~ObjectBase() = default;

    // Determine if a point is inside the primitive
    KOKKOS_FUNCTION
    virtual bool inside( const double x[3] ) const = 0;

    // Get the axis-aligned bounding box of the primitive. Organized as:
    // {-x,-y,-z,+x,+y,+z}
    KOKKOS_FUNCTION
    virtual void boundingBox( double box[6] ) const = 0;
};

//---------------------------------------------------------------------------//
// Interface.
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
// Creation and destruction.
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
// Shapes.
//---------------------------------------------------------------------------//
template<class MemorySpace>
struct Brick : public ObjectBase<MemorySpace>
{
    using memory_space = MemorySpace;

    // Axis-aligned brick centered at a given origin.
    KOKKOS_FUNCTION
    Brick( const Kokkos::Array<double,3> extent,
           const Kokkos::Array<double,3> origin )
    {
        for ( int d = 0; d < 3; ++d )
        {
            _brick[d] = origin[d] - extent[d] * 0.5;
            _brick[d+3] = _brick[d] + extent[d];
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
    type* create( const Kokkos::Array<double,3> extent,
                  const Kokkos::Array<double,3> origin )
    {
        auto obj = (type*) Kokkos::kokkos_malloc<MemorySpace>(sizeof(type));
        Kokkos::parallel_for(
            "create_union",
            Kokkos::RangePolicy<execution_space>(0,1),
            KOKKOS_LAMBDA( const int ){
                new ((type*)obj) type(extent,origin);
            });
        return obj;
    }
};

//---------------------------------------------------------------------------//

} // end namespace Primitives
} // end namespace Geometry
} // end namespace Harlow

#endif // end HARLOW_GEOMETRICPRIMITIVES_HPP
