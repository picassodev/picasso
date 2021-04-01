/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef PICASSO_FIELDTYPES_HPP
#define PICASSO_FIELDTYPES_HPP

#include <Picasso_BatchedLinearAlgebra.hpp>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <sstream>
#include <string>
#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
// General type indexer.
//---------------------------------------------------------------------------//
template <class T, int Size, int N, class Type, class... Types>
struct TypeIndexerImpl
{
    static constexpr std::size_t value =
        TypeIndexerImpl<T, Size, N - 1, Types...>::value *
        ( std::is_same<T, Type>::value ? Size - 1 - N : 1 );
};

template <class T, int Size, class Type, class... Types>
struct TypeIndexerImpl<T, Size, 0, Type, Types...>
{
    static constexpr std::size_t value =
        std::is_same<T, Type>::value ? Size - 1 : 1;
};

template <class T, class... Types>
struct TypeIndexer
{
    static constexpr std::size_t index =
        TypeIndexerImpl<T, sizeof...( Types ), sizeof...( Types ) - 1,
                        Types...>::value;
};

//---------------------------------------------------------------------------//
// Field Layout
//---------------------------------------------------------------------------//
// Field layout. A layout contains a location and a field tag.
template <class Location, class Tag>
struct FieldLayout
{
    using location = Location;
    using tag = Tag;
};

//---------------------------------------------------------------------------//
// FieldViewTuple
//---------------------------------------------------------------------------//
// Device-accessible container for views of fields. This container allows us
// to wrap a parameter pack of views and let a user access them by the field
// location and field tag on the device.
template <class Views, class... Layouts>
struct FieldViewTuple
{
    static_assert( Cabana::is_parameter_pack<Views>::value,
                   "Views must be in a Cajita::ParameterPack" );

    Views _views;

    template <class Location, class FieldTag>
    KOKKOS_INLINE_FUNCTION const auto& get( Location, FieldTag ) const
    {
        return Cabana::get<
            TypeIndexer<FieldLayout<Location, FieldTag>, Layouts...>::index>(
            _views );
    }

    template <class Location, class FieldTag>
    KOKKOS_INLINE_FUNCTION auto& get( Location, FieldTag )
    {
        return Cabana::get<
            TypeIndexer<FieldLayout<Location, FieldTag>, Layouts...>::index>(
            _views );
    }
};

// Creation function.
template <class... Layouts, class Views>
auto createFieldViewTuple( const Views& v )
{
    return FieldViewTuple<Views, Layouts...>{ v };
}

//---------------------------------------------------------------------------//
// Field Location
//---------------------------------------------------------------------------//
namespace FieldLocation
{
struct Cell
{
    using entity_type = Cajita::Cell;
    static std::string label() { return "Cell"; }
};

template <int D>
struct Face
{
    using entity_type = Cajita::Face<D>;
    static std::string label()
    {
        std::stringstream l;
        l << "Face_" << D;
        return l.str();
    }
};

template <int D>
struct Edge
{
    using entity_type = Cajita::Edge<D>;
    static std::string label()
    {
        std::stringstream l;
        l << "Edge_" << D;
        return l.str();
    }
};

struct Node
{
    using entity_type = Cajita::Node;
    static std::string label() { return "Node"; }
};

struct Particle
{
    static std::string label() { return "Particle"; }
};

} // end namespace FieldLocation

//---------------------------------------------------------------------------//

namespace Field
{
//---------------------------------------------------------------------------//
// Field Tags.
//---------------------------------------------------------------------------//
// Forward declarations.
template <class T>
struct Scalar;
template <class T, int D>
struct Vector;
template <class T, int D0, int D1>
struct Tensor;

//---------------------------------------------------------------------------//
// Scalar field.
struct ScalarBase
{
};

template <class T>
struct Scalar : ScalarBase
{
    using value_type = T;
    static constexpr int rank = 0;
    static constexpr int size = 1;
    using data_type = value_type;
    using linear_algebra_type = value_type;
    template <class U>
    using field_type = Scalar<U>;
    template <int NumSpaceDim>
    using gradient_type = Vector<T, NumSpaceDim>;
};

template <class T>
struct is_scalar_impl : std::is_base_of<ScalarBase, T>
{
};

template <class T>
struct is_scalar : is_scalar_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Vector field.
struct VectorBase
{
};

template <class T, int D>
struct Vector : VectorBase
{
    using value_type = T;
    static constexpr int rank = 1;
    static constexpr int size = D;
    static constexpr int dim0 = D;
    using data_type = value_type[D];
    using linear_algebra_type = LinearAlgebra::VectorView<T, D>;
    template <class U>
    using field_type = Vector<U, D>;
    template <int NumSpaceDim>
    using gradient_type = Tensor<T, D, NumSpaceDim>;
};

template <class T>
struct is_vector_impl : std::is_base_of<VectorBase, T>
{
};

template <class T>
struct is_vector : is_vector_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Tensor Field.
struct TensorBase
{
};

template <class T, int D0, int D1>
struct Tensor : TensorBase
{
    using value_type = T;
    static constexpr int rank = 2;
    static constexpr int size = D0 * D1;
    static constexpr int dim0 = D0;
    static constexpr int dim1 = D1;
    using data_type = value_type[D0][D1];
    using linear_algebra_type = LinearAlgebra::MatrixView<T, D0, D1>;
    template <class U>
    using field_type = Tensor<U, D0, D1>;
};

template <class T>
struct is_tensor_impl : std::is_base_of<TensorBase, T>
{
};

template <class T>
struct is_tensor : is_tensor_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Scalar Field View Wrapper
//---------------------------------------------------------------------------//
// Wraps a Kokkos view of a structured grid scalar field at the given index
// allowing for it to be treated as a scalar in a point-wise manner in kernel
// operations without needing explicit dimension indices in the syntax.
template <class View, class Layout>
struct ScalarViewWrapper
{
    using layout_type = Layout;
    using field_tag = typename layout_type::tag;
    using field_location = typename layout_type::location;
    using value_type = typename field_tag::value_type;
    using linear_algebra_type = typename field_tag::linear_algebra_type;

    static constexpr int view_rank = View::Rank;

    static_assert( Field::is_scalar<typename layout_type::tag>::value,
                   "ScalarViewWrappers may only be applied to scalar fields" );

    View _v;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    ScalarViewWrapper() = default;

    // Create a wrapper from an index and a view.
    KOKKOS_INLINE_FUNCTION
    ScalarViewWrapper( const View& v )
        : _v( v )
    {
    }

    // Access the view data through point-wise index arguments.
    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<4 == VR, linear_algebra_type&>
    operator()( const int i0, const int i1, const int i2 ) const
    {
        return _v( i0, i1, i2, 0 );
    }

    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<3 == VR, linear_algebra_type&>
    operator()( const int i0, const int i1 ) const
    {
        return _v( i0, i1, 0 );
    }

    // Access the view data through full index arguments.
    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<4 == VR, value_type&>
    operator()( const int i0, const int i1, const int i2, const int ) const
    {
        return _v( i0, i1, i2, 0 );
    }

    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<3 == VR, value_type&>
    operator()( const int i0, const int i1, const int ) const
    {
        return _v( i0, i1, 0 );
    }
};

//---------------------------------------------------------------------------//
// Vector Field View Wrapper
//---------------------------------------------------------------------------//
// Wraps a Kokkos view of a structured grid vector field at the given index
// allowing for it to be treated as a vector in a point-wise manner in kernel
// operations without needing explicit dimension indices in the syntax.
template <class View, class Layout>
struct VectorViewWrapper
{
    using layout_type = Layout;
    using field_tag = typename layout_type::tag;
    using field_location = typename layout_type::location;
    using value_type = typename layout_type::tag::value_type;
    using linear_algebra_type = typename field_tag::linear_algebra_type;

    static constexpr int view_rank = View::Rank;

    static constexpr int dim0 = layout_type::tag::dim0;

    static_assert( Field::is_vector<typename layout_type::tag>::value,
                   "VectorViewWrappers may only be applied to vector fields" );

    View _v;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    VectorViewWrapper() = default;

    // Create a wrapper from an index and a view.
    KOKKOS_INLINE_FUNCTION
    VectorViewWrapper( const View& v )
        : _v( v )
    {
    }

    // Access the view data as a vector through point-wise index arguments.
    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<4 == VR, linear_algebra_type>
    operator()( const int i0, const int i1, const int i2 ) const
    {
        return linear_algebra_type( &_v( i0, i1, i2, 0 ), _v.stride( 3 ) );
    }

    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<3 == VR, linear_algebra_type>
    operator()( const int i0, const int i1 ) const
    {
        return linear_algebra_type( &_v( i0, i1, 0 ), _v.stride( 2 ) );
    }

    // Access the view data through full index arguments.
    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<4 == VR, value_type&>
    operator()( const int i0, const int i1, const int i2, const int i3 ) const
    {
        return _v( i0, i1, i2, i3 );
    }

    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<3 == VR, value_type&>
    operator()( const int i0, const int i1, const int i2 ) const
    {
        return _v( i0, i1, i2 );
    }
};

//---------------------------------------------------------------------------//
// Tensor Field View Wrapper
//---------------------------------------------------------------------------//
// Wraps a Kokkos view of a structured grid tensor field at the given index
// allowing for it to be treated as a tensor in a point-wise manner in kernel
// operations without needing explicit dimension indices in the syntax.
template <class View, class Layout>
struct TensorViewWrapper
{
    using layout_type = Layout;
    using field_tag = typename layout_type::tag;
    using field_location = typename layout_type::location;
    using value_type = typename layout_type::tag::value_type;
    using linear_algebra_type = typename field_tag::linear_algebra_type;

    static constexpr int view_rank = View::Rank;

    static constexpr int dim0 = layout_type::tag::dim0;
    static constexpr int dim1 = layout_type::tag::dim1;

    static_assert( Field::is_tensor<typename layout_type::tag>::value,
                   "TensorViewWrappers may only be applied to tensor fields" );

    View _v;

    // Default constructor.
    KOKKOS_DEFAULTED_FUNCTION
    TensorViewWrapper() = default;

    // Create a wrapper from an index and a view.
    KOKKOS_INLINE_FUNCTION
    TensorViewWrapper( const View& v )
        : _v( v )
    {
    }

    // Access the view data as a tensor through point-wise index
    // arguments. Note here that because fields are stored as 4D objects the
    // tensor components are unrolled in the last dimension. We unpack the
    // field dimension index to add the extra matrix dimension in a similar
    // way as if we had made a 5D kokkos view such that the matrix data is
    // ordered as [i][j][k][dim0][dim1] if layout-right and
    // [dim0][dim1][k][j][i] if layout-left. Note the difference in
    // layout-left where the dim0 and dim1 dimensions are switched.
    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<4 == VR, linear_algebra_type>
    operator()( const int i0, const int i1, const int i2 ) const
    {
        return linear_algebra_type( &_v( i0, i1, i2, 0 ), dim1 * _v.stride( 3 ),
                                    _v.stride( 3 ) );
    }

    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<3 == VR, linear_algebra_type>
    operator()( const int i0, const int i1 ) const
    {
        return linear_algebra_type( &_v( i0, i1, 0 ), dim1 * _v.stride( 2 ),
                                    _v.stride( 2 ) );
    }

    // Access the view data through full index arguments.
    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<4 == VR, value_type&>
    operator()( const int i0, const int i1, const int i2, const int i3 ) const
    {
        return _v( i0, i1, i2, i3 );
    }

    template <int VR = view_rank>
    KOKKOS_FORCEINLINE_FUNCTION std::enable_if_t<3 == VR, value_type&>
    operator()( const int i0, const int i1, const int i2 ) const
    {
        return _v( i0, i1, i2 );
    }
};

//---------------------------------------------------------------------------//
// Field View Wrapper Creation
//---------------------------------------------------------------------------//
template <class View, class Layout>
auto createViewWrapper(
    Layout, const View& view,
    std::enable_if_t<Field::is_scalar<typename Layout::tag>::value, int*> = 0 )
{
    return ScalarViewWrapper<View, Layout>( view );
}

template <class View, class Layout>
auto createViewWrapper(
    Layout, const View& view,
    std::enable_if_t<Field::is_vector<typename Layout::tag>::value, int*> = 0 )
{
    return VectorViewWrapper<View, Layout>( view );
}

template <class View, class Layout>
auto createViewWrapper(
    Layout, const View& view,
    std::enable_if_t<Field::is_tensor<typename Layout::tag>::value, int*> = 0 )
{
    return TensorViewWrapper<View, Layout>( view );
}

//---------------------------------------------------------------------------//
// Fields
//---------------------------------------------------------------------------//
struct PhysicalPosition : Vector<double, 3>
{
    static std::string label() { return "physical_position"; }
};

struct LogicalPosition : Vector<double, 3>
{
    static std::string label() { return "logical_position"; }
};

struct SignedDistance : Scalar<double>
{
    static std::string label() { return "signed_distance"; }
};

struct DistanceEstimate : Scalar<double>
{
    static std::string label() { return "distance_estimate"; }
};

struct Color : Scalar<int>
{
    static std::string label() { return "color"; }
};

struct VolumeId : Scalar<int>
{
    static std::string label() { return "volume_id"; }
};

struct BoundaryId : Scalar<int>
{
    static std::string label() { return "boundary_id"; }
};

struct CommRank : Scalar<int>
{
    static std::string label() { return "comm_rank"; }
};

//---------------------------------------------------------------------------//

} // end namespace Field
} // end namespace Picasso

#endif // PICASSO_FIELDTYPES_HPP
