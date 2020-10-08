#ifndef PICASSO_FIELDTYPES_HPP
#define PICASSO_FIELDTYPES_HPP

#include <Picasso_BatchedLinearAlgebra.hpp>

#include <Cajita.hpp>

#include <string>
#include <sstream>
#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
// General type indexer.
//---------------------------------------------------------------------------//
template<class T, int Size, int N, class Type, class ... Types>
struct TypeIndexerImpl
{
    static constexpr std::size_t value =
        TypeIndexerImpl<T,Size,N-1,Types...>::value *
        (std::is_same<T,Type>::value ? Size - 1 - N : 1);
};

template<class T, int Size, class Type, class ... Types>
struct TypeIndexerImpl<T,Size,0,Type,Types...>
{
    static constexpr std::size_t value =
        std::is_same<T,Type>::value ? Size - 1 : 1;
};

template<class T, class ... Types>
struct TypeIndexer
{
    static constexpr std::size_t index =
        TypeIndexerImpl<T,sizeof...(Types),sizeof...(Types)-1,Types...>::value;
};

//---------------------------------------------------------------------------//
// Field Layout
//---------------------------------------------------------------------------//
// Field layout. A layout contains a location and a field tag.
template<class Location, class Tag>
struct FieldLayout
{
    using location = Location;
    using tag = Tag;
};

//---------------------------------------------------------------------------//
// FieldViewTuple
// ---------------------------------------------------------------------------//
// Device-accessible container for views of fields. This container allows us
// to wrap a parameter pack of views and let a user access them by the field
// location and field tag on the device.
template<class Views, class ... Layouts>
struct FieldViewTuple
{
    static_assert( Cajita::is_parameter_pack<Views>::value,
                   "Views must be in a Cajita::ParameterPack" );

    Views _views;

    template<class Location, class FieldTag>
    KOKKOS_INLINE_FUNCTION
    const auto& get( Location, FieldTag ) const
    {
        return
            Cajita::get<
                TypeIndexer<FieldLayout<Location,FieldTag>,Layouts...>::index>(
                    _views );
    }

    template<class Location, class FieldTag>
    KOKKOS_INLINE_FUNCTION
    auto& get( Location, FieldTag )
    {
        return
            Cajita::get<
                TypeIndexer<FieldLayout<Location,FieldTag>,Layouts...>::index>(
                    _views );
    }
};

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

template<int D>
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

template<int D>
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
// Field Tags.
//---------------------------------------------------------------------------//
namespace Field
{
//---------------------------------------------------------------------------//
// Scalar field.
template<class T>
struct Scalar
{
    using value_type = T;
    static constexpr int rank = 0;
    static constexpr int size = 1;
    using data_type = value_type;
    using linear_algebra_type = value_type;
};

template <class>
struct is_scalar_impl : std::false_type
{
};

template <class T>
struct is_scalar_impl<Scalar<T>>
    : std::true_type
{
};

template <class T>
struct is_scalar : is_scalar_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Vector field.
template<class T, int D>
struct Vector
{
    using value_type = T;
    static constexpr int rank = 1;
    static constexpr int size = D;
    static constexpr int dim0 = D;
    using data_type = value_type[D];
    using linear_algebra_type = LinearAlgebra::VectorView<T,D>;
};

template <class>
struct is_vector_impl : std::false_type
{
};

template <class T, int D>
struct is_vector_impl<Vector<T,D>>
    : std::true_type
{
};

template <class T>
struct is_vector : is_vector_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Tensor Field.
template<class T, int D0, int D1>
struct Tensor
{
    using value_type = T;
    static constexpr int rank = 2;
    static constexpr int size = D0 * D1;
    static constexpr int dim0 = D0;
    static constexpr int dim1 = D1;
    using data_type = value_type[D0][D1];
    using linear_algebra_type = LinearAlgebra::MatrixView<T,D0,D1>;
};

template <class>
struct is_tensor_impl : std::false_type
{
};

template <class T, int D0, int D1>
struct is_tensor_impl<Tensor<T,D0,D1>>
    : std::true_type
{
};

template <class T>
struct is_tensor : is_tensor_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Fields.
struct PhysicalPosition : Vector<double,3>
{
    static std::string label() { return "physical_position"; }
};

struct LogicalPosition : Vector<double,3>
{
    static std::string label() { return "logical_position"; }
};

struct Mass : Scalar<double>
{
    static std::string label() { return "mass"; }
};

struct Volume : Scalar<double>
{
    static std::string label() { return "volume"; }
};

struct Momentum : Vector<double,3>
{
    static std::string label() { return "momentum"; }
};

struct Velocity : Vector<double,3>
{
    static std::string label() { return "velocity"; }
};

struct Acceleration : Vector<double,3>
{
    static std::string label() { return "acceleration"; }
};

struct AffineVelocity : Tensor<double,3,3>
{
    static std::string label() { return "affine_velocity"; }
};

template<int N>
struct PolynomialVelocity : Tensor<double,N,3>
{
    static constexpr int num_mode = N;
    static std::string label() { return "polynomial_velocity"; }
};

struct Normal : Vector<double,3>
{
    static std::string label() { return "normal"; }
};

struct Temperature : Scalar<double>
{
    static std::string label() { return "temperature"; }
};

struct HeatFlux : Vector<double,3>
{
    static std::string label() { return "heat_flux"; }
};

struct Pressure : Scalar<double>
{
    static std::string label() { return "pressure"; }
};

struct InternalEnergy : Scalar<double>
{
    static std::string label() { return "internal_energy"; }
};

struct Density : Scalar<double>
{
    static std::string label() { return "density"; }
};

struct Stress : Tensor<double,3,3>
{
    static std::string label() { return "stress"; }
};

struct DeformationGradient : Tensor<double,3,3>
{
    static std::string label() { return "deformation_gradient"; }
};

struct DeformationGradientDeterminant : Scalar<double>
{
    static std::string label() { return "deformation_gradient_det"; }
};

struct SignedDistance : Scalar<double>
{
    static std::string label() { return "signed_distance"; }
};

struct Color : Scalar<int>
{
    static std::string label() { return "color"; }
};

struct MaterialId : Scalar<int>
{
    static std::string label() { return "material_id"; }
};

struct PartId : Scalar<int>
{
    static std::string label() { return "part_id"; }
};

struct VolumeId : Scalar<int>
{
    static std::string label() { return "volume_id"; }
};

struct BoundaryId : Scalar<int>
{
    static std::string label() { return "boundary_id"; }
};

//---------------------------------------------------------------------------//

} // end namespace Field
} // end namespace Picasso

#endif // PICASSO_FIELDTYPES_HPP
