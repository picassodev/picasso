#ifndef PICASSO_FIELDTYPES_HPP
#define PICASSO_FIELDTYPES_HPP

#include <Picasso_BatchedLinearAlgebra.hpp>

#include <Cajita.hpp>

#include <string>
#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Field locations.
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
    static std::string label() { return std::string("Face_" + D); }
};

template<int D>
struct Edge
{
    using entity_type = Cajita::Edge<D>;
    static std::string label() { return std::string("Edge_" + D); }
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
// Field type tags
namespace Field
{
//---------------------------------------------------------------------------//
// Scalar field.
template<class T>
struct Scalar
{
    using value_type = T;
    static constexpr int size = 1;
    using data_type = value_type;
    using linear_algebra_type = value_type;
};

template <class>
struct is_scalar_impl : public std::false_type
{
};

template <class T>
struct is_scalar_impl<Scalar<T>>
    : public std::true_type
{
};

template <class T>
struct is_scalar : public is_scalar_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Vector field.
template<class T, int D>
struct Vector
{
    using value_type = T;
    static constexpr int size = D;
    static constexpr int dim0 = D;
    using data_type = value_type[D];
    using linear_algebra_type = LinearAlgebra::VectorView<T,D>;
};

template <class>
struct is_vector_impl : public std::false_type
{
};

template <class T, int D>
struct is_vector_impl<Vector<T,D>>
    : public std::true_type
{
};

template <class T>
struct is_vector : public is_vector_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Tensor Field.
template<class T, int D0, int D1>
struct Tensor
{
    using value_type = T;
    static constexpr int size = D0 * D1;
    static constexpr int dim0 = D0;
    static constexpr int dim1 = D1;
    using data_type = value_type[D0][D1];
    using linear_algebra_type = LinearAlgebra::MatrixView<T,D0,D1>;
};

template <class>
struct is_tensor_impl : public std::false_type
{
};

template <class T, int D0, int D1>
struct is_tensor_impl<Tensor<T,D0,D1>>
    : public std::true_type
{
};

template <class T>
struct is_tensor : public is_tensor_impl<typename std::remove_cv<T>::type>::type
{
};

//---------------------------------------------------------------------------//
// Fields.
struct PhysicalPosition : public Vector<double,3>
{
    static std::string label() { return "physical_position"; }
};

struct LogicalPosition : public Vector<double,3>
{
    static std::string label() { return "logical_position"; }
};

struct Mass : public Scalar<double>
{
    static std::string label() { return "mass"; }
};

struct Volume : public Scalar<double>
{
    static std::string label() { return "volume"; }
};

struct Momentum : public Vector<double,3>
{
    static std::string label() { return "momentum"; }
};

struct Velocity : public Vector<double,3>
{
    static std::string label() { return "velocity"; }
};

struct Acceleration : public Vector<double,3>
{
    static std::string label() { return "acceleration"; }
};

struct AffineVelocity : public Tensor<double,3,3>
{
    static std::string label() { return "affine_velocity"; }
};

template<int N>
struct PolynomialVelocity : public Tensor<double,N,3>
{
    static constexpr int num_mode = N;
    static std::string label() { return "polynomial_velocity"; }
};

struct Normal : public Vector<double,3>
{
    static std::string label() { return "normal"; }
};

struct Temperature : public Scalar<double>
{
    static std::string label() { return "temperature"; }
};

struct HeatFlux : public Vector<double,3>
{
    static std::string label() { return "heat_flux"; }
};

struct Pressure : public Scalar<double>
{
    static std::string label() { return "pressure"; }
};

struct InternalEnergy : public Scalar<double>
{
    static std::string label() { return "internal_energy"; }
};

struct Density : public Scalar<double>
{
    static std::string label() { return "density"; }
};

struct Stress : public Tensor<double,3,3>
{
    static std::string label() { return "stress"; }
};

struct DeformationGradient : public Tensor<double,3,3>
{
    static std::string label() { return "deformation_gradient"; }
};

struct DeformationGradientDeterminant : public Scalar<double>
{
    static std::string label() { return "deformation_gradient_det"; }
};

struct SignedDistance : public Scalar<double>
{
    static std::string label() { return "signed_distance"; }
};

struct Color : public Scalar<int>
{
    static std::string label() { return "color"; }
};

struct MaterialId : public Scalar<int>
{
    static std::string label() { return "material_id"; }
};

struct PartId : public Scalar<int>
{
    static std::string label() { return "part_id"; }
};

struct VolumeId : public Scalar<int>
{
    static std::string label() { return "volume_id"; }
};

struct BoundaryId : public Scalar<int>
{
    static std::string label() { return "boundary_id"; }
};

//---------------------------------------------------------------------------//

} // end namespace Field
} // end namespace Picasso

#endif // PICASSO_FIELDTYPES_HPP
