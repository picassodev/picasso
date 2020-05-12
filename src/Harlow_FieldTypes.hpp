#ifndef HARLOW_FIELDTYPES_HPP
#define HARLOW_FIELDTYPES_HPP

#include <string>

namespace Harlow
{
namespace Field
{
//---------------------------------------------------------------------------//
// Fundamental field types.
template<class T>
struct Scalar
{
    using value_type = T;
    static constexpr int dim = 1;
    using data_type = value_type;
};

template<class T, int D>
struct Vector
{
    using value_type = T;
    static constexpr int dim = D;
    using data_type = value_type[D];
};

template<class T, int D0, int D1>
struct Tensor
{
    using value_type = T;
    static constexpr int dim = D0 * D1;
    using data_type = value_type[D0][D1];
};

//---------------------------------------------------------------------------//
// Fields.
struct PhysicalPosition : public Vector<double,3>
{
    static std::string label() { return "physical_position"; };
};

struct LogicalPosition : public Vector<double,3>
{
    static std::string label() { return "logical_position"; };
};

struct Mass : public Scalar<double>
{
    static std::string label() { return "mass"; };
};

struct Volume : public Scalar<double>
{
    static std::string label() { return "volume"; };
};

struct Momentum : public Vector<double,3>
{
    static std::string label() { return "momentum"; };
};

struct Velocity : public Vector<double,3>
{
    static std::string label() { return "velocity"; };
};

struct Acceleration : public Vector<double,3>
{
    static std::string label() { return "acceleration"; };
};

struct AffineVelocity : public Tensor<double,3,3>
{
    static std::string label() { return "affine_velocity"; };
};

template<int N>
struct PolynomialVelocity : public Tensor<double,N,3>
{
    static std::string label() { return "polynomial_velocity"; };
};

struct Normal : public Vector<double,3>
{
    static std::string label() { return "normal"; };
};

struct Temperature : public Scalar<double>
{
    static std::string label() { return "temperature"; };
};

struct Pressure : public Scalar<double>
{
    static std::string label() { return "pressure"; };
};

struct InternalEnergy : public Scalar<double>
{
    static std::string label() { return "internal_energy"; };
};

struct Density : public Scalar<double>
{
    static std::string label() { return "density"; };
};

struct Stress : public Tensor<double,3,3>
{
    static std::string label() { return "stress"; };
};

struct DeformationGradient : public Tensor<double,3,3>
{
    static std::string label() { return "deformation_gradient"; };
};

struct DeformationGradientDeterminant : public Scalar<double>
{
    static std::string label() { return "deformation_gradient_det"; };
};

struct SignedDistance : public Scalar<double>
{
    static std::string label() { return "signed_distance"; };
};

struct Color : public Scalar<double>
{
    static std::string label() { return "color"; };
};

struct MaterialId : public Scalar<int>
{
    static std::string label() { return "material_id"; };
};

struct PartId : public Scalar<int>
{
    static std::string label() { return "part_id"; };
};

struct VolumeId : public Scalar<int>
{
    static std::string label() { return "volume_id"; };
};

struct BoundaryId : public Scalar<int>
{
    static std::string label() { return "boundary_id"; };
};

//---------------------------------------------------------------------------//

} // end namespace Field
} // end namespace Harlow

#endif // HARLOW_FIELDTYPES_HPP
