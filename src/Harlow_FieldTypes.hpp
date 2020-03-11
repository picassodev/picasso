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
    using particle_member_type = value_type;
};

struct Vector
{
    using value_type = T;
    static constexpr int dim = 3;
    static constexpr particle_member_type = value_type[3];
};

struct Tensor
{
    using value_type = T;
    static constexpr int dim = 9;
    static constexpr particle_member_type = value_type[3][3];
};

//---------------------------------------------------------------------------//
// Fields.
struct PhysicalPosition : public Vector<double>
{
    static std::string label() { return "physical_position" };
};

struct LogicalPosition : public Vector<double>
{
    static std::string label() { return "logical_position" };
};

struct Mass : public Scalar<double>
{
    static std::string label() { return "mass" };
};

struct Volume : public Scalar<double>
{
    static std::string label() { return "volume" };
};

struct Momentum : public Vector<double>
{
    static std::string label() { return "momentum" };
};

struct Velocity : public Vector<double>
{
    static std::string label() { return "velocity" };
};

struct Temperature : public Scalar<double>
{
    static std::string label() { return "temperature" };
};

struct InternalEnergy : public Scalar<double>
{
    static std::string label() { return "internal_energy" };
};

struct Density : public Scalar<double>
{
    static std::string label() { return "density" };
};

struct DeformationGradient : public Tensor<double>
{
    static std::string label() { return "deformation_gradient" };
};

struct DeformationGradientDeterminant : public Tensor<double>
{
    static std::string label() { return "deformation_gradient_det" };
};

struct MaterialId : public Scalar<int>
{
    static std::string label() { return "material_id" };
};

struct Color : public Scalar<int>
{
    static std::string label() { return "color" };
};

//---------------------------------------------------------------------------//

} // end namespace Field
} // end namespace Harlow

#endif // HARLOW_FIELDTYPES_HPP
