#include <Picasso.hpp>

#include <Cabana_Core.hpp>
#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

namespace Picasso
{

namespace Example
{

struct Mass : Field::Scalar<double>
{
    static std::string label() { return "Mass"; }
};

struct Pressure : Field::Scalar<double>
{
    static std::string label() { return "Pressure"; }
};

struct Volume : Field::Scalar<double>
{
    static std::string label() { return "Volume"; }
};

struct Velocity : Field::Vector<double, 3>
{
    static std::string label() { return "Velocity"; }
};

struct OldU : Field::Vector<double, 3>
{
    static std::string label() { return "Old_Velocity"; }
};

struct DeltaUGravity : Field::Vector<double, 3>
{
    static std::string label() { return "velocity_change_from_gravity"; }
};

struct Stress : Field::Matrix<double, 3, 3>
{
    static std::string label() { return "stress"; }
};

struct DeltaUStress : Field::Vector<double, 3>
{
    static std::string label() { return "velocity_change_from_stress"; }
};

struct DetDefGrad : Field::Scalar<double>
{
    static std::string label() { return "Det_deformation_gradient"; }
};

using Position = Picasso::Field::LogicalPosition<3>;

} // namespace Example

namespace APIC
{

struct APicTag
{
};

namespace Field
{

struct Velocity : Picasso::Field::Matrix<double, 4, 3>
{
    static std::string label() { return "velocity"; }
};

} // namespace Field
} // namespace APIC

struct FlipTag
{
};

} // namespace Picasso
