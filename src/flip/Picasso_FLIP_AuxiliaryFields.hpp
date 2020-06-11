#ifndef PICASSO_FLIP_AUXILIARYFIELDS_HPP
#define PICASSO_FLIP_AUXILIARYFIELDS_HPP

#include <Picasso_FieldTypes.hpp>

namespace Picasso
{
namespace FLIP
{
//---------------------------------------------------------------------------//
struct VelocityOld : public Field::Vector<double,3>
{
    static std::string label() { return "velocity_old"; }
};

struct VelocityTheta : public Field::Vector<double,3>
{
    static std::string label() { return "velocity_theta"; }
};

struct VelocityThetaDivergence : public Field::Scalar<double>
{
    static std::string label() { return "velocity_theta_divergence"; }
};

//---------------------------------------------------------------------------//

} // end namespace FLIP
} // end namespace Picasso

#endif // end PICASSO_FLIP_AUXILIARYFIELDS_HPP
