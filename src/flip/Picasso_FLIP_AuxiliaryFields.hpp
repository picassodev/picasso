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

// FIXME: Cajita should allow for assembly of more complex G2P terms from a
// functor instead of a view.
struct CompressionTerm : public Field::Scalar<double>
{
    static std::string label() { return "compression_term"; }
};

// FIXME: Cajita should allow for assembly of more complex G2P terms from a
// functor instead of a view.
struct AccelerationTheta : public Field::Scalar<double>
{
    static std::string label() { return "AccelerationTheta"; }
};

// FIXME: Cajita should allow for assembly of more complex G2P terms from a
// functor instead of a view.
struct AccelerationSquared : public Field::Scalar<double>
{
    static std::string label() { return "AccelerationSquared"; }
};

//---------------------------------------------------------------------------//

} // end namespace FLIP
} // end namespace Picasso

#endif // end PICASSO_FLIP_AUXILIARYFIELDS_HPP
