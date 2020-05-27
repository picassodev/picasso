#ifndef PICASSO_FLIP_AUXILIARYFIELDS_HPP
#define PICASSO_FLIP_AUXILIARYFIELDS_HPP

#include <Picasso_FieldTypes.hpp>

namespace Picasso
{
namespace FLIP
{
//---------------------------------------------------------------------------//
class VelocityOld : public Vector<double,3>
{
    static std::string label() { return "velocity_old"; }
};

class VelocityTheta : public Vector<double,3>
{
    static std::string label() { return "velocity_theta"; }
};

class VelocityThetaDivergence : public Scalar<double>
{
    static std::string label() { return "velocity_theta_divergence"; }
};

// FIXME: Cajita should allow for assembly of more complex G2P terms from a
// functor instead of a view.
class CompressionTerm : public Scalar<double>
{
    static std::string label() { return "compression_term"; }
};

// FIXME: Cajita should allow for assembly of more complex G2P terms from a
// functor instead of a view.
class AccelerationTheta : public Scalar<double>
{
    static std::string label() { return "AccelerationTheta"; }
};

// FIXME: Cajita should allow for assembly of more complex G2P terms from a
// functor instead of a view.
class AccelerationSquared : public Scalar<double>
{
    static std::string label() { return "AccelerationSquared"; }
};

//---------------------------------------------------------------------------//

} // end namespace FLIP
} // end namespace Picasso

#endif // end PICASSO_FLIP_AUXILIARYFIELDS_HPP
