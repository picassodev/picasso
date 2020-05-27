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

//---------------------------------------------------------------------------//

} // end namespace FLIP
} // end namespace Picasso

#endif // end PICASSO_FLIP_AUXILIARYFIELDS_HPP
