#ifndef PICASSO_LPBF_FIELDTYPES_HPP
#define PICASSO_LPBF_FIELDTYPES_HPP

#include <Picasso_FieldTypes.hpp>

namespace Picasso
{
namespace LPBF
{
//---------------------------------------------------------------------------//
// Auxiliary Fields.
struct UpdatedInternalEnergy : public Field::Scalar<double>
{
    static std::string label() { return "updated_internal_energy"; }
};

//---------------------------------------------------------------------------//

} // end namespace LPBF
} // end namespace Picasso

#endif // end PICASSO_LPBF_FIELDTYPES_HPP
