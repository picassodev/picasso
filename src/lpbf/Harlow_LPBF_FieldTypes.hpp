#ifndef HARLOW_LPBF_FIELDTYPES_HPP
#define HARLOW_LPBF_FIELDTYPES_HPP

#include <Harlow_FieldTypes.hpp>

namespace Harlow
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
} // end namespace Harlow

#endif // end HARLOW_LPBF_FIELDTYPES_HPP
