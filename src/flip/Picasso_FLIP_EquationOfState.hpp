#ifndef PICASSO_FLIP_EQUATIONOFSTATE_HPP
#define PICASSO_FLIP_EQUATIONOFSTATE_HPP

#include <Picasso_FieldTypes.hpp>

#include <Kokkos_Core.hpp>

namespace Picasso
{
namespace FLIP
{
//---------------------------------------------------------------------------//
struct IdealGas
{
    double _gamma_m_1;

    IdealGas() {}

    IdealGas( const double gamma )
        : _gamma_m_1( gamma - 1.0 )
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()( Field::InternalEnergy,
                     const double pressure,
                     const double density ) const
    {
        return pressure / ( (gamma_m_1) * density );
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( Field::Pressure,
                     const double internal_energy,
                     const double density ) const
    {
        return (gamma_m_1) * density * internal_energy;
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( Field::Density,
                     const double pressure,
                     const double internal_energy ) const
    {
        return pressure / ( (gamma_m_1) * internal_energy );
    }
};

//---------------------------------------------------------------------------//

} // end namespace FLIP
} // end namespace Picasso

#endif // end PICASSO_FLIP_EQUATIONOFSTATE_HPP
