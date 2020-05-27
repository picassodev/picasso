#ifndef HARLOW_LPBF_LASERSOURCE_HPP
#define HARLOW_LPBF_LASERSOURCE_HPP

#include <Kokkos_Core.hpp>

#include <boost/property_tree/ptree.hpp>

#include <cmath>

namespace Harlow
{
namespace LPBF
{
//---------------------------------------------------------------------------//
struct LaserSource
{
    double _power;
    double _phi_threshold;

    void setup( const boost::property_tree::ptree& ptree )
    {
        const auto& params = ptree.get_child("laser_source");
        _power = params.get<double>("power");
        _phi_threshold = params.get<double>("level_set_threshold",0.45);
    }

    // double operator()( const double x[3], const double phi, const double
    // time ) const
    double operator()( const double[3], const double phi, const double ) const
    {
        if ( phi > -_phi_threshold )
        {
            return _power;
        }
        else
        {
            return 0.0;
        }
    }
};

//---------------------------------------------------------------------------//

} // end namespace LPBF
} // end namespace Harlow

#endif // end HARLOW_LPBF_LASERSOURCE_HPP
