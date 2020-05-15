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
    double _fwhm;
    double _power;
    double _t_on;
    double _t_off;
    double _x0;
    double _y0;
    double _ux;
    double _uy;
    double _phi_threshold;

    void setup( const boost::property_tree::ptree& ptree )
    {
        const auto& params = ptree.get_child("laser_source");
        _fwhm = params.get<double>("fwhm");
        _power = params.get<double>("power");
        _t_on = params.get<double>("time_on");
        _t_off = params.get<double>("time_off");
        _x0 = params.get<double>("x0");
        _y0 = params.get<double>("y0");
        _ux = params.get<double>("ux");
        _uy = params.get<double>("uy");
        _phi_threshold = params.get<double>("level_set_threshold");
    }

    double operator()( const x[3], const double phi, const double time )
    {
        // If we aren't in a time in which the laser is on or we are outside
        // the free surface threshold
        if ( fabs(phi) > _phi_threshold ||
             time < _t_on ||
             time > t_off )
        {
            return 0.0;
        }

        double xl = _x0 + (time - t_off) * ux;
        double yl = _y0 + (time - t_off) * uy;

        double r = (x[0]-xl)*(x[0]-xl) + (y[0]-yl)*(y[0]-yl);

        if (
    }
};

//---------------------------------------------------------------------------//

} // end namespace LPBF
} // end namespace Harlow

#endif // end HARLOW_LPBF_LASERSOURCE_HPP
