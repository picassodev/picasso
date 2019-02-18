#ifndef HARLOW_TOLERANCE_HPP
#define HARLOW_TOLERANCE_HPP

#include <Kokkos_Core.hpp>

namespace Harlow
{
//---------------------------------------------------------------------------//
// traits class to handle Tolerance depending on data type
//---------------------------------------------------------------------------//
template<typename T>
struct Tolerance;

template<>
struct Tolerance<float>
{
   static constexpr float tol = 1e-7;
};

template<>
struct Tolerance<double>
{
   static constexpr double tol = 1e-15;
};

} // end namespace Harlow

#endif // end HARLOW_TOLERANCE_HPP
