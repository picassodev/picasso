#ifndef HARLOW_TYPES_HPP
#define HARLOW_TYPES_HPP

namespace Harlow
{

// Particle order
enum ParticleOrder
{
    CONSTANT = 0, // PIC
    LINEAR = 1,   // APIC
    BILINEAR = 2  // PolyPIC
};

// Grid spline order
enum SplineOrder
{
    LINEAR = 0,
    QUADRATIC = 1
};

} // end namespace Harlow

#endif // HARLOW_TYPES_HPP
