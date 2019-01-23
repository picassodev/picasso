#ifndef HARLOW_TYPES_HPP
#define HARLOW_TYPES_HPP

namespace Harlow
{

// Particle order
struct ParticleOrder
{
    enum Values {
        Constant = 0, // PIC
        Linear = 1,   // APIC
        Bilinear = 2  // PolyPIC
    };
};

// Grid spline order
struct SplineOrder
{
    enum Values {
        Linear = 0,
        Quadratic = 1
    };
};

// Logical dimension index.
struct Dim
{
    enum Values {
        I = 0,
        J = 1,
        K = 2
    };
};

// Physical boundary
struct PhysicalBoundary
{
    enum Values {
        LowX = 0,
        HighX = 1,
        LowY = 2,
        HighY = 3,
        LowZ = 4,
        HighZ = 5
    };
};

// Logical boundary.
struct LogicalBoundary
{
    enum Values {
        Negative = 0,
        Zero = 1,
        Positive = 2,
    };
};

} // end namespace Harlow

#endif // HARLOW_TYPES_HPP
