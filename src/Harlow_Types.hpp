#ifndef HARLOW_TYPES_HPP
#define HARLOW_TYPES_HPP

namespace Harlow
{

// Function order. Note that bilinear is ahead of linear - we only use
// bilinear for particle shape function orders while linear can be used for
// both particles and splines.
struct FunctionOrder
{
    enum Values {
        Bilinear = 1,
        Linear = 2,
        Quadratic = 3,
        Cubic = 4
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

// Domain boundary
struct DomainBoundary
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

// Mesh entity type.
struct MeshEntity
{
    enum Values {
        Node = 0,
        Cell = 1,
    };
};

} // end namespace Harlow

#endif // HARLOW_TYPES_HPP
