#ifndef CAJITA_TYPES_HPP
#define CAJITA_TYPES_HPP

namespace Cajita
{

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

} // end namespace Cajita

#endif // CAJITA_TYPES_HPP
