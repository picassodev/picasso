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

// Mesh cell tag.
struct Cell {};

// Mesh node tag.
struct Node {};

} // end namespace Cajita

#endif // CAJITA_TYPES_HPP
