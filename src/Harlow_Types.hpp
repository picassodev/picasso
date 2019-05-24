#ifndef HARLOW_TYPES_HPP
#define HARLOW_TYPES_HPP

#include <Cajita_Types.hpp>

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
using Dim = Cajita::Dim;

// Domain boundary
using DomainBoundary = Cajita::DomainBoundary;

// Mesh entity type.
using MeshEntity = Cajita::MeshEntity;

} // end namespace Harlow

#endif // HARLOW_TYPES_HPP
