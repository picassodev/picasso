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

// Mesh face tags.
template<int D>
struct Face;

template<>
struct Face<Dim::I> {};

template<>
struct Face<Dim::J> {};

template<>
struct Face<Dim::K> {};

// Owned decomposition tag.
struct Own {};

// Ghosted decomposition tag.
struct Ghost {};

} // end namespace Cajita

#endif // CAJITA_TYPES_HPP
