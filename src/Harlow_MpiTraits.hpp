#ifndef HARLOW_MPITRAITS_HPP
#define HARLOW_MPITRAITS_HPP

#include <mpi.h>

namespace Harlow
{

template<typename T>
struct MpiTraits;

template<>
struct MpiTraits<char>
{
    static MPI_Datatype type() { return MPI_CHAR; }
};

template<>
struct MpiTraits<int>
{
    static MPI_Datatype type() { return MPI_INT; }
};

template<>
struct MpiTraits<long>
{
    static MPI_Datatype type() { return MPI_LONG; }
};

template<>
struct MpiTraits<float>
{
    static MPI_Datatype type() { return MPI_FLOAT; }
};

template<>
struct MpiTraits<double>
{
    static MPI_Datatype type() { return MPI_DOUBLE; }
};

} // end namespace Harlow

#endif // end HARLOW_MPITRAITS_HPP
