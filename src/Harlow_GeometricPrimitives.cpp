#include <Harlow_GeometricPrimitives.hpp>

namespace Harlow
{
namespace Geometry
{
namespace Primitives
{
//---------------------------------------------------------------------------//
#define HARLOW_INST_GEOMETRIC_PRIMITIVE_HOSTSPACE( PRIMITIVE ) \
    template struct PRIMITIVE<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
#define HARLOW_INST_GEOMETRIC_PRIMITIVE_CUDASPACE( PRIMITIVE )  \
    template struct PRIMITIVE<Kokkos::CudaSpace>;
#else
#define HARLOW_INST_GEOMETRIC_PRIMITIVE_CUDASPACE( PRIMITIVE )
#endif

#define HARLOW_INST_GEOMETRIC_PRIMITIVE( PRIMITIVE ) \
    HARLOW_INST_GEOMETRIC_PRIMITIVE_HOSTSPACE( PRIMITIVE ) \
    HARLOW_INST_GEOMETRIC_PRIMITIVE_CUDASPACE( PRIMITIVE )

HARLOW_INST_GEOMETRIC_PRIMITIVE( ObjectBase )
HARLOW_INST_GEOMETRIC_PRIMITIVE( Object )
HARLOW_INST_GEOMETRIC_PRIMITIVE( Union )
HARLOW_INST_GEOMETRIC_PRIMITIVE( Difference )
HARLOW_INST_GEOMETRIC_PRIMITIVE( Intersection )
HARLOW_INST_GEOMETRIC_PRIMITIVE( Move )
HARLOW_INST_GEOMETRIC_PRIMITIVE( Rotate )
HARLOW_INST_GEOMETRIC_PRIMITIVE( Brick )
HARLOW_INST_GEOMETRIC_PRIMITIVE( Sphere )

//---------------------------------------------------------------------------//

} // end namespace Primitives
} // end namespace Geometry
} // end namespace Harlow
