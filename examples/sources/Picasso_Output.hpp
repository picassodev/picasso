#ifndef PICASSO_OUTPUT_HPP
#define PICASSO_OUTPUT_HPP

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Picasso.hpp>

namespace Picasso
{
namespace Output
{

namespace Impl
{

Cabana::Experimental::HDF5ParticleOutput::HDF5Config setupHDF5Config()
{
    Cabana::Experimental::HDF5ParticleOutput::HDF5Config h5_config;
    h5_config.collective = true;
    return h5_config;
}

} // namespace Impl
// Momentum-only output.
template <class ExecutionSpace, class ParticleVelocity, class ParticleList>
void outputParticles( MPI_Comm comm, ExecutionSpace, ParticleVelocity,
                      const int n, const int write_frequency, const double time,
                      const ParticleList particles )
{
    if ( write_frequency > 0 && n % write_frequency == 0 )
    {
        auto h5_config = Impl::setupHDF5Config();

        auto x = particles.slice( Picasso::Example::Position() );
        auto p_p = particles.slice( Picasso::Example::Pressure() );
        auto vel_p = particles.slice( ParticleVelocity() );
        auto m_p = particles.slice( Picasso::Example::Mass() );
        auto v_p = particles.slice( Picasso::Example::Volume() );

        std::string prefix = "particles";
        Cabana::Experimental::HDF5ParticleOutput::writeTimeStep(
            h5_config, prefix, comm, n / write_frequency, time, x.size(), x,
            p_p, vel_p, m_p, v_p );
    }
}

void outputTimestep( MPI_Comm comm, const int n, const int write_frequency,
                     const double total_time, const double dt )
{
    if ( write_frequency > 0 && n % write_frequency == 0 )
    {
        int rank;
        MPI_Comm_rank( comm, &rank );
        if ( rank == 0 )
            std::cout << "Current step: " << n
                      << "; Current time: " << total_time
                      << "; Current dt: " << dt << std::endl;
    }
}

} // namespace Output
} // namespace Picasso

#endif
