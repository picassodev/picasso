#ifndef PICASSO_LPBFSOLVER_HPP
#define PICASSO_LPBFSOLVER_HPP

#include <Picasso_LPBF_ProblemManager.hpp>
#include <Picasso_LPBF_TimeIntegrator.hpp>

#include <boost/property_tree/ptree.hpp>

#include <memory>

namespace Picasso
{
namespace LPBF
{
//---------------------------------------------------------------------------//
class SolverBase
{
  public:
    virtual ~SolverBase() = default;
    virtual void solve() = 0;
};

//---------------------------------------------------------------------------//
template<class MemorySpace, class ExecutionSpace>
class Solver : public SolverBase
{
  public:

    // Memory space.
    using memory_space = MemorySpace;

    // Exeuction space.
    using execution_space = ExecutionSpace;

    // Problem manager.
    using problem_manager = ProblemManager<memory_space>;

  public:

    // Constructor.
    Solver( const boost::property_tree::ptree& ptree, MPI_Comm comm )
    {
        // Get the problem parameters.
        const auto& params = ptree.get_child("lpbf");
        _t_final = params.get<double>("final_time");
        _write_freq = params.get<int>("write_frequency");

        // Get the mpi rank
        MPI_Comm_rank( comm, &_rank );

        // Create the problem manager.
        _problem_manager = std::make_shared<problem_manager>(
            execution_space(), ptree, comm );
    }

    // Solve the problem.
    void solve() override
    {
        double time = 0.0;

        // Write initial particle data to file.
        _problem_manager->writeParticleFields( 0, time );

        // Time step
        int num_step = _t_final / _problem_manager->timeStepSize();
        double delta_t = _t_final / num_step;
        for ( int t = 0; t < num_step; ++t )
        {
            // Print if at the write frequency.
            if ( 0 == _rank && 0 == t % _write_freq )
                printf( "Step %d / %d\n", t+1, num_step );

            // Update the particle level set for new particle positions.
            _problem_manager->levelSet()->updateSignedDistance(
                execution_space() );

            // Step forward one time step.
            TimeIntegrator::step( execution_space(), *_problem_manager, time );

            // Communicate particles if needed.
            _problem_manager->communicateParticles( execution_space() );

            // Write particle data to file if at the write frequency.
            if ( 0 == t % _write_freq )
                _problem_manager->writeParticleFields( t+1, time );

            // Update time.
            time += delta_t;

            Cajita::BovWriter::Experimental::writeTimeStep(
                t, time,
                *(_problem_manager->auxiliaryFields()->array(
                      FieldLocation::Node(), Field::SignedDistance())) );
        }
    }

  private:

    double _t_final;
    int _write_freq;
    int _rank;
    std::shared_ptr<problem_manager> _problem_manager;
};

//---------------------------------------------------------------------------//
// Creation method.
std::shared_ptr<SolverBase>
createSolver( const std::string& device,
              const boost::property_tree::ptree& ptree,
              MPI_Comm comm )
{


    if ( 0 == device.compare("serial") )
    {
#ifdef KOKKOS_ENABLE_SERIAL
        return std::make_shared<Solver<Kokkos::HostSpace,Kokkos::Serial>>(
            ptree, comm );
#else
        throw std::runtime_error( "Serial Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare("openmp") )
    {
#ifdef KOKKOS_ENABLE_OPENMP
        return std::make_shared<Solver<Kokkos::HostSpace,Kokkos::OpenMP>>(
            ptree, comm );
#else
        throw std::runtime_error( "OpenMP Backend Not Enabled" );
#endif
    }
    else if ( 0 == device.compare("cuda") )
    {
#ifdef KOKKOS_ENABLE_CUDA
        return std::make_shared<Solver<Kokkos::CudaSpace,Kokkos::Cuda>>(
            ptree, comm );
#else
        throw std::runtime_error( "CUDA Backend Not Enabled" );
#endif
    }
    else
    {
        throw std::runtime_error( "invalid backend" );
        return nullptr;
    }
}

//---------------------------------------------------------------------------//

} // end namespace LPBF
} // end namespace Picasso

#endif // end PICASSO_LPBFSOLVER_HPP
