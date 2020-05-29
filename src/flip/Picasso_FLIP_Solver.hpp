#ifndef PICASSO_FLIPSOLVER_HPP
#define PICASSO_FLIPSOLVER_HPP

#include <Picasso_FLIP_ProblemManager.hpp>
#include <Picasso_FLIP_TimeIntegrator.hpp>

#include <boost/property_tree/ptree.hpp>

#include <memory>

namespace Picasso
{
namespace FLIP
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
        const auto& params = ptree.get_child("flip");
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
        _problem_manager->writeGridFields( 0, time );

        // Time step
        int num_step = _t_final / _problem_manager->timeStepSize();
        double delta_t = _t_final / num_step;
        bool do_write;
        for ( int t = 0; t < num_step; ++t )
        {
            // Write frequency.
            do_write = 0 == ((t+1) % _write_freq);

            // Print if at the write frequency.
            if ( 0 == _rank && do_write )
                printf( "Step %d / %d\n", t+1, num_step );

            // Step forward one time step.
            TimeIntegrator::step( execution_space(), *_problem_manager );

            // Communicate particles if needed.
            _problem_manager->communicateParticles( execution_space() );

            // Write particle data to file if at the write frequency.
            if ( do_write )
                _problem_manager->writeGridFields( t+1, time );

            // Update time.
            time += delta_t;
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
createSolver( const boost::property_tree::ptree& ptree,
              MPI_Comm comm )
{
    auto device = ptree.get<std::string>("device_type");

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

} // end namespace FLIP
} // end namespace Picasso

#endif // end PICASSO_FLIPSOLVER_HPP
