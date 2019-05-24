#ifndef HARLOW_SILOPARTICLEFIELDWRITER_HPP
#define HARLOW_SILOPARTICLEFIELDWRITER_HPP

#include <Cajita_GlobalGrid.hpp>

#include <Kokkos_Core.hpp>

#include <silo.h>

#include <mpi.h>

#include <pmpio.h>

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace Harlow
{
namespace SiloParticleFieldWriter
{
//---------------------------------------------------------------------------//
// Silo Particle Field Writer.
//---------------------------------------------------------------------------//
// BOV Format traits.
template<typename T>
struct SiloTraits;

template<>
struct SiloTraits<short>
{
    static int type()
    { return DB_SHORT; }
};

template<>
struct SiloTraits<int>
{
    static int type()
    { return DB_INT; }
};

template<>
struct SiloTraits<float>
{
    static int type()
    { return DB_FLOAT; }
};

template<>
struct SiloTraits<double>
{
    static int type()
    { return DB_DOUBLE; }
};

//---------------------------------------------------------------------------//
// Rank-0 field
template<class ViewType>
void writeFieldsImpl(
    DBfile* silo_file,
    const std::string& mesh_name,
    const ViewType& view,
    typename std::enable_if<
    1==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    // Mirror the field to the host.
    Kokkos::View<typename ViewType::data_type,
                 Kokkos::HostSpace> host_view( "host_view",
                                               view.extent(0) );
    Kokkos::deep_copy( host_view, view );

    // Write the field.
    DBPutPointvar1( silo_file,
                    view.label().c_str(),
                    mesh_name.c_str(),
                    host_view.data(),
                    view.extent(0),
                    SiloTraits<typename ViewType::value_type>::type(),
                    nullptr );
}

// Rank-1 field
template<class ViewType>
void writeFieldsImpl(
    DBfile* silo_file,
    const std::string& mesh_name,
    const ViewType& view,
    typename std::enable_if<
    2==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    // Mirror the field to the host and reorder in a blocked format.
    Kokkos::View<typename ViewType::data_type,
                 Kokkos::LayoutLeft,
                 Kokkos::HostSpace> host_view( "host_view",
                                               view.extent(0) );
    Kokkos::deep_copy( host_view, view );

    // Get the data pointers.
    std::vector<typename ViewType::value_type*> ptrs( view.extent(1) );
    for ( unsigned d = 0; d < view.extent(1); ++d )
        ptrs[d] = &host_view(0,d);

    // Write the field.
    DBPutPointvar( silo_file,
                   view.label().c_str(),
                   mesh_name.c_str(),
                   view.extent(1),
                   ptrs.data(),
                   view.extent(0),
                   SiloTraits<typename ViewType::value_type>::type(),
                   nullptr );
}

// Rank-2 field
template<class ViewType>
void writeFieldsImpl(
    DBfile* silo_file,
    const std::string& mesh_name,
    const ViewType& view,
    typename std::enable_if<
    3==ViewType::traits::dimension::rank,int*>::type = 0 )
{
    // Mirror the field to the host and reorder in a blocked format.
    Kokkos::View<typename ViewType::data_type,
                 Kokkos::LayoutLeft,
                 Kokkos::HostSpace> host_view( "host_view",
                                               view.extent(0) );
    Kokkos::deep_copy( host_view, view );

    // Get the data pointers.
    std::vector<typename ViewType::value_type*> ptrs;
    ptrs.reserve( view.extent(1) * view.extent(2) );
    for ( unsigned d1 = 0; d1 < view.extent(1); ++d1 )
        for ( unsigned d2 = 0; d2 < view.extent(2); ++d2 )
            ptrs.push_back( &host_view(0,d1,d2) );

    // Write the field.
    DBPutPointvar( silo_file,
                   view.label().c_str(),
                   mesh_name.c_str(),
                   view.extent(1) * view.extent(2), ptrs.data(),
                   view.extent(0),
                   SiloTraits<typename ViewType::value_type>::type(),
                   nullptr );
}

template<class ViewType>
void writeFields( DBfile* silo_file,
                  const std::string& mesh_name,
                  const ViewType& view )
{
    writeFieldsImpl( silo_file, mesh_name, view );
}

template<class ViewType, class ... FieldViewTypes>
void writeFields( DBfile* silo_file,
                  const std::string& mesh_name,
                  const ViewType& view,
                  FieldViewTypes&&... fields )
{
    writeFieldsImpl( silo_file, mesh_name, view );
    writeFields( silo_file, mesh_name, fields... );
}

//---------------------------------------------------------------------------//
// parallel i/o callbacks
void* createFile( const char* file_name, const char* dir_name, void* user_data )
{
    std::ignore = user_data;
    DBfile* silo_file =
        DBCreate( file_name, DB_CLOBBER, DB_LOCAL, nullptr, DB_PDB );
    if ( silo_file )
    {
        DBMkDir( silo_file, dir_name );
        DBSetDir( silo_file, dir_name );
    }

    return (void*) silo_file;
}

void* openFile( const char* file_name, const char* dir_name,
                PMPIO_iomode_t io_mode, void* user_data )
{
    std::ignore = io_mode;
    std::ignore = user_data;
    DBfile* silo_file = DBOpen( file_name, DB_PDB, DB_APPEND );
    if ( silo_file )
    {
        DBMkDir( silo_file, dir_name );
        DBSetDir( silo_file, dir_name );
    }
    return (void*) silo_file;
}

void closeFile( void* file, void* user_data )
{
    std::ignore = user_data;
    DBfile *silo_file = (DBfile *) file;
    if ( silo_file ) DBClose( silo_file );
}

//---------------------------------------------------------------------------//
// Get field names.
template<class ViewType>
void getFieldNamesImpl( std::vector<std::string>& names,
                        const ViewType& view )
{
    names.push_back( view.label() );
}

// Get field names.
template<class ViewType, class ... FieldViewTypes>
void getFieldNamesImpl( std::vector<std::string>& names,
                        const ViewType& view,
                        FieldViewTypes&&... fields )
{
    getFieldNamesImpl( names, view );
    getFieldNamesImpl( names, fields... );
}


template<class ... FieldViewTypes>
std::vector<std::string> getFieldNames( FieldViewTypes&&... fields )
{
    std::vector<std::string> names;
    getFieldNamesImpl( names, fields... );
    return names;
}

//---------------------------------------------------------------------------//
// Write a multimesh hierarchy.
template<class ... FieldViewTypes>
void writeMultiMesh( PMPIO_baton_t* baton,
                     DBfile* silo_file,
                     const int comm_size,
                     const std::string& mesh_name,
                     const int time_step_index,
                     const double time,
                     FieldViewTypes&&... fields )
{
    // Go to the root directory of the file.
    DBSetDir( silo_file, "/" );

    // Create the mesh block names.
    std::vector<std::string> mb_names;
    for ( int r = 0; r < comm_size; ++r )
    {
        int group_rank = PMPIO_GroupRank( baton, r );
        if ( 0 == group_rank )
        {
            std::stringstream bname;
            bname << "rank_" << r << "/" << mesh_name;
            mb_names.push_back(bname.str());
        }
        else
        {
            std::stringstream bname;
            bname << "particles_" << time_step_index << "_group_"
                  << group_rank << ".silo:/rank_" << r << "/" << mesh_name;
            mb_names.push_back(bname.str());
        }
    }
    char** mesh_block_names = (char **) malloc(comm_size * sizeof(char*));
    for ( int r = 0; r < comm_size; ++r )
        mesh_block_names[r] = const_cast<char*>(mb_names[r].c_str());

    std::vector<int> mesh_block_types( comm_size, DB_POINTMESH );

    // Get the names of the fields.
    std::vector<std::string> field_names = getFieldNames( fields... );

    // Create the field block names.
    int num_field = field_names.size();
    std::vector<std::vector<std::string> > fb_names( num_field );
    for ( int f = 0; f < num_field; ++f )
    {
        for ( int r = 0; r < comm_size; ++r )
        {
            int group_rank = PMPIO_GroupRank( baton, r );
            if ( 0 == group_rank )
            {
                std::stringstream bname;
                bname << "rank_" << r << "/" << field_names[f];
                fb_names[f].push_back(bname.str());
            }
            else
            {

                std::stringstream bname;
                bname << "particles_" << time_step_index << "_group_"
                      << group_rank << ".silo:/rank_"
                      << r << "/" << field_names[f];
                fb_names[f].push_back(bname.str());
            }
        }
    }

    std::vector<char**> field_block_names( num_field );
    for ( int f = 0; f < num_field; ++f )
    {
        field_block_names[f] = (char**) malloc( comm_size * sizeof(char*) );
        for ( int r = 0; r < comm_size; ++r )
            field_block_names[f][r] = const_cast<char*>(fb_names[f][r].c_str());
    }

    std::vector<int> field_block_types( comm_size, DB_POINTVAR );

    // Create options.
    DBoptlist* options = DBMakeOptlist( 1 );
    DBAddOption( options, DBOPT_DTIME, (void*) &time );
    DBAddOption( options, DBOPT_CYCLE, (void*) &time_step_index );

    // Add the multiblock mesh.
    std::stringstream mbname;
    mbname << "multi_" << mesh_name;
    DBPutMultimesh( silo_file, mbname.str().c_str(), comm_size,
                    mesh_block_names, mesh_block_types.data(), options );

    // Add the multiblock fields.
    for ( int f = 0; f < num_field; ++f )
    {
        std::stringstream mfname;
        mfname << "multi_" << field_names[f];
        DBPutMultivar( silo_file, mfname.str().c_str(), comm_size,
                       field_block_names[f], field_block_types.data(),
                       options );
    }

    // Cleanup.
    free( mesh_block_names );
    for ( auto& f_name : field_block_names ) free( f_name );
    DBFreeOptlist( options );
}

//---------------------------------------------------------------------------//
// Write a time step.
template<class CoordViewType, class ... FieldViewTypes>
void writeTimeStep( const Cajita::GlobalGrid& global_grid,
                    const int time_step_index,
                    const double time,
                    const CoordViewType& coords,
                    FieldViewTypes&&... fields )
{
    // Pick a number of groups. We want to write approximately the N^3 blocks
    // to N^2 groups. Pick the block dimension with the largest number of
    // ranks as the number of groups. We may want to tweak this as an optional
    // input later with this behavior as the default.
    int num_group = 0;
    for ( int d = 0; d < 3; ++d )
        if ( global_grid.numBlock(d) > num_group )
            num_group = global_grid.numBlock(d);

    // Create the parallel baton.
    int mpi_tag = 1948;
    PMPIO_baton_t* baton =
        PMPIO_Init( num_group, PMPIO_WRITE, global_grid.comm(), mpi_tag,
                    createFile, openFile, closeFile, nullptr );

    // Compose a data file name.
    int comm_rank;
    MPI_Comm_rank( global_grid.comm(), &comm_rank );
    int group_rank = PMPIO_GroupRank( baton, comm_rank );
    std::stringstream file_name;

    // Group 0 writes a master file for the time step.
    if ( 0 == group_rank )
        file_name << "particles_" << time_step_index << ".silo";

    // The other groups write auxiliary files.
    else
        file_name << "particles_" << time_step_index << "_group_"
                  << group_rank << ".silo";

    // Compose a directory name.
    std::stringstream dir_name;
    dir_name << "rank_" << comm_rank;

    // Wait for our turn to write to the file.
    DBfile* silo_file = (DBfile*) PMPIO_WaitForBaton(
        baton, file_name.str().c_str(), dir_name.str().c_str() );

    // Mirror the coordinate field to the host and reorder the coordinates in
    // a blocked format.
    Kokkos::View<typename CoordViewType::data_type,
                 Kokkos::LayoutLeft,
                 Kokkos::HostSpace> host_coords( "host_coords",
                                                 coords.extent(0) );
    Kokkos::deep_copy( host_coords, coords );

    // Add the point mesh.
    std::string mesh_name = "particles";
    double* ptrs[3] =
        {&host_coords(0,0), &host_coords(0,1), &host_coords(0,2)};
    DBPutPointmesh(
        silo_file,
        mesh_name.c_str(),
        host_coords.extent(1),
        ptrs,
        host_coords.extent(0),
        SiloTraits<typename CoordViewType::value_type>::type(),
        nullptr );

    // Add variables.
    writeFields( silo_file, mesh_name, fields... );

    // Root rank writes the global multimesh hierarchy for parallel
    // simulations.
    int comm_size;
    MPI_Comm_size( global_grid.comm(), &comm_size );
    if ( 0 == comm_rank && comm_size > 1 )
        writeMultiMesh( baton, silo_file,
                        comm_size, mesh_name,
                        time_step_index, time, fields... );

    // Hand off the baton.
    PMPIO_HandOffBaton( baton, silo_file );

    // Finish.
    PMPIO_Finish( baton );
}

//---------------------------------------------------------------------------//

} // end namespace SiloParticleFieldWriter
} // end namespace Harlow

#endif // HARLOW_SILOPARTICLEFIELDWRITER_HPP
