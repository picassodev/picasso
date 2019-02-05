#ifndef HARLOW_BOVGRIDFIELDWRITER_HPP
#define HARLOW_BOVGRIDFIELDWRITER_HPP

#include <Harlow_MpiTraits.hpp>
#include <Harlow_Types.hpp>
#include <Harlow_GridExecPolicy.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <type_traits>

namespace Harlow
{
namespace BovGridFieldWriter
{
//---------------------------------------------------------------------------//
// VisIt Brick-of-Values (BOV) grid field writer.
//---------------------------------------------------------------------------//
// BOV Format traits.
template<typename T>
struct BovTraits;

template<>
struct BovTraits<short>
{
    static std::string format()
    { return "SHORT"; }
};

template<>
struct BovTraits<int>
{
    static std::string format()
    { return "INT"; }
};

template<>
struct BovTraits<float>
{
    static std::string format()
    { return "FLOAT"; }
};

template<>
struct BovTraits<double>
{
    static std::string format()
    { return "DOUBLE"; }
};

//---------------------------------------------------------------------------//
// Reorder field values for from IJK ordering to JKI ordering.

// rank 0 field
template<class SrcViewType, class DstViewType>
KOKKOS_INLINE_FUNCTION
void reorderFieldValue(
    const int i,
    const int j,
    const int k,
    const DstViewType& src,
    SrcViewType& dst,
    typename std::enable_if<
    3==SrcViewType::traits::dimension::rank,int*>::type = 0 )
{
    dst( k, j, i ) = src( i, j, k );
}

// rank 1 field
template<class SrcViewType, class DstViewType>
KOKKOS_INLINE_FUNCTION
void reorderFieldValue(
    const int i,
    const int j,
    const int k,
    const SrcViewType& src,
    DstViewType& dst,
    typename std::enable_if<
    4==SrcViewType::traits::dimension::rank,int*>::type = 0 )
{
    for ( unsigned d = 0; d < src.extent(3); ++d )
        dst( k, j, i, d ) = src( i, j, k, d );
}

//---------------------------------------------------------------------------//
// Create the MPI subarray for the given field.
template<class GridFieldType>
MPI_Datatype
createSubarray(
    const GridFieldType& field,
    typename std::enable_if<
    3==GridFieldType::view_type::traits::dimension::rank,int*>::type = 0 )
{
    const auto& global_grid = field.globalGrid();
    const auto& block = field.block();

    int local_start[3] = { global_grid.globalOffset(Dim::K),
                           global_grid.globalOffset(Dim::J),
                           global_grid.globalOffset(Dim::I) };
    int local_size[3] = { block.localNumEntity(field.location(),Dim::K),
                          block.localNumEntity(field.location(),Dim::J),
                          block.localNumEntity(field.location(),Dim::I) };
    int global_size[3] = { global_grid.numEntity(field.location(),Dim::K),
                           global_grid.numEntity(field.location(),Dim::J),
                           global_grid.numEntity(field.location(),Dim::I) };

    MPI_Datatype subarray;
    MPI_Type_create_subarray(
        3, global_size, local_size, local_start,
        MPI_ORDER_C,
        MpiTraits<typename GridFieldType::value_type>::type(),
        &subarray );

    return subarray;
}

//---------------------------------------------------------------------------//
// Create the MPI subarray for the given field.
template<class GridFieldType>
MPI_Datatype
createSubarray(
    const GridFieldType& field,
    typename std::enable_if<
    4==GridFieldType::view_type::traits::dimension::rank,int*>::type = 0 )
{
    const auto& global_grid = field.globalGrid();
    const auto& block = field.block();
    int extent = field.data().extent(3);

    int local_start[4] = { global_grid.globalOffset(Dim::K),
                           global_grid.globalOffset(Dim::J),
                           global_grid.globalOffset(Dim::I),
                           0 };
    int local_size[4] = { block.localNumEntity(field.location(),Dim::K),
                          block.localNumEntity(field.location(),Dim::J),
                          block.localNumEntity(field.location(),Dim::I),
                          extent };
    int global_size[4] = { global_grid.numEntity(field.location(),Dim::K),
                           global_grid.numEntity(field.location(),Dim::J),
                           global_grid.numEntity(field.location(),Dim::I),
                           extent };

    MPI_Datatype subarray;
    MPI_Type_create_subarray(
        4, global_size, local_size, local_start,
        MPI_ORDER_C,
        MpiTraits<typename GridFieldType::value_type>::type(),
        &subarray );

    return subarray;
}

//---------------------------------------------------------------------------//
/*!
  \brief Write a grid field to a VisIt BOV.

  This version writes a single output and does not use bricklets. We will do
  this in the future to improve parallel visualization.

  \param time_step_index The index of the time step we are writing.
  \param time The current time
  \param field The field to write
*/
template<class GridFieldType>
void writeTimeStep( const int time_step_index,
                    const double time,
                    const GridFieldType& field )
{
    // Mirror the field to the host and reorder the data in KJI
    // ordering. Our fields are in IJK ordering.
    auto view = field.data();
    using ReorderedView =
        Kokkos::View<typename decltype(view)::data_type,Kokkos::HostSpace>;
    ReorderedView reordered_view(
        Kokkos::ViewAllocateWithoutInitializing("reordered_view"),
        view.extent(2), view.extent(1), view.extent(0) );
    {
        auto view_mirror = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), view );
        Kokkos::parallel_for(
            "bov_write_reorder_view",
            GridExecution::createLocalEntityPolicy<
            typename ReorderedView::execution_space>(
                field.block(), field.location() ),
            KOKKOS_LAMBDA( const int i, const int j, const int k ){
                reorderFieldValue( i, j, k, view_mirror, reordered_view );
            } );
    }

    // Compose a data file name prefix.
    std::stringstream file_name;
    file_name << "grid_" << field.name() << "_"
              << std::setfill('0') << std::setw(6)
              << time_step_index;

    // Open a binary data file.
    std::string data_file_name = file_name.str() + ".dat";
    MPI_File data_file;
    MPI_File_open( field.comm(), data_file_name.c_str(),
                   MPI_MODE_WRONLY | MPI_MODE_CREATE,
                   MPI_INFO_NULL, &data_file );

    // Create the global subarray in which we are writing the local data.
    auto subarray = createSubarray( field );
    MPI_Type_commit( &subarray );

    // Set the data in the file this process is going to write to.
    MPI_File_set_view(
        data_file, 0,
        MpiTraits<typename GridFieldType::value_type>::type(),
        subarray, "native", MPI_INFO_NULL );

    // Write the field to binary.
    MPI_Status status;
    MPI_File_write_all(
        data_file, reordered_view.data(), reordered_view.size(),
        MpiTraits<typename GridFieldType::value_type>::type(),
        &status );

    // Clean up.
    MPI_File_close( &data_file );
    MPI_Type_free( &subarray );

    // Create a VisIt BOV header with global data. Only create the header
    // on rank 0.
    const auto& global_grid = field.globalGrid();
    int rank;
    MPI_Comm_rank( field.comm(), &rank );
    if ( 0 == rank )
    {
        // Open a file for writing.
        std::string header_file_name = file_name.str() + ".bov";
        std::fstream header;
        header.open( header_file_name, std::fstream::out );

        // Write the current time.
        header << "TIME: " << time << std::endl;

        // Data file name.
        header << "DATA_FILE: " << data_file_name << std::endl;

        // Global data size.
        header << "DATA_SIZE: "
               << global_grid.numEntity(field.location(),Dim::I) << " "
               << global_grid.numEntity(field.location(),Dim::J) << " "
               << global_grid.numEntity(field.location(),Dim::K)
               << std::endl;

        // Data format.
        header << "DATA_FORMAT: "
               << BovTraits<typename GridFieldType::value_type>::format()
               << std::endl;

        // Variable name.
        header << "VARIABLE: " << field.name() << std::endl;

        // Endian order
        header << "DATA_ENDIAN: LITTLE" << std::endl;

        // Data location.
        if ( MeshEntity::Cell == field.location() )
            header << "CENTERING: zonal" << std::endl;
        else if ( MeshEntity::Node == field.location() )
            header << "CENTERING: nodal" << std::endl;

        // Mesh low corner.
        header << "BRICK_ORIGIN: "
               << global_grid.lowCorner(Dim::I) << " "
               << global_grid.lowCorner(Dim::J) << " "
               << global_grid.lowCorner(Dim::K) << std::endl;

        // Mesh global width
        header << "BRICK_SIZE: "
               << global_grid.numEntity(MeshEntity::Cell,Dim::I) *
            global_grid.cellSize() << " "
               << global_grid.numEntity(MeshEntity::Cell,Dim::J) *
            global_grid.cellSize() << " "
               << global_grid.numEntity(MeshEntity::Cell,Dim::K) *
            global_grid.cellSize() << std::endl;

        // Number of data components. Scalar and vector types are
        // supported.
        if ( 3 == view.Rank )
            header << "DATA_COMPONENTS: " << 1 << std::endl;
        else if ( 4 == view.Rank )
            header << "DATA_COMPONENTS: " << view.extent(3) << std::endl;

        // Close the header.
        header.close();
    }
}

//---------------------------------------------------------------------------//

} // end namespace BovGridFieldWriter
} // end namespace Harlow

#endif // end HARLOW_BOVGRIDFIELDWRITER_HPP
