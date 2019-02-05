#ifndef HARLOW_PARTICLEFIELDOPS_HPP
#define HARLOW_PARTICLEFIELDOPS_HPP

#include <Harlow_GridBlock.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

namespace Harlow
{
namespace ParticleFieldOps
{
//---------------------------------------------------------------------------//
// Resize a list of particle fields.
//---------------------------------------------------------------------------//
template<class ViewType>
void resizeImpl( const std::size_t n, ViewType& view )
{
    Kokkos::resize( view, n );
}

template<class ViewType, class ... ViewList>
void resizeImpl( const std::size_t n, ViewType& view, ViewList&&... list )
{
    Kokkos::resize( view, n );
    resizeImpl( n, list... );
}

template<class ... ViewList>
void resize( const std::size_t n, ViewList&&... list )
{
    resizeImpl( n, list... );
}

//---------------------------------------------------------------------------//
// Binning Permutation
//---------------------------------------------------------------------------//
// Apply a perumutation vector to a list of particle fields.
template<class BinSortType, class ViewType>
void permuteImpl( const BinSortType& bin_sort, ViewType& view )
{
    bin_sort.sort( view );
}

template<class BinSortType, class ViewType, class ... ViewList>
void permuteImpl(
    const BinSortType& bin_sort, ViewType& view, ViewList&&... list )
{
    bin_sort.sort( view );
    permuteImpl( bin_sort, list... );
}

template<class BinSortType, class ... ViewList>
void permute( const BinSortType& bin_sort, ViewList&&... list )
{
    permuteImpl( bin_sort, list... );
}

//---------------------------------------------------------------------------//
// Cell binning.
// ---------------------------------------------------------------------------//
// Given a grid block and particle coordinates create a permutation structure
// that bins particles by their cell location.
template<class CoordViewType>
Kokkos::BinSort<CoordViewType,Kokkos::BinOp3D<CoordViewType> >
createCellBinning( const GridBlock& block,
                   const CoordViewType& coords )
{
    // Create the binning operator.
    int max_bins[3] = { block.numEntity(MeshEntity::Cell,Dim::I),
                        block.numEntity(MeshEntity::Cell,Dim::J),
                        block.numEntity(MeshEntity::Cell,Dim::K) };
    typename CoordViewType::value_type min[3] =
        { block.lowCorner(Dim::I),
          block.lowCorner(Dim::J),
          block.lowCorner(Dim::K) };
    typename CoordViewType::value_type max[3] =
        { min[Dim::I] + max_bins[Dim::I] * block.cellSize(),
          min[Dim::J] + max_bins[Dim::J] * block.cellSize(),
          min[Dim::K] + max_bins[Dim::K] * block.cellSize() };
    Kokkos::BinOp3D<CoordViewType> bin_op( max_bins, min, max );

    // Create the binning. Do not sort the particles within a cell.
    Kokkos::BinSort<CoordViewType,Kokkos::BinOp3D<CoordViewType> >
        bin_sort( coords, 0, coords.extent(0), bin_op, false );
    bin_sort.create_permute_vector();
    return bin_sort;
}

//---------------------------------------------------------------------------//
// Bin particles by cell and permute them in a single function. The
// coordinates will also be permuted.
template<class CoordViewType, class ... ViewList>
void binByCellAndPermute( const GridBlock& block,
                          CoordViewType& coords,
                          ViewList&&... list )
{
    auto bin_sort = createCellBinning( block, coords );
    permute( bin_sort, coords, list... );
}

//---------------------------------------------------------------------------//
// Key binning.
// ---------------------------------------------------------------------------//
// Given a set of keys create a permutation structure that bins particles by
// their linear key.
template<class KeyViewType>
Kokkos::BinSort<KeyViewType,Kokkos::BinOp1D<KeyViewType> >
createKeyBinning( const KeyViewType& keys,
                  const int bin_size,
                  const typename KeyViewType::value_type key_min,
                  const typename KeyViewType::value_type key_max )
{
    // Create the binning operator.
    int max_bins = (key_max - key_min) / bin_size;
    Kokkos::BinOp1D<KeyViewType> bin_op( max_bins, key_min, key_max );

    // Create the binning. Do not sort the particles within a key bin.
    Kokkos::BinSort<KeyViewType,Kokkos::BinOp1D<KeyViewType> >
        bin_sort( keys, 0, keys.extent(0), bin_op, false );
    bin_sort.create_permute_vector();
    return bin_sort;
}

//---------------------------------------------------------------------------//
// Bin particles by key and permute them in a single function. The
// keys will also be permuted.
template<class KeyViewType, class ... ViewList>
void binByKeyAndPermute( const KeyViewType& keys,
                         const int bin_size,
                         const typename KeyViewType::value_type key_min,
                         const typename KeyViewType::value_type key_max,
                         ViewList&&... list )
{
    auto bin_sort = createKeyBinning( keys, bin_size, key_min, key_max );
    permute( bin_sort, keys, list... );
}

//---------------------------------------------------------------------------//

} // end namespace ParticleFieldOps
} // end namespace Harlow

#endif // end HARLOW_PARTICLEFIELDOPS_HPP
