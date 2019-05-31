#ifndef CAJITA_INDEXSPACE_HPP
#define CAJITA_INDEXSPACE_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>

namespace Cajita
{
//---------------------------------------------------------------------------//
/*!
  \class IndexSpace
  \brief Structured index space.
 */
template<long N>
class IndexSpace
{
  public:

    //! Number of dimensions.
    static constexpr long Rank = N;

    /*!
      \brief Size constructor.
    */
    IndexSpace( const std::initializer_list<long>& size )
    {
        std::fill( _min.data(), _min.data() + Rank, 0 );
        std::copy( size.begin(), size.end(), _max.data() );
    }

    /*!
      \brief Range constructor.
    */
    IndexSpace( const std::initializer_list<long>& min,
                const std::initializer_list<long>& max )
    {
        std::copy( min.begin(), min.end(), _min.data() );
        std::copy( max.begin(), max.end(), _max.data() );
    }

    //! Get the minimum index in a given dimension.
    long min( const long dim ) const
    { return _min[dim]; }

    //! Get the minimum indices in all dimensions.
    Kokkos::Array<long,Rank> min() const
    { return _min; }

    //! Get the maximum index in a given dimension.
    long max( const long dim ) const
    { return _max[dim]; }

    //! Get the maximum indices in all dimensions.
    Kokkos::Array<long,Rank> max() const
    { return _max; }

    //! Get the range of a given dimension.
    Kokkos::pair<long,long> range( const long dim ) const
    { return Kokkos::pair<long,long>(_min[dim],_max[dim]); }

    //! Get the number of dimensions.
    long rank() const
    { return Rank; }

    //! Get the extent of a given dimension.
    long extent( const long dim ) const
    { return _max[dim] - _min[dim]; }

  private:

    // Minimum index bounds.
    Kokkos::Array<long,Rank> _min;

    // Maximum index bounds.
    Kokkos::Array<long,Rank> _max;
};

//---------------------------------------------------------------------------//
/*!
  \brief Create a multi-dimensional execution policy over an index space.

  Rank-1 specialization.
*/
template<class IndexSpace_t, class ExecutionSpace>
typename std::enable_if<(1==IndexSpace_t::Rank),
                        Kokkos::RangePolicy<ExecutionSpace>>::type
createExecutionPolicy( const IndexSpace_t& index_space,
                       const ExecutionSpace& )
{
    return Kokkos::RangePolicy<ExecutionSpace>(
        index_space.min(0), index_space.max(0) );
}

//---------------------------------------------------------------------------//
/*!
  \brief Create a multi-dimensional execution policy over an index space.

  Higher-rank specialization.
*/
template<class IndexSpace_t, class ExecutionSpace>
typename std::enable_if<(1<IndexSpace_t::Rank),
    Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<IndexSpace_t::Rank>>
    >::type
createExecutionPolicy( const IndexSpace_t& index_space,
                       const ExecutionSpace& )
{
    return Kokkos::MDRangePolicy<ExecutionSpace,
                                 Kokkos::Rank<IndexSpace_t::Rank> >(
                                     index_space.min(), index_space.max() );
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a view create a subview over the given index space.

  Rank-1 specialization.
*/
template<class ViewType, class IndexSpace_t>
auto createSubview(
    const ViewType& view,
    const IndexSpace_t& index_space,
    typename std::enable_if<(1==IndexSpace_t::Rank),int>::type* = 0 )
    -> decltype( Kokkos::subview(view,
                                 index_space.range(0)) )
{
    static_assert( 1 == ViewType::Rank, "Incorrect view rank" );
    return Kokkos::subview(view,
                           index_space.range(0));
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a view create a subview over the given index space.

  Rank-2 specialization.
*/
template<class ViewType, class IndexSpace_t>
auto createSubview(
    const ViewType& view,
    const IndexSpace_t& index_space,
    typename std::enable_if<(2==IndexSpace_t::Rank),int>::type* = 0 )
    -> decltype( Kokkos::subview(view,
                                 index_space.range(0),
                                 index_space.range(1)) )
{
    static_assert( 2 == ViewType::Rank, "Incorrect view rank" );
    return Kokkos::subview(view,
                           index_space.range(0),
                           index_space.range(1));
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a view create a subview over the given index space.

  Rank-3 specialization.
*/
template<class ViewType, class IndexSpace_t>
auto createSubview(
    const ViewType& view,
    const IndexSpace_t& index_space,
    typename std::enable_if<(3==IndexSpace_t::Rank),int>::type* = 0 )
    -> decltype( Kokkos::subview(view,
                                 index_space.range(0),
                                 index_space.range(1),
                                 index_space.range(2)) )
{
    static_assert( 3 == ViewType::Rank, "Incorrect view rank" );
    return Kokkos::subview(view,
                           index_space.range(0),
                           index_space.range(1),
                           index_space.range(2));
}

//---------------------------------------------------------------------------//
/*!
  \brief Given a view create a subview over the given index space.

  Rank-4 specialization.
*/
template<class ViewType, class IndexSpace_t>
auto createSubview(
    const ViewType& view,
    const IndexSpace_t& index_space,
    typename std::enable_if<(4==IndexSpace_t::Rank),int>::type* = 0 )
    -> decltype( Kokkos::subview(view,
                                 index_space.range(0),
                                 index_space.range(1),
                                 index_space.range(2),
                                 index_space.range(3)) )
{
    static_assert( 4 == ViewType::Rank, "Incorrect view rank" );
    return Kokkos::subview(view,
                           index_space.range(0),
                           index_space.range(1),
                           index_space.range(2),
                           index_space.range(3));
}

//---------------------------------------------------------------------------//

} // end namespace Cajita

#endif // end CAJITA_INDEXSPACE_HPP
