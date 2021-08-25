/****************************************************************************
 * Copyright (c) 2021 by the Picasso authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Picasso library. Picasso is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef PICASSO_GRIDOPERATOR_HPP
#define PICASSO_GRIDOPERATOR_HPP

#include <Picasso_FieldManager.hpp>
#include <Picasso_FieldTypes.hpp>

#include <Cajita.hpp>

#include <Cabana_Core.hpp>

#include <Kokkos_Core.hpp>

#include <memory>
#include <type_traits>

namespace Picasso
{
//---------------------------------------------------------------------------//
// Grid operator gather dependencies. Defines which grid fields will be
// gathered prior to the evaluation of an operator. The parameter pack
// arguments must be FieldLayout types which give a location and tag to fully
// define the field. Gather dependencies are read-only.
template <class... Layouts>
struct GatherDependencies
{
};

//---------------------------------------------------------------------------//
// Grid operator scatter dependencies. Defines which grid fields will be
// scattered after the evaluation of an operator. The parameter pack arguments
// must be FieldLayout types which give a location and tag to fully define the
// field. Scatter dependencies are write-only and they are set to zero before
// the application of the operator.
template <class... Layouts>
struct ScatterDependencies
{
};

//---------------------------------------------------------------------------//
// Grid operator local dependencies. Defines which grid fields an operator
// will read/write from that require no gather/scatter operations to
// successfully complete the application of the operator (e.g. read/write
// operations are purely local or a gather has already been performed). The
// parameter pack arguments must be FieldLayout types which give a location
// and tag to fully define the field.
template <class... Layouts>
struct LocalDependencies
{
};

//---------------------------------------------------------------------------//
// Grid operator dependency traits.
//
// Allows for the composition of operators with different
// dependencies. Dependencies must appear in the order of gather -> scatter ->
// local. If a dependency type does not exist it does not need to be
// listed. It is possible for an operator to have no dependencies. Viable
// template arguments for the GridOperator are:
//
// GridOperator<Mesh,GatherDependencies,ScatterDependencies,LocalDependencies>
// GridOperator<Mesh,GatherDependencies,ScatterDependencies>
// GridOperator<Mesh,GatherDependencies,LocalDependencies>
// GridOperator<Mesh,ScatterDependencies,LocalDependencies>
// GridOperator<Mesh,GatherDependencies>
// GridOperator<Mesh,ScatterDependencies>
// GridOperator<Mesh,LocalDependencies>
// GridOperator<Mesh>
//
// This is implemented using a template recursion style. By requiring
// dependencies (if they exist) to presented in a given order the user only
// specifies what they need. In the implementation below, the parameter pack
// containing the dependencies is unrolled in the specified order and
// implements the functionality for each dependency type. If no dependencies
// are specified or the dependency template unrolling reaches the end
// (i.e. when a certain type of dependency is not specified) then the empty
// implementation below is generated.
template <class... Dependencies>
struct GridOperatorDependencies;

// No dependencies.
template <>
struct GridOperatorDependencies<>
{
    // Dependency types.
    using gather_dep_type = GatherDependencies<>;
    using scatter_dep_type = ScatterDependencies<>;
    using local_dep_type = LocalDependencies<>;

    // Add gather fields to the field manager.
    template <class FieldManager_t>
    static void addGatherFields( FieldManager_t& )
    {
    }

    // Add scatter fields to the field manager.
    template <class FieldManager_t>
    static void addScatterFields( FieldManager_t& )
    {
    }

    // Add local fields to the field manager.
    template <class FieldManager_t>
    static void addLocalFields( FieldManager_t& )
    {
    }

    // Create a halo for the gather fields.
    template <class FieldManager_t, class MemorySpace>
    static std::shared_ptr<Cajita::Halo<MemorySpace>>
    createGatherHalo( const FieldManager_t&, MemorySpace )
    {
        return nullptr;
    }

    // Create a halo for the scatter fields.
    template <class FieldManager_t, class MemorySpace>
    static std::shared_ptr<Cajita::Halo<MemorySpace>>
    createScatterHalo( const FieldManager_t&, MemorySpace )
    {
        return nullptr;
    }

    // Gather the gather fields.
    template <class Halo, class FieldManager_t, class ExecutionSpace>
    static void gather( const Halo&, const FieldManager_t&,
                        const ExecutionSpace& )
    {
    }

    // Scatter the scatter fields.
    template <class Halo, class FieldManager_t, class ExecutionSpace>
    static void scatter( const Halo&, const FieldManager_t&,
                         const ExecutionSpace& )
    {
    }
};

// Gather dependenices. These come first in the order if they exist.
template <class... Layouts, class... Dependencies>
struct GridOperatorDependencies<GatherDependencies<Layouts...>, Dependencies...>
{
    // Dependency types.
    using gather_dep_type = GatherDependencies<Layouts...>;
    using scatter_dep_type =
        typename GridOperatorDependencies<Dependencies...>::scatter_dep_type;
    using local_dep_type =
        typename GridOperatorDependencies<Dependencies...>::local_dep_type;

    // Add gather fields to the field manager.
    template <class FieldManager_t>
    static void addGatherFields( FieldManager_t& fm )
    {
        std::ignore =
            std::initializer_list<int>{ ( fm.add( Layouts{} ), 0 )... };
    }

    // Add scatter fields to the field manager.
    template <class FieldManager_t>
    static void addScatterFields( FieldManager_t& fm )
    {
        GridOperatorDependencies<Dependencies...>::addScatterFields( fm );
    }

    // Add local fields to the field manager.
    template <class FieldManager_t>
    static void addLocalFields( FieldManager_t& fm )
    {
        GridOperatorDependencies<Dependencies...>::addLocalFields( fm );
    }

    // Create a halo for the gather fields.
    template <class FieldManager_t, class MemorySpace>
    static std::shared_ptr<Cajita::Halo<MemorySpace>>
    createGatherHalo( const FieldManager_t& fm, MemorySpace )
    {
        return Cajita::createHalo(
            Cajita::NodeHaloPattern<FieldManager_t::mesh_type::num_space_dim>(),
            -1, ( *fm.array( Layouts{} ) )... );
    }

    // Create a halo for the scatter fields.
    template <class FieldManager_t, class MemorySpace>
    static std::shared_ptr<Cajita::Halo<MemorySpace>>
    createScatterHalo( const FieldManager_t& fm, MemorySpace space )
    {
        return GridOperatorDependencies<Dependencies...>::createScatterHalo(
            fm, space );
    }

    // Gather the gather fields.
    template <class Halo, class FieldManager_t, class ExecutionSpace>
    static void gather( const Halo& halo, const FieldManager_t& fm,
                        const ExecutionSpace& space )
    {
        halo->gather( space, *( fm.array( Layouts{} ) )... );
    }

    // Scatter the scatter fields.
    template <class Halo, class FieldManager_t, class ExecutionSpace>
    static void scatter( const Halo& halo, const FieldManager_t& fm,
                         const ExecutionSpace& space )
    {
        GridOperatorDependencies<Dependencies...>::scatter( halo, fm, space );
    }
};

// Scatter dependencies. These come second after gather dependencies if they
// exist.
template <class... Layouts, class... Dependencies>
struct GridOperatorDependencies<ScatterDependencies<Layouts...>,
                                Dependencies...>
{
    // Dependency types.
    using gather_dep_type = GatherDependencies<>;
    using scatter_dep_type = ScatterDependencies<Layouts...>;
    using local_dep_type =
        typename GridOperatorDependencies<Dependencies...>::local_dep_type;

    // Add gather fields to the field manager.
    template <class FieldManager_t>
    static void addGatherFields( FieldManager_t& )
    {
    }

    // Add scatter fields to the field manager.
    template <class FieldManager_t>
    static void addScatterFields( FieldManager_t& fm )
    {
        std::ignore =
            std::initializer_list<int>{ ( fm.add( Layouts{} ), 0 )... };
    }

    // Add local fields to the field manager.
    template <class FieldManager_t>
    static void addLocalFields( FieldManager_t& fm )
    {
        GridOperatorDependencies<Dependencies...>::addLocalFields( fm );
    }

    // Create a halo for the gather fields.
    template <class FieldManager_t, class MemorySpace>
    static std::shared_ptr<Cajita::Halo<MemorySpace>>
    createGatherHalo( const FieldManager_t&, MemorySpace )
    {
        return nullptr;
    }

    // Create a halo for the scatter fields.
    template <class FieldManager_t, class MemorySpace>
    static std::shared_ptr<Cajita::Halo<MemorySpace>>
    createScatterHalo( const FieldManager_t& fm, MemorySpace )
    {
        return Cajita::createHalo(
            Cajita::NodeHaloPattern<FieldManager_t::mesh_type::num_space_dim>(),
            -1, ( *fm.array( Layouts{} ) )... );
    }

    // Gather the gather fields.
    template <class Halo, class FieldManager_t, class ExecutionSpace>
    static void gather( const Halo&, const FieldManager_t&,
                        const ExecutionSpace& )
    {
    }

    // Scatter the scatter fields.
    template <class Halo, class FieldManager_t, class ExecutionSpace>
    static void scatter( const Halo& halo, const FieldManager_t& fm,
                         const ExecutionSpace& space )
    {
        halo->scatter( space, Cajita::ScatterReduce::Sum(),
                       *( fm.array( Layouts{} ) )... );
    }
};

// Local dependencies. These come last after scatter dependencies if they
// exist.
template <class... Layouts>
struct GridOperatorDependencies<LocalDependencies<Layouts...>>
{
    // Dependency types.
    using gather_dep_type = GatherDependencies<>;
    using scatter_dep_type = ScatterDependencies<>;
    using local_dep_type = LocalDependencies<Layouts...>;

    // Add gather fields to the field manager.
    template <class FieldManager_t>
    static void addGatherFields( FieldManager_t& )
    {
    }

    // Add scatter fields to the field manager.
    template <class FieldManager_t>
    static void addScatterFields( FieldManager_t& )
    {
    }

    // Add local fields to the field manager.
    template <class FieldManager_t>
    static void addLocalFields( FieldManager_t& fm )
    {
        std::ignore =
            std::initializer_list<int>{ ( fm.add( Layouts{} ), 0 )... };
    }

    // Create a halo for the gather fields.
    template <class FieldManager_t, class MemorySpace>
    static std::shared_ptr<Cajita::Halo<MemorySpace>>
    createGatherHalo( const FieldManager_t&, MemorySpace )
    {
        return nullptr;
    }

    // Create a halo for the scatter fields.
    template <class FieldManager_t, class MemorySpace>
    static std::shared_ptr<Cajita::Halo<MemorySpace>>
    createScatterHalo( const FieldManager_t&, MemorySpace )
    {
        return nullptr;
    }

    // Gather the gather fields.
    template <class Halo, class FieldManager_t, class ExecutionSpace>
    static void gather( const Halo&, const FieldManager_t&,
                        const ExecutionSpace& )
    {
    }

    // Scatter the scatter fields.
    template <class Halo, class FieldManager_t, class ExecutionSpace>
    static void scatter( const Halo&, const FieldManager_t&,
                         const ExecutionSpace& )
    {
    }
};

//---------------------------------------------------------------------------//
// Grid operator.
//
// A grid operator encapsulates all of the field management work associated
// with parallel grid operations. Both particle-centric loops are supported
// (for operations like P2G and G2P) as well as entity centric loops for more
// traditional discrete grid operations.
//
// An operator is defined over a given mesh type and a user defines the fields
// which must be gathered on the grid to apply the operator and the fields
// which will be scattered on the grid after the operation is complete. All
// distributed parallel work is managed by these dependencies - the user needs
// to only write the body of the operator kernel to be applied at each data
// point (e.g. each particle or mesh entity).
//
// The order of the dependency templates (if there are any dependencies) is
// defined above in the documentation for GridOperatorDependencies.
template <class Mesh, class... Dependencies>
class GridOperator
{
  public:
    using mesh_type = Mesh;
    using memory_space = typename mesh_type::memory_space;
    using field_deps = GridOperatorDependencies<Dependencies...>;

    // Constructor.
    GridOperator( const std::shared_ptr<Mesh>& mesh )
        : _mesh( mesh )
    {
    }

    // Setup the operator
    void setup( FieldManager<Mesh>& fm )
    {
        // Add dependencies to the field manager.
        field_deps::addGatherFields( fm );
        field_deps::addScatterFields( fm );
        field_deps::addLocalFields( fm );

        // Create halos. Gather arrays are fused into a single
        // pack/comm. Scatter arrays are also fused into a single pack/comm.
        _gather_halo = field_deps::createGatherHalo( fm, memory_space() );
        _scatter_halo = field_deps::createScatterHalo( fm, memory_space() );
    }

    // Apply the operator in a loop over particles. A work tag specifies the
    // functor instance to use.
    //
    // Functor signature:
    // func( work_tag, local_mesh,
    //       gather_deps, scatter_deps, local_deps, particle_view )
    //
    // The functor is given a ParticleView allowing the kernel to read and
    // write particle data.
    template <class ExecutionSpace, class ParticleList_t, class WorkTag,
              class Func>
    void apply( const std::string label, FieldLocation::Particle,
                const ExecutionSpace& exec_space, const FieldManager<Mesh>& fm,
                const ParticleList_t& pl, const WorkTag&,
                const Func& func ) const
    {
        applyImpl<WorkTag>( label, fm, exec_space, FieldLocation::Particle(),
                            pl, func );
    }

    // Apply the operator in a loop over particles. Functor does not have a
    // work tag.
    //
    // Functor signature:
    // func( local_mesh, gather_deps, scatter_deps, local_deps, particle_view )
    //
    // The functor is given a ParticleView allowing the kernel to read and
    // write particle data.
    template <class ExecutionSpace, class ParticleList_t, class Func>
    void apply( const std::string label, FieldLocation::Particle,
                const ExecutionSpace& exec_space, const FieldManager<Mesh>& fm,
                const ParticleList_t& pl, const Func& func ) const
    {
        applyImpl<void>( label, fm, exec_space, FieldLocation::Particle(), pl,
                         func );
    }

    // Apply the operator in a loop over the owned entities of the given
    // type. A work tag specifies the functor instance to use.
    //
    // Functor signature:
    // func( work_tag, local_mesh,
    //       gather_deps, scatter_deps, local_deps, i, j, k )
    template <class ExecutionSpace, class Location, class WorkTag, class Func>
    void apply( const std::string label, const Location& location,
                const ExecutionSpace& exec_space, const FieldManager<Mesh>& fm,
                const WorkTag&, const Func& func ) const
    {
        applyImpl<WorkTag>( label, fm, exec_space, location, func );
    }

    // Apply the operator in a loop over the owned entities of the given
    // type. Functor does not have a work tag.
    // Functor signature:
    // func( local_mesh, gather_deps, scatter_deps, local_deps, i, j, k )
    template <class ExecutionSpace, class Location, class Func>
    void apply( const std::string label, const Location& location,
                const ExecutionSpace& exec_space, const FieldManager<Mesh>& fm,
                const Func& func ) const
    {
        applyImpl<void>( label, fm, exec_space, location, func );
    }

  public:
    // Manage field dependencies and apply the operator.
    template <class WorkTag, class ExecutionSpace, class... Args>
    void applyImpl( const std::string label, const FieldManager<Mesh>& fm,
                    const ExecutionSpace& exec_space,
                    const Args&... args ) const
    {
        // Gather distributed dependencies.
        field_deps::gather( _gather_halo, fm, exec_space );

        // Create gather dependency data structure for device capture.
        auto gather_deps =
            createDependencies( fm, typename field_deps::gather_dep_type() );

        // Create scatter dependency data structure for device capture.
        auto scatter_deps =
            createDependencies( fm, typename field_deps::scatter_dep_type() );

        // Create local dependency data structure for device capture.
        auto local_deps =
            createDependencies( fm, typename field_deps::local_dep_type() );

        // Create local mesh.
        auto local_mesh =
            Cajita::createLocalMesh<ExecutionSpace>( *( _mesh->localGrid() ) );

        // Apply the operator.
        applyOp<WorkTag>( label, local_mesh, gather_deps, scatter_deps,
                          local_deps, exec_space, args... );

        // Contribute local scatter view results.
        contributeScatterDependencies( fm, scatter_deps );

        // Scatter distributed dependencies.
        field_deps::scatter( _scatter_halo, fm, exec_space );
    }

    // Create parameter pack of gather dependency views. Gather dependencies
    // don't require a scatter in a kernel so we store them as a parameter
    // pack Kokkos::View for on-device access. The resulting views are stored
    // in field view wrappers so the can be accessed in a point-wise fashion
    // as needed.
    template <class... Layouts>
    auto createDependencies( const FieldManager<Mesh>& fm,
                             GatherDependencies<Layouts...> ) const
    {
        // Create a parameter pack of views. The use of (...) here gets a view
        // of each field in the layout list, wraps it for linear algebra
        // operations, and expands it as a parameter pack.
        auto views = Cabana::makeParameterPack(
            Field::createViewWrapper( Layouts{}, fm.view( Layouts{} ) )... );

        // Assign the parameter pack to the dependency fields.
        return createFieldViewTuple<Layouts...>( views );
    }

    // Create a parameter pack of scatter dependency scatter views. Scatter
    // dependencies are write-only in a kernel so we store them as a parameter
    // pack of Kokkos::ScatterView for on-device access.
    template <class... Layouts>
    auto createDependencies( const FieldManager<Mesh>& fm,
                             ScatterDependencies<Layouts...> ) const
    {
        // While we are still using C++14, this expression allows us to reset
        // each scatter field to 0. The (...) in the initializer list is a
        // trick to call the deep_copy function on each view of each field in
        // the layout list. The comma and 0 after the deep copy call is using
        // the comma operator to evaluate the first expression and return the
        // second, hence giving the initializer list something to hold on to
        // while it expands the rest of the Layouts parameter pack. With C++17
        // we would just use fold expressions.
        std::ignore = std::initializer_list<int>{
            ( Kokkos::deep_copy( fm.view( Layouts{} ), 0.0 ), 0 )... };

        // Create a parameter pack of views. The use of (...) here gets a view
        // of each field in the layout list and expands it as a parameter
        // pack.
        auto scatter_views = Cabana::makeParameterPack(
            Kokkos::Experimental::create_scatter_view(
                fm.view( Layouts{} ) )... );

        // Assign the parameter pack to the dependency fields.
        return createFieldViewTuple<Layouts...>( scatter_views );
    }

    // Create a parameter pack of local dependency views. Local dependencies
    // don't require a scatter in a kernel so we store them as a parameter
    // pack Kokkos::View for on-device access. The resulting views are stored
    // in field view wrappers so the can be accessed in a point-wise fashion
    // as needed.
    template <class... Layouts>
    auto createDependencies( const FieldManager<Mesh>& fm,
                             LocalDependencies<Layouts...> ) const
    {
        // Create a parameter pack of views. The use of (...) here gets a view
        // of each field in the layout list, wraps it for linear algebra
        // operations, and expands it as a parameter pack.
        auto views = Cabana::makeParameterPack(
            Field::createViewWrapper( Layouts{}, fm.view( Layouts{} ) )... );

        // Assign the parameter pack to the dependency fields.
        return createFieldViewTuple<Layouts...>( views );
    }

    // Complete local scatter by summing the scatter view results into the main
    // view results.
    template <class Views, class... Layouts>
    void contributeScatterDependencies(
        const FieldManager<Mesh>& fm,
        const FieldViewTuple<Views, Layouts...>& scatter_deps ) const
    {
        // The Kokkos scatter view contribute interface wants a non-const
        // reference to the destination view even though the implementation
        // details of this in Kokkos take a const reference. Hence we need to
        // make a local copy of all views here so we can avoid taking a
        // non-const reference to a temporary. Create a parameter pack of
        // views. The use of (...) here gets a view of each field in the
        // layout list and expands it as a parameter pack.
        auto view_pack = Cabana::makeParameterPack( fm.view( Layouts{} )... );

        // Assign the parameter pack to the dependency fields.
        auto views = createFieldViewTuple<Layouts...>( view_pack );

        // Use the initializer list here to achieve a C++17 fold expression in
        // C++14. Contributes each scatter view into its original view in the
        // field manager
        std::ignore = std::initializer_list<int>{
            ( Kokkos::Experimental::contribute( views.get( Layouts{} ),
                                                scatter_deps.get( Layouts{} ) ),
              0 )... };
    }

    // Call a functor without a work tag.
    template <class WorkTag, class Functor, class... Args>
    KOKKOS_FORCEINLINE_FUNCTION static std::enable_if_t<
        std::is_same<WorkTag, void>::value>
    functorTagDispatch( const Functor& functor, Args&&... args )
    {
        functor( std::forward<Args>( args )... );
    }

    // Call a functor with a work tag
    template <class WorkTag, class Functor, class... Args>
    KOKKOS_FORCEINLINE_FUNCTION static std::enable_if_t<
        !std::is_same<WorkTag, void>::value>
    functorTagDispatch( const Functor& functor, Args&&... args )
    {
        functor( WorkTag{}, std::forward<Args>( args )... );
    }

    // Apply the operator in a particle loop.
    template <class WorkTag, class LocalMesh, class GatherFields,
              class ScatterFields, class LocalFields, class ExecutionSpace,
              class ParticleList_t, class Func>
    void applyOp( const std::string label, const LocalMesh& local_mesh,
                  const GatherFields& gather_deps,
                  const ScatterFields& scatter_deps,
                  const LocalFields& local_deps, const ExecutionSpace&,
                  FieldLocation::Particle, const ParticleList_t& pl,
                  const Func& func ) const
    {
        // Get the particle aosoa.
        const int vector_length = ParticleList_t::aosoa_type::vector_length;
        auto aosoa = pl.aosoa();

        // Apply kernel to each particle. The user functor gets a local mesh
        // for geometric operations, gather, scatter, and local dependencies
        // for field operations (all of which may be empty), and a view of
        // the particle they are currently working on.
        Cabana::SimdPolicy<vector_length, ExecutionSpace> simd_policy(
            0, pl.size() );
        Cabana::simd_parallel_for(
            simd_policy,
            KOKKOS_LAMBDA( const int s, const int a ) {
                typename ParticleList_t::particle_view_type particle(
                    aosoa.access( s ), a );
                functorTagDispatch<WorkTag>( func, local_mesh, gather_deps,
                                             scatter_deps, local_deps,
                                             particle );
            },
            label );
    }

    // Apply the operator in a loop over the owned entities of the given
    // type. 3D specialization.
    template <class WorkTag, class LocalMesh, class GatherFields,
              class ScatterFields, class LocalFields, class ExecutionSpace,
              class Location, class Func>
    std::enable_if_t<3 == LocalMesh::num_space_dim, void>
    applyOp( const std::string label, const LocalMesh& local_mesh,
             const GatherFields& gather_deps, const ScatterFields& scatter_deps,
             const LocalFields& local_deps, const ExecutionSpace& exec_space,
             Location, const Func& func ) const
    {
        // Apply kernel to each entity. The user functor gets a local mesh for
        // geometric operations, gather, scatter, and local dependencies for
        // field operations (all of which may be empty), and the local ijk
        // index of the entity they are currently working on.
        Cajita::grid_parallel_for(
            label, exec_space, *( _mesh->localGrid() ), Cajita::Own(),
            typename Location::entity_type(),
            KOKKOS_LAMBDA( const int i, const int j, const int k ) {
                functorTagDispatch<WorkTag>( func, local_mesh, gather_deps,
                                             scatter_deps, local_deps, i, j,
                                             k );
            } );
    }

    // Apply the operator in a loop over the owned entities of the given
    // type. 2D specialization.
    template <class WorkTag, class LocalMesh, class GatherFields,
              class ScatterFields, class LocalFields, class ExecutionSpace,
              class Location, class Func>
    std::enable_if_t<2 == LocalMesh::num_space_dim, void>
    applyOp( const std::string label, const LocalMesh& local_mesh,
             const GatherFields& gather_deps, const ScatterFields& scatter_deps,
             const LocalFields& local_deps, const ExecutionSpace& exec_space,
             Location, const Func& func ) const
    {
        // Apply kernel to each entity. The user functor gets a local mesh for
        // geometric operations, gather, scatter, and local dependencies for
        // field operations (all of which may be empty), and the local ijk
        // index of the entity they are currently working on.
        auto grid = *( _mesh->localGrid() );
        Cajita::grid_parallel_for(
            label, exec_space, grid, Cajita::Own(),
            typename Location::entity_type(),
            KOKKOS_LAMBDA( const int i, const int j ) {
                functorTagDispatch<WorkTag>( func, local_mesh, gather_deps,
                                             scatter_deps, local_deps, i, j );
            } );
    }

  private:
    std::shared_ptr<Mesh> _mesh;
    std::shared_ptr<Cajita::Halo<memory_space>> _gather_halo;
    std::shared_ptr<Cajita::Halo<memory_space>> _scatter_halo;
};

//---------------------------------------------------------------------------//
// Creation function. Dependencies (if there are any) must be ordered by the
// precedence established in GridOperatorDependencies.
template <class Mesh, class... Dependencies>
auto createGridOperator( const std::shared_ptr<Mesh>& mesh,
                         const Dependencies&... )
{
    return std::make_shared<GridOperator<Mesh, Dependencies...>>( mesh );
}

//---------------------------------------------------------------------------//

} // end namespace Picasso

#endif // end PICASSO_GRIDOPERATOR_HPP
