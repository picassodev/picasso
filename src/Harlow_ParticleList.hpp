#ifndef HARLOW_PARTICLELIST_HPP
#define HARLOW_PARTICLELIST_HPP

#include <Harlow_ParticleCommunication.hpp>

#include <Cabana_Core.hpp>

#include <memory>

namespace Harlow
{
//---------------------------------------------------------------------------//
template<class MemorySpace, class ... FieldTags>
struct ParticleTraits;

//---------------------------------------------------------------------------//
// 1-Field particle
template<class MemorySpace,
         class Tag0>
struct ParticleTraits<MemorySpace,
                      Tag0>
{
    using memory_space = MemorySpace;

    using member_types = Cabana::MemberTypes<typename Tag0::data_type>;

    using aosoa_type = Cabana::AoSoA<member_types,memory_space>;

    using particle_type = typename aosoa_type::tuple_type;

    template<std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;

    static slice_type<0> slice( const aosoa_type& aosoa, Tag0 )
    {
        return Cabana::slice<0>( aosoa, Tag0::label() );
    }
};

//---------------------------------------------------------------------------//
// 2-field particle.
template<class MemorySpace,
         class Tag0,
         class Tag1>
struct ParticleTraits<MemorySpace,
                      Tag0,
                      Tag1>
{
    using memory_space = MemorySpace;

    using member_types = Cabana::MemberTypes<typename Tag0::data_type,
                                             typename Tag1::data_type>;

    using aosoa_type = Cabana::AoSoA<member_types,memory_space>;

    using particle_type = typename aosoa_type::tuple_type;

    template<std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;

    static slice_type<0> slice( const aosoa_type& aosoa, Tag0 )
    {
        return Cabana::slice<0>( aosoa, Tag0::label() );
    }

    static slice_type<1> slice( const aosoa_type& aosoa, Tag1 )
    {
        return Cabana::slice<1>( aosoa, Tag1::label() );
    }
};

//---------------------------------------------------------------------------//
// 3-field particle.
template<class MemorySpace,
         class Tag0,
         class Tag1,
         class Tag2>
struct ParticleTraits<MemorySpace,
                      Tag0,
                      Tag1,
                      Tag2>
{
    using memory_space = MemorySpace;

    using member_types = Cabana::MemberTypes<typename Tag0::data_type,
                                             typename Tag1::data_type,
                                             typename Tag2::data_type>;

    using aosoa_type = Cabana::AoSoA<member_types,memory_space>;

    using particle_type = typename aosoa_type::tuple_type;

    template<std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;

    static slice_type<0> slice( const aosoa_type& aosoa, Tag0 )
    {
        return Cabana::slice<0>( aosoa, Tag0::label() );
    }

    static slice_type<1> slice( const aosoa_type& aosoa, Tag1 )
    {
        return Cabana::slice<1>( aosoa, Tag1::label() );
    }

    static slice_type<2> slice( const aosoa_type& aosoa, Tag2 )
    {
        return Cabana::slice<2>( aosoa, Tag2::label() );
    }
};

//---------------------------------------------------------------------------//
// 4-field particle.
template<class MemorySpace,
         class Tag0,
         class Tag1,
         class Tag2,
         class Tag3>
struct ParticleTraits<MemorySpace,
                      Tag0,
                      Tag1,
                      Tag2,
                      Tag3>
{
    using memory_space = MemorySpace;

    using member_types = Cabana::MemberTypes<typename Tag0::data_type,
                                             typename Tag1::data_type,
                                             typename Tag2::data_type,
                                             typename Tag3::data_type>;

    using aosoa_type = Cabana::AoSoA<member_types,memory_space>;

    using particle_type = typename aosoa_type::tuple_type;

    template<std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;

    static slice_type<0> slice( const aosoa_type& aosoa, Tag0 )
    {
        return Cabana::slice<0>( aosoa, Tag0::label() );
    }

    static slice_type<1> slice( const aosoa_type& aosoa, Tag1 )
    {
        return Cabana::slice<1>( aosoa, Tag1::label() );
    }

    static slice_type<2> slice( const aosoa_type& aosoa, Tag2 )
    {
        return Cabana::slice<2>( aosoa, Tag2::label() );
    }

    static slice_type<3> slice( const aosoa_type& aosoa, Tag3 )
    {
        return Cabana::slice<3>( aosoa, Tag3::label() );
    }
};

//---------------------------------------------------------------------------//
// 5-field particle.
template<class MemorySpace,
         class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4>
struct ParticleTraits<MemorySpace,
                      Tag0,
                      Tag1,
                      Tag2,
                      Tag3,
                      Tag4>
{
    using memory_space = MemorySpace;

    using member_types = Cabana::MemberTypes<typename Tag0::data_type,
                                             typename Tag1::data_type,
                                             typename Tag2::data_type,
                                             typename Tag3::data_type,
                                             typename Tag4::data_type>;

    using aosoa_type = Cabana::AoSoA<member_types,memory_space>;

    using particle_type = typename aosoa_type::tuple_type;

    template<std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;

    static slice_type<0> slice( const aosoa_type& aosoa, Tag0 )
    {
        return Cabana::slice<0>( aosoa, Tag0::label() );
    }

    static slice_type<1> slice( const aosoa_type& aosoa, Tag1 )
    {
        return Cabana::slice<1>( aosoa, Tag1::label() );
    }

    static slice_type<2> slice( const aosoa_type& aosoa, Tag2 )
    {
        return Cabana::slice<2>( aosoa, Tag2::label() );
    }

    static slice_type<3> slice( const aosoa_type& aosoa, Tag3 )
    {
        return Cabana::slice<3>( aosoa, Tag3::label() );
    }

    static slice_type<4> slice( const aosoa_type& aosoa, Tag4 )
    {
        return Cabana::slice<4>( aosoa, Tag4::label() );
    }
};

//---------------------------------------------------------------------------//
// 6-field particle.
template<class MemorySpace,
         class Tag0,
         class Tag1,
         class Tag2,
         class Tag3,
         class Tag4,
         class Tag5>
struct ParticleTraits<MemorySpace,
                      Tag0,
                      Tag1,
                      Tag2,
                      Tag3,
                      Tag4,
                      Tag5>
{
    using memory_space = MemorySpace;

    using member_types = Cabana::MemberTypes<typename Tag0::data_type,
                                             typename Tag1::data_type,
                                             typename Tag2::data_type,
                                             typename Tag3::data_type,
                                             typename Tag4::data_type,
                                             typename Tag5::data_type>;

    using aosoa_type = Cabana::AoSoA<member_types,memory_space>;

    using particle_type = typename aosoa_type::tuple_type;

    template<std::size_t M>
    using slice_type = typename aosoa_type::template member_slice_type<M>;

    static slice_type<0> slice( const aosoa_type& aosoa, Tag0 )
    {
        return Cabana::slice<0>( aosoa, Tag0::label() );
    }

    static slice_type<1> slice( const aosoa_type& aosoa, Tag1 )
    {
        return Cabana::slice<1>( aosoa, Tag1::label() );
    }

    static slice_type<2> slice( const aosoa_type& aosoa, Tag2 )
    {
        return Cabana::slice<2>( aosoa, Tag2::label() );
    }

    static slice_type<3> slice( const aosoa_type& aosoa, Tag3 )
    {
        return Cabana::slice<3>( aosoa, Tag3::label() );
    }

    static slice_type<4> slice( const aosoa_type& aosoa, Tag4 )
    {
        return Cabana::slice<4>( aosoa, Tag4::label() );
    }

    static slice_type<5> slice( const aosoa_type& aosoa, Tag5 )
    {
        return Cabana::slice<5>( aosoa, Tag5::label() );
    }
};

//---------------------------------------------------------------------------//
template<class Mesh, class ... FieldTags>
class ParticleList
{
  public:

    using mesh_type = Mesh;
    using memory_space = typename Mesh::memory_space;
    using traits = ParticleTraits<memory_space,FieldTags...>;
    using aosoa_type = typename traits::aosoa_type;
    using particle_type = typename traits::particle_type;

    // Default constructor.
    ParticleList( const std::string& label,
                  const std::shared_ptr<Mesh>& mesh)
        : _aosoa(label)
        , _mesh(mesh)
    {}

    // Get the AoSoA
    aosoa_type& aosoa() { return _aosoa; }
    const aosoa_type& aosoa() const { return _aosoa; }

    // Get a slice of a given field.
    template<class FieldTag>
    auto slice( FieldTag tag ) const
        -> decltype(traits::slice(aosoa_type(),FieldTag()))
    {
        return traits::slice(_aosoa,tag);
    }

    // Redistribute particles to new owning domains.
    void redistribute( const int minimum_halo_width )
    {
        // Particles move in logical coordinates in adaptive meshes.
        if ( is_adaptive_mesh<Mesh>::value )
        {
            ParticleCommunication::redistribute(
                _mesh->localGrid(),
                minimum_halo_width,
                this->slice(Field::LogicalPosition()),
                _aosoa );
        }

        // Particles move in physical coordinates in uniform meshes.
        else if ( is_uniform_mesh<Mesh>::value )
        {
            ParticleCommunication::redistribute(
                _mesh->localGrid(),
                minimum_halo_width,
                this->slice(Field::PhysicalPosition()),
                _aosoa );
        }
    }

  private:

    aosoa_type _aosoa;
    std::shared_ptr<Mesh> _mesh;
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_PARTICLELIST_HPP
