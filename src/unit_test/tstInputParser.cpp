#include <Harlow_InputParser.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( input_parser, parser_test )
{
    std::vector<std::string> args =
        { "other_thing", "--harlow-input-file",
          "input_parser_test.json", "something_else" };
    const int argc = 4;
    char* argv[argc];
    for ( int n = 0; n < argc; ++n )
        argv[n] = const_cast<char*>(args[n].data());

    Harlow::InputParser parser( argc, argv );
    const auto& pt = parser.propertyTree();

    EXPECT_EQ( pt.get<double>("pi"), 3.141 );
    EXPECT_TRUE( pt.get<bool>("happy") );
    EXPECT_EQ(pt.get<std::string>("name"), "Niels");
    EXPECT_EQ( pt.get<int>("answer.everything"), 42 );

    int counter = 0;
    for ( auto& element : pt.get_child("list") )
    {
        EXPECT_EQ( element.second.get_value<int>(), counter );
        ++counter;
    }

    EXPECT_EQ( pt.get<std::string>("object.currency"), "USD" );
    EXPECT_EQ( pt.get<double>("object.value"), 42.99 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
