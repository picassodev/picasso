#include <Harlow_InputParser.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
void testParser( const std::vector<std::string>& args )
{
    const int argc = 4;
    char* argv[argc];
    for ( int n = 0; n < argc; ++n )
        argv[n] = const_cast<char*>(args[n].data());

    Harlow::InputParser parser( argc, argv );
    const auto& pt = parser.propertyTree();

    EXPECT_EQ( pt.get<double>("pi"), 3.141 );
    EXPECT_TRUE( pt.get<bool>("happy") );
    EXPECT_EQ( pt.get<std::string>("name"), "Niels" );
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
TEST( input_parser, json_test )
{
    std::vector<std::string> args =
        { "other_thing", "--harlow-input-json",
          "input_parser_test.json", "something_else" };
    testParser( args );
}

//---------------------------------------------------------------------------//
TEST( input_parser, xml_test )
{
    std::vector<std::string> args =
        { "other_thing", "--harlow-input-xml",
          "input_parser_test.xml", "something_else" };
    testParser( args );
}

//---------------------------------------------------------------------------//
TEST( input_parser, info_test )
{
    std::vector<std::string> args =
        { "other_thing", "--harlow-input-info",
          "input_parser_test.info", "something_else" };
    testParser( args );
}

//---------------------------------------------------------------------------//

} // end namespace Test
