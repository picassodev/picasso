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
    const auto& db = parser.database();

    EXPECT_EQ( db["pi"], 3.141 );
    EXPECT_TRUE( db["happy"] );
    EXPECT_EQ(db["name"], "Niels");
    EXPECT_EQ( db["nothing"], nullptr );
    EXPECT_EQ( db["answer"]["everything"], 42 );

    auto vec = db["list"];
    EXPECT_EQ( vec[0], 1 );
    EXPECT_EQ( vec[1], 0 );
    EXPECT_EQ( vec[2], 2 );

    auto obj = db["object"];
    EXPECT_EQ( obj["currency"], "USD" );
    EXPECT_EQ( obj["value"], 42.99 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
