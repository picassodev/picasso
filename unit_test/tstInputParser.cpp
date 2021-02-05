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

#include <Picasso_InputParser.hpp>

#include <gtest/gtest.h>

namespace Test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
void testParser( const Picasso::InputParser& parser )
{
    const auto& pt = parser.propertyTree();

    EXPECT_EQ( pt.get<double>( "pi" ), 3.141 );
    EXPECT_TRUE( pt.get<bool>( "happy" ) );
    EXPECT_EQ( pt.get<std::string>( "name" ), "Niels" );
    EXPECT_EQ( pt.get<int>( "answer.everything" ), 42 );

    int counter = 0;
    for ( auto& element : pt.get_child( "list" ) )
    {
        EXPECT_EQ( element.second.get_value<int>(), counter );
        ++counter;
    }

    EXPECT_EQ( pt.get<std::string>( "object.currency" ), "USD" );
    EXPECT_EQ( pt.get<double>( "object.value" ), 42.99 );
}

//---------------------------------------------------------------------------//
TEST( input_parser, json_cl_test )
{
    std::vector<std::string> args = { "other_thing", "--picasso-input-json",
                                      "input_parser_test.json",
                                      "something_else" };

    const int argc = 4;
    char* argv[argc];
    for ( int n = 0; n < argc; ++n )
        argv[n] = const_cast<char*>( args[n].data() );

    Picasso::InputParser parser( argc, argv );

    testParser( parser );
}

//---------------------------------------------------------------------------//
TEST( input_parser, xml_cl_test )
{
    std::vector<std::string> args = { "other_thing", "--picasso-input-xml",
                                      "input_parser_test.xml",
                                      "something_else" };

    const int argc = 4;
    char* argv[argc];
    for ( int n = 0; n < argc; ++n )
        argv[n] = const_cast<char*>( args[n].data() );

    Picasso::InputParser parser( argc, argv );

    testParser( parser );
}

//---------------------------------------------------------------------------//
TEST( input_parser, json_file_test )
{
    Picasso::InputParser parser( "input_parser_test.json", "json" );
    testParser( parser );
}

//---------------------------------------------------------------------------//
TEST( input_parser, xml_file_test )
{
    Picasso::InputParser parser( "input_parser_test.xml", "xml" );
    testParser( parser );
}

//---------------------------------------------------------------------------//

} // end namespace Test
