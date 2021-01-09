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

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace Picasso
{
//---------------------------------------------------------------------------//
//! Input argument constructor.
InputParser::InputParser( int argc, char* argv[] )
{
    // Get the filename from the input.
    bool found_arg = false;
    std::string filename;
    for ( int n = 0; n < argc-1; ++n )
    {
        if( 0 == std::strcmp(argv[n],"--picasso-input-json") )
        {
            filename = std::string(argv[n+1]);
            found_arg = true;
            parse( filename, "json" );
            break;
        }
        else if( 0 == std::strcmp(argv[n],"--picasso-input-xml") )
        {
            filename = std::string(argv[n+1]);
            found_arg = true;
            parse( filename, "xml" );
            break;
        }
    }

    // Check that we found the filename.
    if ( !found_arg )
        throw std::runtime_error(
            "No Picasso input file specified: --picasso-input-*type* [file name] is required.\
             Where *type* can be json or xml" );
}

//---------------------------------------------------------------------------//
//! Filename constructor.
InputParser::InputParser( const std::string& filename,
                          const std::string& type )
{
    parse( filename, type );
}

//---------------------------------------------------------------------------//
//! Get the ptree.
const boost::property_tree::ptree& InputParser::propertyTree() const
{
    return _ptree;
}

//---------------------------------------------------------------------------//
// Parse the field.
void InputParser::parse( const std::string& filename,
                         const std::string& type )
{
    // Get the filename from the input.
    if ( 0 == type.compare("json") )
    {
        boost::property_tree::read_json( filename, _ptree );
    }
    else if ( 0 == type.compare("xml") )
    {
        boost::property_tree::read_xml( filename, _ptree );
    }
    else
    // Check that we found the filename.
    {
        throw std::runtime_error(
            "Only json or xml inputs supported" );
    }
}

//---------------------------------------------------------------------------//

} // end namespace Picasso
