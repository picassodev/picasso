#include <Harlow_InputParser.hpp>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include <string>

namespace Harlow
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
        if( 0 == std::strcmp(argv[n],"--harlow-input-json") )
        {
            filename = std::string(argv[n+1]);
            found_arg = true;
            boost::property_tree::read_json( filename, _ptree );
            break;
        }
        else if( 0 == std::strcmp(argv[n],"--harlow-input-xml") )
        {
            filename = std::string(argv[n+1]);
            found_arg = true;
            boost::property_tree::read_xml( filename, _ptree );
            break;
        }
    }

    // Check that we found the filename.
    if ( !found_arg )
        throw std::runtime_error(
            "No Harlow input file specified: --harlow-input-*type* [file name] is required.\
             Where *type* can be json or xml" );
}

//---------------------------------------------------------------------------//
//! Get the ptree.
const boost::property_tree::ptree& InputParser::propertyTree() const
{
    return _ptree;
}

//---------------------------------------------------------------------------//

} // end namespace Harlow
