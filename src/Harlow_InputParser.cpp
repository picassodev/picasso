#include <Harlow_InputParser.hpp>

#include <boost/property_tree/json_parser.hpp>

#include <fstream>
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
        if( 0 == std::strcmp(argv[n],"--harlow-input-file") )
        {
            filename = std::string(argv[n+1]);
            found_arg = true;
            break;
        }
    }

    // Check that we found the filename.
    if ( !found_arg )
        throw std::runtime_error(
            "No Harlow input file specified: --harlow-input-file [file name] is required.");

    // Read the file.
    boost::property_tree::read_json( filename, _ptree );
}

//---------------------------------------------------------------------------//
//! Get the ptree.
const boost::property_tree::ptree& InputParser::propertyTree() const
{
    return _ptree;
}

//---------------------------------------------------------------------------//

} // end namespace Harlow
