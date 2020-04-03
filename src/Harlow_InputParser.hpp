#ifndef HARLOW_INPUTPARSER_HPP
#define HARLOW_INPUTPARSER_HPP

#include <boost/property_tree/ptree.hpp>

#include <string>

namespace Harlow
{
//---------------------------------------------------------------------------//
class InputParser
{
  public:

    //! Input argument constructor.
    InputParser( int argc, char* argv[] );

    //! Filename constructor.
    InputParser( const std::string& filename, const std::string& type );

    //! Get the ptree.
    const boost::property_tree::ptree& propertyTree() const;

  private:

    void parse( const std::string& filename, const std::string& type );

  private:

    boost::property_tree::ptree _ptree;
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_INPUTPARSER_HPP
