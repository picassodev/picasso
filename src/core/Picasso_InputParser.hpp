#ifndef PICASSO_INPUTPARSER_HPP
#define PICASSO_INPUTPARSER_HPP

#include <boost/property_tree/ptree.hpp>

#include <string>

namespace Picasso
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

} // end namespace Picasso

#endif // end PICASSO_INPUTPARSER_HPP
