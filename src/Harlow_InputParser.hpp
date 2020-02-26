#ifndef HARLOW_INPUTPARSER_HPP
#define HARLOW_INPUTPARSER_HPP

#include <boost/property_tree/ptree.hpp>

namespace Harlow
{
//---------------------------------------------------------------------------//
class InputParser
{
  public:

    //! Input argument constructor.
    InputParser( int argc, char* argv[] );

    //! Get the ptree.
    const boost::property_tree::ptree& propertyTree() const;

  private:

    boost::property_tree::ptree _ptree;
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_INPUTPARSER_HPP
