#ifndef HARLOW_INPUTPARSER_HPP
#define HARLOW_INPUTPARSER_HPP

#include <nlohmann/json.hpp>

namespace Harlow
{
//---------------------------------------------------------------------------//
class InputParser
{
  public:

    //! Input argument constructor.
    InputParser( int argc, char* argv[] );

    //! Get the database.
    const nlohmann::json& database() const;

  private:

    nlohmann::json _json;
};

//---------------------------------------------------------------------------//

} // end namespace Harlow

#endif // end HARLOW_INPUTPARSER_HPP
