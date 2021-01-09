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
