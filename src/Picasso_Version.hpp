#ifndef PICASSO_VERSION_HPP
#define PICASSO_VERSION_HPP

#include <Picasso_config.hpp>

#include <string>

namespace Picasso
{

std::string version();

std::string git_commit_hash();

} // end namespace Picasso

#endif // end PICASSO_VERSION_HPP
