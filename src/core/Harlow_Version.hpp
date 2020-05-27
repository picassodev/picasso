#ifndef HARLOW_VERSION_HPP
#define HARLOW_VERSION_HPP

#include <Harlow_config.hpp>

#include <string>

namespace Harlow
{

std::string version();

std::string git_commit_hash();

} // end namespace Harlow

#endif // end HARLOW_VERSION_HPP
