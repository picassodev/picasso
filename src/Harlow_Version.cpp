#include <Harlow_Version.hpp>

namespace Harlow
{

std::string version() { return Harlow_VERSION_STRING; }

std::string git_commit_hash() { return Harlow_GIT_COMMIT_HASH; }

} // end namespace Harlow
