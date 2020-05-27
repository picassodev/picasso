#include <Picasso_Version.hpp>

namespace Picasso
{

std::string version() { return Picasso_VERSION_STRING; }

std::string git_commit_hash() { return Picasso_GIT_COMMIT_HASH; }

} // end namespace Picasso
