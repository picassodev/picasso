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

#include <Picasso_UniformMesh.hpp>

namespace Picasso
{
//---------------------------------------------------------------------------//
template class UniformMesh<Kokkos::HostSpace>;

#ifdef KOKKOS_ENABLE_CUDA
template class UniformMesh<Kokkos::CudaSpace>;
#endif

//---------------------------------------------------------------------------//

} // end namespace Picasso
