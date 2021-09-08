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

#ifndef PICASSO_MARCHINGCUBES_HPP
#define PICASSO_MARCHINGCUBES_HPP

#include <Picasso_Types.hpp>

#include <Cajita.hpp>

#include <Kokkos_Core.hpp>

#include <mpi.h>

#include <fstream>
#include <string>

namespace Picasso
{
namespace MarchingCubes
{
//---------------------------------------------------------------------------//
struct LookupTable
{
    // Facet counts.
    inline static constexpr short _counts[256] = {
        0, 1, 1, 2, 1, 2, 4, 3, 1, 4, 2, 3, 2, 3, 3, 2, 1, 2, 4, 3, 4, 3, 5, 4,
        6, 5, 5, 4, 5, 4, 4, 3, 1, 4, 2, 3, 6, 5, 5, 4, 4, 5, 3, 4, 5, 4, 4, 3,
        2, 3, 3, 2, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 1, 4, 6, 5, 2, 3, 5, 4,
        4, 5, 5, 4, 3, 4, 4, 3, 2, 3, 5, 4, 3, 2, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2,
        4, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2,
        4, 3, 3, 2, 3, 2, 2, 1, 1, 6, 4, 5, 4, 5, 5, 4, 2, 5, 3, 4, 3, 4, 4, 3,
        4, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2, 2, 5, 3, 4, 5, 4, 4, 3,
        3, 4, 2, 3, 4, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1,
        2, 5, 5, 4, 3, 4, 4, 3, 3, 4, 4, 3, 2, 3, 3, 2, 3, 4, 4, 3, 4, 3, 3, 2,
        4, 3, 3, 2, 3, 2, 2, 1, 3, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1,
        2, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0 };

    // Access counts.
    static KOKKOS_INLINE_FUNCTION short counts( const int i )
    {
        return _counts[i];
    }

    // Facet offsets.
    inline static constexpr short _offsets[256] = {
        0,   0,   1,   2,   4,   5,   7,   11,  14,  15,  19,  21,  24,  26,
        29,  32,  34,  35,  37,  41,  44,  48,  51,  56,  60,  66,  71,  76,
        80,  85,  89,  93,  96,  97,  101, 103, 106, 112, 117, 122, 126, 130,
        135, 138, 142, 147, 151, 155, 158, 160, 163, 166, 168, 173, 177, 181,
        184, 189, 193, 197, 200, 204, 207, 210, 212, 213, 217, 223, 228, 230,
        233, 238, 242, 246, 251, 256, 260, 263, 267, 271, 274, 276, 279, 284,
        288, 291, 293, 297, 300, 305, 309, 313, 316, 320, 323, 326, 328, 332,
        337, 342, 346, 351, 355, 359, 362, 367, 371, 375, 378, 382, 385, 388,
        390, 393, 397, 401, 404, 408, 411, 414, 416, 420, 423, 426, 428, 431,
        433, 435, 436, 437, 443, 447, 452, 456, 461, 466, 470, 472, 477, 480,
        484, 487, 491, 495, 498, 502, 507, 512, 516, 521, 525, 529, 532, 537,
        541, 545, 548, 552, 555, 558, 560, 562, 567, 570, 574, 579, 583, 587,
        590, 593, 597, 599, 602, 606, 609, 612, 614, 617, 621, 625, 628, 632,
        635, 638, 640, 644, 647, 650, 652, 655, 657, 659, 660, 662, 667, 672,
        676, 679, 683, 687, 690, 693, 697, 701, 704, 706, 709, 712, 714, 717,
        721, 725, 728, 732, 735, 738, 740, 744, 747, 750, 752, 755, 757, 759,
        760, 763, 767, 771, 774, 778, 781, 784, 786, 790, 793, 796, 798, 801,
        803, 805, 806, 808, 811, 814, 816, 819, 821, 823, 824, 827, 829, 831,
        832, 834, 835, 836 };

    // Access offsets.
    static KOKKOS_INLINE_FUNCTION short offsets( const int i )
    {
        return _offsets[i];
    }

    // Facet nodes.
    inline static constexpr short _nodes[836][3] = {
        { 4, 0, 8 },   { 0, 5, 9 },   { 4, 5, 9 },   { 4, 9, 8 },
        { 1, 4, 10 },  { 10, 0, 8 },  { 10, 1, 0 },  { 9, 1, 5 },
        { 1, 9, 10 },  { 9, 0, 10 },  { 0, 4, 10 },  { 10, 9, 8 },
        { 1, 9, 10 },  { 9, 1, 5 },   { 5, 1, 11 },  { 11, 5, 0 },
        { 4, 1, 8 },   { 1, 11, 8 },  { 11, 0, 8 },  { 9, 0, 11 },
        { 0, 1, 11 },  { 11, 9, 8 },  { 4, 1, 8 },   { 1, 11, 8 },
        { 11, 4, 10 }, { 4, 11, 5 },  { 11, 5, 0 },  { 10, 11, 8 },
        { 11, 0, 8 },  { 0, 4, 10 },  { 11, 9, 10 }, { 9, 0, 10 },
        { 10, 11, 8 }, { 11, 9, 8 },  { 8, 2, 6 },   { 4, 0, 6 },
        { 0, 2, 6 },   { 0, 6, 8 },   { 0, 5, 6 },   { 6, 5, 2 },
        { 5, 9, 2 },   { 4, 5, 6 },   { 6, 5, 2 },   { 5, 9, 2 },
        { 1, 2, 6 },   { 10, 1, 6 },  { 2, 1, 8 },   { 1, 4, 8 },
        { 1, 2, 6 },   { 10, 1, 6 },  { 2, 1, 0 },   { 10, 1, 6 },
        { 9, 1, 5 },   { 0, 4, 8 },   { 6, 1, 2 },   { 1, 9, 2 },
        { 1, 9, 2 },   { 9, 1, 5 },   { 6, 1, 2 },   { 10, 1, 6 },
        { 11, 2, 6 },  { 5, 1, 8 },   { 11, 5, 2 },  { 1, 11, 6 },
        { 2, 5, 8 },   { 8, 1, 6 },   { 4, 1, 6 },   { 1, 11, 6 },
        { 6, 11, 2 },  { 11, 5, 2 },  { 5, 0, 2 },   { 11, 9, 2 },
        { 11, 2, 6 },  { 1, 11, 6 },  { 0, 1, 8 },   { 1, 6, 8 },
        { 1, 11, 6 },  { 4, 1, 6 },   { 11, 9, 2 },  { 11, 2, 6 },
        { 10, 11, 6 }, { 6, 11, 2 },  { 11, 5, 2 },  { 2, 5, 8 },
        { 5, 4, 8 },   { 11, 5, 2 },  { 5, 0, 2 },   { 11, 2, 6 },
        { 10, 11, 6 }, { 10, 11, 6 }, { 0, 4, 8 },   { 6, 11, 2 },
        { 11, 9, 2 },  { 11, 9, 2 },  { 6, 11, 2 },  { 10, 11, 6 },
        { 9, 7, 2 },   { 4, 7, 2 },   { 4, 0, 7 },   { 0, 9, 7 },
        { 8, 4, 2 },   { 0, 7, 2 },   { 0, 5, 7 },   { 4, 5, 7 },
        { 4, 7, 2 },   { 4, 2, 8 },   { 1, 4, 9 },   { 1, 9, 7 },
        { 1, 7, 10 },  { 10, 7, 2 },  { 4, 10, 2 },  { 9, 4, 2 },
        { 1, 9, 7 },   { 10, 7, 2 },  { 9, 1, 0 },   { 1, 7, 10 },
        { 8, 10, 2 },  { 1, 7, 10 },  { 1, 5, 7 },   { 0, 4, 2 },
        { 4, 10, 2 },  { 10, 7, 2 },  { 1, 5, 7 },   { 1, 7, 10 },
        { 10, 7, 2 },  { 8, 10, 2 },  { 7, 1, 11 },  { 5, 1, 9 },
        { 1, 7, 2 },   { 9, 1, 2 },   { 9, 5, 0 },   { 1, 2, 8 },
        { 7, 1, 11 },  { 1, 7, 2 },   { 4, 1, 8 },   { 1, 11, 7 },
        { 0, 1, 2 },   { 1, 7, 2 },   { 1, 7, 2 },   { 1, 11, 7 },
        { 4, 1, 8 },   { 8, 1, 2 },   { 4, 10, 2 },  { 4, 9, 5 },
        { 10, 7, 2 },  { 11, 7, 10 }, { 9, 4, 2 },   { 11, 7, 10 },
        { 5, 0, 9 },   { 10, 7, 2 },  { 10, 2, 8 },  { 0, 4, 2 },
        { 4, 10, 2 },  { 11, 7, 10 }, { 10, 7, 2 },  { 11, 7, 10 },
        { 10, 7, 2 },  { 10, 2, 8 },  { 8, 9, 6 },   { 9, 7, 6 },
        { 4, 7, 6 },   { 4, 0, 7 },   { 0, 9, 7 },   { 0, 6, 8 },
        { 0, 5, 6 },   { 5, 7, 6 },   { 4, 5, 7 },   { 4, 7, 6 },
        { 9, 4, 8 },   { 1, 4, 9 },   { 1, 9, 7 },   { 10, 7, 6 },
        { 1, 7, 10 },  { 1, 7, 10 },  { 1, 0, 9 },   { 1, 9, 7 },
        { 10, 7, 6 },  { 10, 7, 6 },  { 1, 7, 10 },  { 1, 5, 7 },
        { 0, 4, 8 },   { 1, 5, 7 },   { 1, 7, 10 },  { 10, 7, 6 },
        { 5, 1, 8 },   { 1, 11, 6 },  { 11, 7, 6 },  { 1, 6, 8 },
        { 9, 5, 8 },   { 5, 0, 9 },   { 11, 7, 6 },  { 4, 1, 6 },
        { 1, 11, 6 },  { 1, 11, 6 },  { 11, 7, 6 },  { 0, 1, 8 },
        { 1, 6, 8 },   { 11, 7, 6 },  { 1, 11, 6 },  { 4, 1, 6 },
        { 5, 4, 9 },   { 9, 4, 8 },   { 10, 11, 6 }, { 11, 7, 6 },
        { 5, 0, 9 },   { 10, 11, 6 }, { 11, 7, 6 },  { 0, 4, 8 },
        { 10, 11, 6 }, { 11, 7, 6 },  { 10, 11, 6 }, { 11, 7, 6 },
        { 3, 10, 6 },  { 0, 8, 6 },   { 4, 0, 10 },  { 0, 3, 10 },
        { 3, 0, 6 },   { 0, 5, 10 },  { 9, 0, 6 },   { 5, 9, 3 },
        { 0, 10, 6 },  { 5, 3, 10 },  { 3, 9, 6 },   { 4, 5, 10 },
        { 10, 5, 3 },  { 5, 9, 3 },   { 9, 8, 6 },   { 3, 9, 6 },
        { 3, 1, 6 },   { 1, 4, 6 },   { 6, 0, 8 },   { 3, 1, 0 },
        { 3, 0, 6 },   { 5, 9, 3 },   { 1, 5, 3 },   { 3, 9, 6 },
        { 9, 0, 6 },   { 0, 4, 6 },   { 3, 9, 6 },   { 9, 8, 6 },
        { 1, 5, 3 },   { 5, 9, 3 },   { 11, 5, 3 },  { 3, 5, 6 },
        { 5, 1, 6 },   { 1, 10, 6 },  { 11, 5, 0 },  { 4, 1, 10 },
        { 3, 0, 6 },   { 11, 0, 3 },  { 6, 0, 8 },   { 0, 1, 10 },
        { 11, 9, 3 },  { 9, 0, 6 },   { 0, 10, 6 },  { 3, 9, 6 },
        { 11, 9, 3 },  { 4, 1, 10 },  { 9, 8, 6 },   { 3, 9, 6 },
        { 11, 5, 3 },  { 3, 5, 6 },   { 5, 4, 6 },   { 0, 11, 5 },
        { 11, 0, 3 },  { 0, 8, 6 },   { 3, 0, 6 },   { 9, 0, 6 },
        { 3, 9, 6 },   { 0, 4, 6 },   { 11, 9, 3 },  { 6, 9, 8 },
        { 11, 9, 3 },  { 3, 9, 6 },   { 10, 8, 2 },  { 10, 2, 3 },
        { 0, 2, 3 },   { 4, 0, 10 },  { 0, 3, 10 },  { 10, 5, 3 },
        { 0, 5, 10 },  { 5, 9, 3 },   { 3, 9, 2 },   { 0, 10, 8 },
        { 5, 9, 3 },   { 3, 9, 2 },   { 10, 5, 3 },  { 4, 5, 10 },
        { 1, 2, 3 },   { 2, 1, 8 },   { 1, 4, 8 },   { 1, 0, 3 },
        { 0, 2, 3 },   { 5, 9, 3 },   { 1, 5, 3 },   { 0, 4, 8 },
        { 3, 9, 2 },   { 1, 5, 3 },   { 5, 9, 3 },   { 3, 9, 2 },
        { 11, 5, 2 },  { 3, 11, 2 },  { 1, 10, 8 },  { 5, 1, 8 },
        { 2, 5, 8 },   { 4, 1, 10 },  { 11, 5, 2 },  { 5, 0, 2 },
        { 3, 11, 2 },  { 0, 1, 10 },  { 0, 10, 8 },  { 3, 11, 2 },
        { 11, 9, 2 },  { 4, 1, 10 },  { 3, 11, 2 },  { 11, 9, 2 },
        { 3, 11, 2 },  { 2, 5, 8 },   { 5, 4, 8 },   { 11, 5, 2 },
        { 3, 11, 2 },  { 11, 5, 2 },  { 5, 0, 2 },   { 0, 4, 8 },
        { 3, 11, 2 },  { 11, 9, 2 },  { 3, 11, 2 },  { 11, 9, 2 },
        { 10, 6, 2 },  { 10, 9, 3 },  { 9, 7, 3 },   { 9, 10, 2 },
        { 0, 3, 10 },  { 0, 9, 7 },   { 4, 0, 10 },  { 0, 7, 3 },
        { 2, 8, 6 },   { 0, 5, 10 },  { 0, 10, 6 },  { 10, 5, 3 },
        { 5, 7, 3 },   { 0, 6, 2 },   { 5, 7, 3 },   { 4, 5, 10 },
        { 5, 3, 10 },  { 8, 6, 2 },   { 1, 4, 9 },   { 1, 9, 7 },
        { 1, 7, 3 },   { 9, 4, 2 },   { 4, 6, 2 },   { 1, 7, 3 },
        { 1, 0, 9 },   { 1, 9, 7 },   { 8, 6, 2 },   { 1, 7, 3 },
        { 2, 0, 6 },   { 0, 4, 6 },   { 1, 5, 7 },   { 1, 5, 7 },
        { 1, 7, 3 },   { 8, 6, 2 },   { 5, 1, 9 },   { 1, 10, 6 },
        { 2, 1, 6 },   { 11, 7, 3 },  { 9, 1, 2 },   { 5, 0, 9 },
        { 4, 1, 10 },  { 11, 7, 3 },  { 8, 6, 2 },   { 0, 1, 10 },
        { 11, 7, 3 },  { 0, 10, 6 },  { 0, 6, 2 },   { 4, 1, 10 },
        { 11, 7, 3 },  { 2, 8, 6 },   { 5, 4, 9 },   { 4, 6, 2 },
        { 11, 7, 3 },  { 9, 4, 2 },   { 5, 0, 9 },   { 11, 7, 3 },
        { 2, 8, 6 },   { 11, 7, 3 },  { 0, 6, 2 },   { 0, 4, 6 },
        { 11, 7, 3 },  { 8, 6, 2 },   { 9, 7, 3 },   { 10, 9, 3 },
        { 10, 8, 9 },  { 0, 9, 7 },   { 0, 7, 3 },   { 4, 0, 10 },
        { 0, 3, 10 },  { 0, 5, 10 },  { 10, 5, 3 },  { 5, 7, 3 },
        { 0, 10, 8 },  { 5, 7, 3 },   { 4, 5, 10 },  { 5, 3, 10 },
        { 1, 4, 9 },   { 1, 9, 7 },   { 1, 7, 3 },   { 9, 4, 8 },
        { 1, 9, 7 },   { 9, 1, 0 },   { 1, 7, 3 },   { 1, 5, 7 },
        { 1, 7, 3 },   { 0, 4, 8 },   { 1, 5, 7 },   { 1, 7, 3 },
        { 11, 7, 3 },  { 5, 1, 8 },   { 9, 5, 8 },   { 1, 10, 8 },
        { 9, 5, 0 },   { 4, 1, 10 },  { 11, 7, 3 },  { 0, 1, 10 },
        { 11, 7, 3 },  { 0, 10, 8 },  { 4, 1, 10 },  { 11, 7, 3 },
        { 5, 4, 9 },   { 11, 7, 3 },  { 9, 4, 8 },   { 11, 7, 3 },
        { 5, 0, 9 },   { 11, 7, 3 },  { 0, 4, 8 },   { 11, 7, 3 },
        { 7, 11, 3 },  { 4, 0, 11 },  { 11, 0, 7 },  { 7, 0, 8 },
        { 4, 11, 3 },  { 3, 7, 8 },   { 4, 3, 8 },   { 0, 5, 11 },
        { 0, 11, 3 },  { 9, 0, 7 },   { 7, 0, 3 },   { 4, 5, 11 },
        { 4, 3, 8 },   { 4, 11, 3 },  { 7, 9, 8 },   { 3, 7, 8 },
        { 7, 4, 3 },   { 1, 4, 7 },   { 1, 7, 11 },  { 4, 10, 3 },
        { 11, 0, 7 },  { 11, 1, 0 },  { 3, 7, 8 },   { 10, 3, 8 },
        { 7, 0, 8 },   { 1, 5, 11 },  { 3, 0, 10 },  { 9, 0, 7 },
        { 0, 4, 10 },  { 7, 0, 3 },   { 1, 5, 11 },  { 10, 3, 8 },
        { 7, 9, 8 },   { 3, 7, 8 },   { 5, 1, 7 },   { 7, 1, 3 },
        { 7, 5, 0 },   { 4, 3, 8 },   { 4, 1, 3 },   { 3, 7, 8 },
        { 7, 0, 8 },   { 0, 1, 3 },   { 9, 0, 7 },   { 7, 0, 3 },
        { 4, 3, 8 },   { 4, 1, 3 },   { 3, 7, 8 },   { 7, 9, 8 },
        { 4, 7, 5 },   { 3, 4, 10 },  { 7, 4, 3 },   { 0, 7, 5 },
        { 7, 0, 8 },   { 10, 3, 8 },  { 3, 7, 8 },   { 3, 0, 10 },
        { 0, 4, 10 },  { 9, 0, 7 },   { 7, 0, 3 },   { 3, 7, 8 },
        { 10, 3, 8 },  { 7, 9, 8 },   { 7, 11, 2 },  { 8, 11, 3 },
        { 8, 3, 6 },   { 11, 8, 2 },  { 4, 0, 11 },  { 4, 11, 3 },
        { 11, 0, 7 },  { 7, 0, 2 },   { 4, 3, 6 },   { 0, 6, 8 },
        { 0, 5, 11 },  { 0, 3, 6 },   { 0, 11, 3 },  { 7, 9, 2 },
        { 4, 5, 11 },  { 7, 9, 2 },   { 4, 11, 3 },  { 4, 3, 6 },
        { 1, 7, 11 },  { 10, 3, 6 },  { 1, 4, 8 },   { 7, 1, 2 },
        { 1, 8, 2 },   { 11, 0, 7 },  { 1, 0, 11 },  { 10, 3, 6 },
        { 7, 0, 2 },   { 1, 5, 11 },  { 7, 9, 2 },   { 0, 4, 8 },
        { 10, 3, 6 },  { 10, 3, 6 },  { 1, 5, 11 },  { 7, 9, 2 },
        { 5, 1, 8 },   { 7, 5, 2 },   { 1, 3, 6 },   { 1, 6, 8 },
        { 2, 5, 8 },   { 5, 0, 7 },   { 4, 1, 6 },   { 1, 3, 6 },
        { 7, 0, 2 },   { 0, 1, 8 },   { 7, 9, 2 },   { 8, 1, 6 },
        { 1, 3, 6 },   { 7, 9, 2 },   { 1, 3, 6 },   { 4, 1, 6 },
        { 7, 5, 2 },   { 5, 8, 2 },   { 5, 4, 8 },   { 10, 3, 6 },
        { 7, 5, 0 },   { 10, 3, 6 },  { 7, 0, 2 },   { 7, 9, 2 },
        { 10, 3, 6 },  { 0, 4, 8 },   { 10, 3, 6 },  { 7, 9, 2 },
        { 9, 11, 2 },  { 11, 3, 2 },  { 0, 9, 11 },  { 4, 0, 11 },
        { 4, 11, 3 },  { 4, 3, 8 },   { 3, 2, 8 },   { 0, 5, 11 },
        { 0, 3, 2 },   { 0, 11, 3 },  { 4, 5, 11 },  { 4, 3, 8 },
        { 4, 11, 3 },  { 8, 3, 2 },   { 10, 3, 2 },  { 9, 4, 2 },
        { 1, 4, 9 },   { 1, 9, 11 },  { 4, 10, 2 },  { 8, 10, 2 },
        { 1, 0, 11 },  { 0, 9, 11 },  { 10, 3, 2 },  { 1, 5, 11 },
        { 4, 10, 2 },  { 0, 4, 2 },   { 10, 3, 2 },  { 1, 5, 11 },
        { 10, 3, 2 },  { 8, 10, 2 },  { 5, 1, 9 },   { 9, 1, 2 },
        { 1, 3, 2 },   { 5, 0, 9 },   { 4, 3, 8 },   { 4, 1, 3 },
        { 3, 2, 8 },   { 0, 3, 2 },   { 0, 1, 3 },   { 4, 1, 3 },
        { 3, 2, 8 },   { 4, 3, 8 },   { 5, 4, 9 },   { 10, 3, 2 },
        { 9, 4, 2 },   { 4, 10, 2 },  { 10, 3, 2 },  { 5, 0, 9 },
        { 8, 10, 2 },  { 4, 10, 2 },  { 0, 4, 2 },   { 10, 3, 2 },
        { 10, 3, 2 },  { 8, 10, 2 },  { 8, 3, 6 },   { 11, 8, 9 },
        { 8, 11, 3 },  { 0, 9, 11 },  { 4, 0, 11 },  { 4, 11, 3 },
        { 4, 3, 6 },   { 8, 0, 6 },   { 0, 5, 11 },  { 0, 3, 6 },
        { 0, 11, 3 },  { 4, 11, 3 },  { 4, 3, 6 },   { 4, 5, 11 },
        { 1, 4, 9 },   { 1, 9, 11 },  { 9, 4, 8 },   { 10, 3, 6 },
        { 1, 0, 11 },  { 0, 9, 11 },  { 10, 3, 6 },  { 1, 5, 11 },
        { 10, 3, 6 },  { 0, 4, 8 },   { 1, 5, 11 },  { 10, 3, 6 },
        { 8, 1, 6 },   { 9, 5, 8 },   { 5, 1, 8 },   { 1, 3, 6 },
        { 5, 0, 9 },   { 1, 3, 6 },   { 4, 1, 6 },   { 1, 6, 8 },
        { 0, 1, 8 },   { 1, 3, 6 },   { 4, 1, 6 },   { 1, 3, 6 },
        { 5, 4, 9 },   { 10, 3, 6 },  { 9, 4, 8 },   { 10, 3, 6 },
        { 5, 0, 9 },   { 10, 3, 6 },  { 0, 4, 8 },   { 10, 3, 6 },
        { 7, 11, 6 },  { 11, 10, 6 }, { 4, 0, 11 },  { 4, 11, 10 },
        { 11, 0, 7 },  { 6, 7, 8 },   { 7, 0, 8 },   { 5, 11, 10 },
        { 9, 0, 6 },   { 0, 5, 10 },  { 0, 10, 6 },  { 7, 9, 6 },
        { 4, 5, 11 },  { 4, 11, 10 }, { 7, 9, 6 },   { 6, 9, 8 },
        { 7, 4, 6 },   { 1, 4, 7 },   { 1, 7, 11 },  { 1, 0, 11 },
        { 7, 0, 8 },   { 0, 7, 11 },  { 7, 8, 6 },   { 1, 5, 11 },
        { 9, 0, 6 },   { 0, 4, 6 },   { 7, 9, 6 },   { 1, 5, 11 },
        { 7, 9, 6 },   { 9, 8, 6 },   { 7, 5, 6 },   { 1, 10, 6 },
        { 5, 1, 6 },   { 5, 0, 7 },   { 7, 0, 8 },   { 4, 1, 10 },
        { 6, 7, 8 },   { 9, 0, 6 },   { 7, 9, 6 },   { 0, 1, 10 },
        { 0, 10, 6 },  { 4, 1, 10 },  { 7, 9, 6 },   { 9, 8, 6 },
        { 7, 4, 6 },   { 4, 7, 5 },   { 7, 0, 8 },   { 5, 0, 7 },
        { 6, 7, 8 },   { 9, 0, 6 },   { 0, 4, 6 },   { 7, 9, 6 },
        { 9, 8, 6 },   { 7, 9, 6 },   { 10, 8, 11 }, { 7, 11, 2 },
        { 11, 8, 2 },  { 4, 0, 11 },  { 4, 11, 10 }, { 11, 0, 7 },
        { 7, 0, 2 },   { 0, 5, 10 },  { 5, 11, 10 }, { 7, 9, 2 },
        { 0, 10, 8 },  { 7, 9, 2 },   { 4, 11, 10 }, { 4, 5, 11 },
        { 11, 1, 7 },  { 1, 8, 2 },   { 7, 1, 2 },   { 1, 4, 8 },
        { 11, 0, 7 },  { 11, 1, 0 },  { 7, 0, 2 },   { 1, 5, 11 },
        { 7, 9, 2 },   { 0, 4, 8 },   { 1, 5, 11 },  { 7, 9, 2 },
        { 1, 10, 8 },  { 7, 5, 2 },   { 2, 5, 8 },   { 5, 1, 8 },
        { 7, 5, 0 },   { 4, 1, 10 },  { 7, 0, 2 },   { 0, 10, 8 },
        { 0, 1, 10 },  { 7, 9, 2 },   { 4, 1, 10 },  { 7, 9, 2 },
        { 7, 5, 2 },   { 2, 5, 8 },   { 5, 4, 8 },   { 7, 5, 0 },
        { 7, 0, 2 },   { 0, 4, 8 },   { 7, 9, 2 },   { 7, 9, 2 },
        { 11, 10, 9 }, { 10, 6, 2 },  { 9, 10, 2 },  { 4, 0, 11 },
        { 0, 9, 11 },  { 4, 11, 10 }, { 2, 8, 6 },   { 5, 11, 10 },
        { 0, 5, 10 },  { 0, 6, 2 },   { 0, 10, 6 },  { 4, 11, 10 },
        { 4, 5, 11 },  { 2, 8, 6 },   { 1, 4, 9 },   { 1, 9, 11 },
        { 4, 6, 2 },   { 9, 4, 2 },   { 1, 0, 11 },  { 0, 9, 11 },
        { 2, 8, 6 },   { 1, 5, 11 },  { 0, 4, 6 },   { 0, 6, 2 },
        { 1, 5, 11 },  { 6, 2, 8 },   { 9, 1, 2 },   { 5, 1, 9 },
        { 1, 6, 2 },   { 1, 10, 6 },  { 4, 1, 10 },  { 5, 0, 9 },
        { 2, 8, 6 },   { 0, 1, 10 },  { 0, 6, 2 },   { 0, 10, 6 },
        { 4, 1, 10 },  { 2, 8, 6 },   { 5, 4, 9 },   { 9, 4, 2 },
        { 4, 6, 2 },   { 5, 0, 9 },   { 6, 2, 8 },   { 0, 4, 6 },
        { 2, 0, 6 },   { 6, 2, 8 },   { 10, 8, 11 }, { 11, 8, 9 },
        { 4, 0, 11 },  { 0, 9, 11 },  { 4, 11, 10 }, { 0, 5, 10 },
        { 5, 11, 10 }, { 0, 10, 8 },  { 4, 5, 11 },  { 4, 11, 10 },
        { 1, 4, 9 },   { 1, 9, 11 },  { 9, 4, 8 },   { 1, 0, 11 },
        { 0, 9, 11 },  { 1, 5, 11 },  { 0, 4, 8 },   { 1, 5, 11 },
        { 1, 10, 8 },  { 9, 5, 8 },   { 5, 1, 8 },   { 0, 9, 5 },
        { 4, 1, 10 },  { 0, 1, 10 },  { 0, 10, 8 },  { 4, 1, 10 },
        { 4, 9, 5 },   { 9, 4, 8 },   { 5, 0, 9 },   { 0, 4, 8 } };

    // Access nodes.
    static KOKKOS_INLINE_FUNCTION short nodes( const int i, const int n )
    {
        return _nodes[i][n];
    }
};

//---------------------------------------------------------------------------//
template <class MemorySpace>
struct Data
{
    // Cell facet case ids and offsets. First value is the case id, second is
    // the offset.
    Kokkos::View<int*** [2], MemorySpace> cell_case_ids_and_offsets;

    // Number of facets.
    int num_facet;

    // Facets. Facets are only created for owned cells.
    Kokkos::View<double* [3][3], MemorySpace> facets;

    // Constructor.
    template <class Mesh>
    Data( const Mesh& mesh )
    {
        auto cell_space = mesh.localGrid()->indexSpace(
            Cajita::Ghost{}, Cajita::Cell{}, Cajita::Local{} );

        cell_case_ids_and_offsets =
            Kokkos::View<int*** [2], typename Mesh::memory_space>(
                Kokkos::ViewAllocateWithoutInitializing(
                    "cell_case_ids_and_offset" ),
                cell_space.extent( 0 ), cell_space.extent( 1 ),
                cell_space.extent( 2 ) );
        Kokkos::deep_copy( cell_case_ids_and_offsets, 0 );
        num_facet = 0;
        facets = Kokkos::View<double* [3][3], typename Mesh::memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "facets" ), 0 );
    }
};

//---------------------------------------------------------------------------//
// Data creation function.
template <class Mesh>
Data<typename Mesh::memory_space> createData( const Mesh& mesh )
{
    return Data<typename Mesh::memory_space>( mesh );
}

//---------------------------------------------------------------------------//
namespace Impl
{
//---------------------------------------------------------------------------//
/*
 Compute vertex data. Ordering of cell vertices is:

    1------------3
   /|           /|
  / |          / |
 5------------7  |
 |  |         |  |
 |  |         |  |
 |  0---------|--2
 | /          | /
 |/           |/
 4------------6

in coordinate layout:

    z
    |
    |
    O------y
   /
  /
 x

*/
template <class SignedDistanceView>
KOKKOS_INLINE_FUNCTION void
vertexData( const SignedDistanceView signed_distance, const int i, const int j,
            const int k, Kokkos::Array<double, 8>& vertex_data )
{
    vertex_data[0] = signed_distance( i, j, k, 0 );
    vertex_data[1] = signed_distance( i, j, k + 1, 0 );
    vertex_data[2] = signed_distance( i, j + 1, k, 0 );
    vertex_data[3] = signed_distance( i, j + 1, k + 1, 0 );
    vertex_data[4] = signed_distance( i + 1, j, k, 0 );
    vertex_data[5] = signed_distance( i + 1, j, k + 1, 0 );
    vertex_data[6] = signed_distance( i + 1, j + 1, k, 0 );
    vertex_data[7] = signed_distance( i + 1, j + 1, k + 1, 0 );
}

//---------------------------------------------------------------------------//
// Compute vertex signs.
KOKKOS_INLINE_FUNCTION
void vertexSigns( const Kokkos::Array<double, 8>& vertex_data,
                  Kokkos::Array<int, 8>& vertex_signs )
{
    for ( int i = 0; i < 8; ++i )
        vertex_signs[i] = ( vertex_data[i] <= 0.0 ) ? 1 : 0;
}

//---------------------------------------------------------------------------//
// Compute the case id.
KOKKOS_INLINE_FUNCTION
int caseId( const Kokkos::Array<int, 8>& vertex_signs )
{
    int case_id = 0;
    for ( int i = 0; i < 8; ++i )
        if ( vertex_signs[i] )
            case_id |= 1UL << i;
    return case_id;
}

//---------------------------------------------------------------------------//
// Get vertex locations.
template <class LocalMesh>
KOKKOS_INLINE_FUNCTION void
vertexLocations( const LocalMesh& local_mesh, const int i, const int j,
                 const int k, Kokkos::Array<double, 6>& vertex_locations )
{
    int index[3] = { i, j, k };
    local_mesh.coordinates( Cajita::Node{}, index, vertex_locations.data() );
    for ( int d = 0; d < 3; ++d )
    {
        ++index[d];
    }
    local_mesh.coordinates( Cajita::Node{}, index,
                            vertex_locations.data() + 3 );
    for ( int d = 0; d < 3; ++d )
    {
        vertex_locations[d + 3] -= vertex_locations[d];
    }
}

//---------------------------------------------------------------------------//
// Compute the edges.
KOKKOS_INLINE_FUNCTION
void computeEdges( const Kokkos::Array<double, 8>& vertex_data,
                   const Kokkos::Array<int, 8>& vertex_signs,
                   Kokkos::Array<Kokkos::Array<double, 3>, 12>& edges )
{
    // Calculate edge weights via linear interpolation.
    auto compute_weight = [&]( const int v1, const int v2 ) {
        return ( vertex_signs[v1] != vertex_signs[v2] )
                   ? vertex_data[v1] / ( vertex_data[v1] - vertex_data[v2] )
                   : -1.0;
    };

    // z edges
    edges[0][0] = 0.0;
    edges[0][1] = 0.0;
    edges[0][2] = compute_weight( 0, 1 );

    edges[1][0] = 0.0;
    edges[1][1] = 1.0;
    edges[1][2] = compute_weight( 2, 3 );

    edges[2][0] = 1.0;
    edges[2][1] = 0.0;
    edges[2][2] = compute_weight( 4, 5 );

    edges[3][0] = 1.0;
    edges[3][1] = 1.0;
    edges[3][2] = compute_weight( 6, 7 );

    // y edges
    edges[4][0] = 0.0;
    edges[4][1] = compute_weight( 0, 2 );
    edges[4][2] = 0.0;

    edges[5][0] = 0.0;
    edges[5][1] = compute_weight( 1, 3 );
    edges[5][2] = 1.0;

    edges[6][0] = 1.0;
    edges[6][1] = compute_weight( 4, 6 );
    edges[6][2] = 0.0;

    edges[7][0] = 1.0;
    edges[7][1] = compute_weight( 5, 7 );
    edges[7][2] = 1.0;

    // x edges
    edges[8][0] = compute_weight( 0, 4 );
    edges[8][1] = 0.0;
    edges[8][2] = 0.0;

    edges[9][0] = compute_weight( 1, 5 );
    edges[9][1] = 0.0;
    edges[9][2] = 1.0;

    edges[10][0] = compute_weight( 2, 6 );
    edges[10][1] = 1.0;
    edges[10][2] = 0.0;

    edges[11][0] = compute_weight( 3, 7 );
    edges[11][1] = 1.0;
    edges[11][2] = 1.0;
}

//---------------------------------------------------------------------------//

} // end namespace Impl

//---------------------------------------------------------------------------//
/*!
  \brief Build the marching cubes triangulation over the owned cells. This
  requires the signed distance values to have been gathered.
  \param exec_space The execution space to use for parallel operations.
  \param signed_distance The signed distance array from which to build the
  facets. This array must be on the nodes as this is a cell-based operation.
  \param data The resulting facet data.
  \param reset_data_memory If true, the facet data will always be
  resized/reallocated. If false, reallocation will only occur when the facet
  or cell count has exceeded the existing count limits.
*/
template <class ExecutionSpace, class Mesh, class SignedDistanceArray,
          class MemorySpace>
void build( const ExecutionSpace& exec_space, const Mesh& mesh,
            const SignedDistanceArray& signed_distance, Data<MemorySpace>& data,
            bool reset_data_memory = false )
{
    static_assert( std::is_same<typename SignedDistanceArray::entity_type,
                                Cajita::Node>::value,
                   "Marching cubes facets may only be constructed from nodal "
                   "distance fields" );

    // Get a view of the signed distance data.
    // NOTE: This array must be gathered before calling this function
    // otherwise parallel data dependencies will not be satisfied for boundary
    // cells.
    auto distance_view = signed_distance.view();

    // Get the cell space we are working on.
    auto cell_space = signed_distance.layout()->localGrid()->indexSpace(
        Cajita::Own{}, Cajita::Cell{}, Cajita::Local{} );

    // Get the case id and number of facets for each cell.
    data.num_facet = 0;
    Cajita::grid_parallel_reduce(
        "marching_cubes_facet_count", exec_space, cell_space,
        KOKKOS_LAMBDA( const int i, const int j, const int k, int& result ) {
            Kokkos::Array<double, 8> vertex_data;
            Impl::vertexData( distance_view, i, j, k, vertex_data );
            Kokkos::Array<int, 8> vertex_signs;
            Impl::vertexSigns( vertex_data, vertex_signs );
            int case_id = Impl::caseId( vertex_signs );
            int num_facet = LookupTable::counts( case_id );
            data.cell_case_ids_and_offsets( i, j, k, 0 ) = case_id;
            data.cell_case_ids_and_offsets( i, j, k, 1 ) = num_facet;
            result += num_facet;
        },
        data.num_facet );

    // Allocate facet data if necessary.
    if ( data.num_facet > static_cast<int>( data.facets.extent( 0 ) ) ||
         reset_data_memory )
    {
        Kokkos::realloc( data.facets, data.num_facet );
    }

    // Compute cell offsets into facet data via exclusive scan.
    Kokkos::parallel_scan(
        "marching_cubes_facet_offset",
        Kokkos::RangePolicy<ExecutionSpace>( exec_space, 0, cell_space.size() ),
        KOKKOS_LAMBDA( const int c, int& update, const bool final_pass ) {
            int ek = cell_space.extent( Dim::K );
            int ej = cell_space.extent( Dim::J );
            int ejk = ej * ek;
            int i = c / ejk;
            int j = ( c - i * ejk ) / ek;
            int k = c - i * ejk - j * ek;
            i += cell_space.min( Dim::I );
            j += cell_space.min( Dim::J );
            k += cell_space.min( Dim::K );
            const int value = data.cell_case_ids_and_offsets( i, j, k, 1 );
            if ( final_pass )
            {
                data.cell_case_ids_and_offsets( i, j, k, 1 ) = update;
            }
            update += value;
        } );

    // Fill facets.
    auto local_mesh =
        Cajita::createLocalMesh<MemorySpace>( *( mesh.localGrid() ) );
    Cajita::grid_parallel_for(
        "marching_cubes_fill_facets", exec_space, cell_space,
        KOKKOS_LAMBDA( const int i, const int j, const int k ) {
            int case_id = data.cell_case_ids_and_offsets( i, j, k, 0 );
            int face_offset = data.cell_case_ids_and_offsets( i, j, k, 1 );
            int num_facet = LookupTable::counts( case_id );
            if ( num_facet > 0 )
            {
                Kokkos::Array<double, 8> vertex_data;
                Impl::vertexData( distance_view, i, j, k, vertex_data );
                Kokkos::Array<int, 8> vertex_signs;
                Impl::vertexSigns( vertex_data, vertex_signs );
                Kokkos::Array<double, 6> vertex_locations;
                Impl::vertexLocations( local_mesh, i, j, k, vertex_locations );
                Kokkos::Array<Kokkos::Array<double, 3>, 12> edges;
                Impl::computeEdges( vertex_data, vertex_signs, edges );
                int case_offset = LookupTable::offsets( case_id );
                int edge_id;
                int face_id;
                for ( int f = 0; f < num_facet; ++f )
                {
                    face_id = f + face_offset;
                    for ( int n = 0; n < 3; ++n )
                    {
                        edge_id = LookupTable::nodes( case_offset + f, n );
                        for ( int d = 0; d < 3; ++d )
                        {
                            data.facets( face_id, n, d ) =
                                vertex_locations[d] +
                                vertex_locations[d + 3] * edges[edge_id][d];
                        }
                    }
                }
            }
        } );
}

//---------------------------------------------------------------------------//
// Write facet data to an STL ASCII file for visualization.
template <class MemorySpace>
void writeDataToSTL( const Data<MemorySpace>& data, MPI_Comm comm,
                     const std::string& stl_filename )
{
    // Move facets to the host.
    auto facets =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, data.facets );

    // Create the STL header.
    std::ofstream file( stl_filename );
    if ( !file.is_open() )
        throw std::runtime_error( "Unable to open STL ASCII file" );

    // Make a single surface.
    file << "solid Surface 1" << std::endl;

    // Add each rank's facets to the file. Do a simple parallel file I/O for
    // now.
    int comm_rank;
    MPI_Comm_rank( comm, &comm_rank );
    int comm_size;
    MPI_Comm_size( comm, &comm_size );
    for ( int r = 0; r < comm_size; ++r )
    {
        if ( r == comm_rank )
        {
            int num_facet = facets.extent( 0 );
            for ( int f = 0; f < num_facet; ++f )
            {
                file << "  facet normal 0.000000e+00 0.000000e+00 0.000000e+00"
                     << std::endl;
                file << "    outer loop" << std::endl;
                for ( int n = 0; n < 3; ++n )
                {
                    file << "      vertex " << facets( f, n, 0 ) << " "
                         << facets( f, n, 1 ) << " " << facets( f, n, 2 )
                         << std::endl;
                }
                file << "    endloop" << std::endl;
                file << "  endfacet" << std::endl;
            }
        }
        MPI_Barrier( comm );
    }

    // Finish the surface.
    file << "endsolid Surface 1" << std::endl;

    // Close the file.
    file.close();
}

//---------------------------------------------------------------------------//

} // end namespace MarchingCubes
} // end namespace Picasso

#endif // end PICASSO_MARCHINGCUBES_HPP
