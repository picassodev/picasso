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

#include <Picasso_APIC.hpp>
#include <Picasso_BatchedLinearAlgebra.hpp>
#include <Picasso_FieldManager.hpp>
#include <Picasso_InputParser.hpp>
#include <Picasso_ParticleInterpolation.hpp>
#include <Picasso_Types.hpp>
#include <Picasso_UniformMesh.hpp>

#include <Cabana_Grid.hpp>

#include <Kokkos_Core.hpp>

#include <cmath>
#include <type_traits>

#include <gtest/gtest.h>

using namespace Picasso;

namespace Test
{
//---------------------------------------------------------------------------//
// Field tags.
struct Foo : Field::Vector<double, 3>
{
    static std::string label() { return "foo"; }
};

struct Bar : Field::Scalar<double>
{
    static std::string label() { return "bar"; }
};

struct Baz : Field::Scalar<double>
{
    static std::string label() { return "baz"; }
};

//---------------------------------------------------------------------------//
// Linear
//---------------------------------------------------------------------------//
// Check the grid velocity. Computed in mathematica.
template <class GridVelocity>
void checkGridVelocity( std::integral_constant<int, 1>, const int cx,
                        const int cy, const int cz, const GridVelocity& gv_host,
                        const double near_eps, const int test_dim,
                        const int array_dim )
{
    double mathematica[8][3] = {
        { -1041.4514891968, -1036.5091579931998, -1029.5668973207999 },
        { -1039.6577396436, -1034.71540844, -1027.7731477676 },
        { -1080.689462254, -1075.7555034336, -1068.821615994 },
        { -1078.8522294412, -1073.9182706208, -1066.9843831812 },
        { -1068.8518602504003, -1063.8168117349, -1056.7818343482002 },
        { -1067.0275793381002, -1061.9925308226, -1054.9575534359 },
        { -1109.1131484693003, -1104.0866148322, -1097.0601531807001 },
        { -1107.2446441701002, -1102.2181105329998, -1095.1916488815 } };

    int n = 0;
    for ( int i = cx; i < cx + 2; ++i )
        for ( int j = cy; j < cy + 2; ++j )
            for ( int k = cz; k < cz + 2; ++k, ++n )
                EXPECT_NEAR( gv_host( i, j, k, array_dim ),
                             mathematica[n][test_dim], near_eps );
}

//---------------------------------------------------------------------------//
// Check particle velocity and affine velocity. Computed in Mathematica.
template <class AffineVelocity>
void checkParticleVelocity( std::integral_constant<int, 1>,
                            const AffineVelocity& pb_host,
                            const double near_eps, const int array_dim )
{
    double mathematica_u[3] = { -1075.3973434395382, -1070.4012502238586,
                                -1063.4052282856794 };

    double mathematica_b[3][3] = {
        { -3.27964099553369, -4.910550811089088, 0.1993230728031321 },
        { -3.2687262820030547, -4.911593173052676, 0.1993230728031179 },
        { -3.257811639242931, -4.91263564023696, 0.1993230728031392 } };

    EXPECT_NEAR( pb_host( 0, array_dim ), mathematica_u[array_dim], near_eps );

    EXPECT_NEAR( pb_host( 1, array_dim ), mathematica_b[array_dim][0],
                 near_eps );
    EXPECT_NEAR( pb_host( 2, array_dim ), mathematica_b[array_dim][1],
                 near_eps );
    EXPECT_NEAR( pb_host( 3, array_dim ), mathematica_b[array_dim][2],
                 near_eps );
}

//---------------------------------------------------------------------------//
// Check the grid momentum. Computed in mathematica.
template <class GridMomentum>
void checkGridMomentum( std::integral_constant<int, 1>, const int cx,
                        const int cy, const int cz, const GridMomentum& gv_host,
                        const double near_eps, const int test_dim,
                        const int array_dim )
{
    double mathematica[8][3] = {
        { -20.18891996702764, -20.09308575061202, -19.95847222978198 },
        { -9.48395134540833, -9.43885289062451, -9.37550535141037 },
        { -16.46997557468053, -16.39480616324114, -16.28916731134517 },
        { -7.737442054021616, -7.702068213344255, -7.652355812452032 },
        { -33.82061098308364, -33.66131859171988, -33.4387547226864 },
        { -15.88830673881129, -15.81334561346365, -15.70860967509495 },
        { -27.56411799293011, -27.43916999833615, -27.26450872108876 },
        { -12.94991936493497, -12.89112030865546, -12.80892676642139 } };

    int n = 0;
    for ( int i = cx; i < cx + 2; ++i )
        for ( int j = cy; j < cy + 2; ++j )
            for ( int k = cz; k < cz + 2; ++k, ++n )
                EXPECT_NEAR( gv_host( i, j, k, array_dim ),
                             mathematica[n][test_dim], near_eps );
}

//---------------------------------------------------------------------------//
// Check the grid mass. Computed in mathematica.
template <class GridMass>
void checkGridMass( std::integral_constant<int, 1>, const int cx, const int cy,
                    const int cz, const GridMass& gm_host,
                    const double near_eps )
{
    double mathematica[8] = { 0.019390335999999897, 0.009124863999999963,
                              0.015235263999999998, 0.0071695360000000085,
                              0.031636863999999966, 0.014887936000000006,
                              0.024857536000000104, 0.011697664000000064 };

    int n = 0;
    for ( int i = cx; i < cx + 2; ++i )
        for ( int j = cy; j < cy + 2; ++j )
            for ( int k = cz; k < cz + 2; ++k, ++n )
                EXPECT_NEAR( gm_host( i, j, k, 0 ), mathematica[n], near_eps );
}

//---------------------------------------------------------------------------//
// Quadratic
//---------------------------------------------------------------------------//
// Check the grid velocity. Computed in mathematica.
template <class GridVelocity>
void checkGridVelocity( std::integral_constant<int, 2>, const int cx,
                        const int cy, const int cz, const GridVelocity& gv_host,
                        const double near_eps, const int test_dim,
                        const int array_dim )
{
    double mathematica[27][3] = {
        { -978.9259866239001, -974.0650036214, -967.2040897101 },
        { -977.2231571579001, -972.3621741554, -965.5012602441 },
        { -975.5019187247001, -970.6409357222, -963.7800218108999 },
        { -1016.2663703435001, -1011.4135198706, -1004.5607393312999 },
        { -1014.5217552050001, -1009.6689047321, -1002.8161241928 },
        { -1012.7582793623001, -1007.9054288894, -1001.0526483501 },
        { -1054.5408038135001, -1049.6961844310001, -1042.8516358245 },
        { -1052.7538965095, -1047.909277127, -1041.0647285204998 },
        { -1050.9476712887001, -1046.1030519062, -1039.2585032997 },
        { -1004.9034686363999, -999.9528653024, -993.0023316500001 },
        { -1003.1714065803999, -998.2208032464, -991.2702695940001 },
        { -1001.4206195291999, -996.4700161952001, -989.5194825428001 },
        { -1043.2260542628, -1038.2837230592, -1031.3414623868 },
        { -1041.4514891968, -1036.5091579931998, -1029.5668973207999 },
        { -1039.6577396436, -1034.71540844, -1027.7731477676 },
        { -1082.507045518, -1077.5730866976, -1070.639199258 },
        { -1080.689462254, -1075.7555034336, -1068.821615994 },
        { -1078.8522294412, -1073.9182706208, -1066.9843831812 },
        { -1031.3347727453001, -1026.2913113106, -1019.2479201478999 },
        { -1029.5732293113, -1024.5297678765999, -1017.4863767138999 },
        { -1027.7926421645002, -1022.7491807298, -1015.7057895670999 },
        { -1070.6566301369, -1065.6215816213999, -1058.5866042347 },
        { -1068.8518602504003, -1063.8168117349, -1056.7818343482002 },
        { -1067.0275793381002, -1061.9925308226, -1054.9575534359 },
        { -1110.9616687653001, -1105.9351351281998, -1098.9086734767 },
        { -1109.1131484693003, -1104.0866148322, -1097.0601531807001 },
        { -1107.2446441701002, -1102.2181105329998, -1095.1916488815 } };

    int n = 0;
    for ( int i = cx - 1; i < cx + 2; ++i )
        for ( int j = cy - 1; j < cy + 2; ++j )
            for ( int k = cz - 1; k < cz + 2; ++k, ++n )
                EXPECT_NEAR( gv_host( i, j, k, array_dim ),
                             mathematica[n][test_dim], near_eps );
}

//---------------------------------------------------------------------------//
// Check particle velocity and affine velocity. Computed in Mathematica.
template <class AffineVelocity>
void checkParticleVelocity( std::integral_constant<int, 2>,
                            const AffineVelocity& pb_host,
                            const double near_eps, const int array_dim )
{
    double mathematica_u[3] = { -1042.881103306653, -1037.9269247222057,
                                -1030.9728166896425 };

    double mathematica_b[3][3] = {
        { -3.3995426204643957, -4.850400281330729, 0.22254272203838787 },
        { -3.388082785485041, -4.8514415741128385, 0.22254272203841385 },
        { -3.3766230251516975, -4.852482973202919, 0.22254272203843906 } };

    EXPECT_NEAR( pb_host( 0, array_dim ), mathematica_u[array_dim], near_eps );

    EXPECT_NEAR( pb_host( 1, array_dim ), mathematica_b[array_dim][0],
                 near_eps );
    EXPECT_NEAR( pb_host( 2, array_dim ), mathematica_b[array_dim][1],
                 near_eps );
    EXPECT_NEAR( pb_host( 3, array_dim ), mathematica_b[array_dim][2],
                 near_eps );
}

//---------------------------------------------------------------------------//
// Check the grid momentum. Computed in mathematica.
template <class GridMomentum>
void checkGridMomentum( std::integral_constant<int, 2>, const int cx,
                        const int cy, const int cz, const GridMomentum& gv_host,
                        const double near_eps, const int test_dim,
                        const int array_dim )
{
    double mathematica[27][3] = {
        { -0.3428079748786512, -0.341103650773257, -0.33869788610259033 },
        { -1.0620714378928868, -1.0567815461193968, -1.04931451875058 },
        { -0.07563939162336615, -0.07526196344777708, -0.07472919895323496 },
        { -1.696620109846103, -1.6885210872709926, -1.677083065873948 },
        { -5.256754361758869, -5.231616565046501, -5.196115162994081 },
        { -0.37440561226440716, -0.37261205709206424, -0.3700790696546563 },
        { -0.2284350697893827, -0.22738651795115858, -0.2259049342955215 },
        { -0.7078217186654041, -0.7045672169391514, -0.6999686683609281 },
        { -0.050416962849643976, -0.050184757598272525, -0.04985665602747398 },
        { -3.589836126576884, -3.572144245733693, -3.547305834062998 },
        { -11.12240976227116, -11.067497592318416, -10.990403975471999 },
        { -0.7921633031728564, -0.7882453780034296, -0.7827448300901959 },
        { -17.74839758410941, -17.66432232037319, -17.54622812588882 },
        { -54.99359213612441, -54.73263880888777, -54.366097312512395 },
        { -3.917028124470889, -3.898409380944528, -3.872257033377054 },
        { -2.387375726599699, -2.3764904519452714, -2.361193292113902 },
        { -7.397761537877467, -7.363975754607495, -7.316496334646461 },
        { -0.5269526992957216, -0.5245421194414538, -0.5211545131120152 },
        { -0.9633565812935954, -0.9586483908050519, -0.9520729316719899 },
        { -2.9849128658218858, -2.9702995548591744, -2.949890603847559 },
        { -0.21260226172431945, -0.2115596174638807, -0.21010346042403294 },
        { -4.758232670664542, -4.735857746840902, -4.7045942469739614 },
        { -14.744072919555455, -14.674625491632252, -14.577589646024476 },
        { -1.0502217089748815, -1.0452667085433485, -1.0383433037285237 },
        { -0.639457299817261, -0.6365603159929738, -0.6325105805675304 },
        { -1.9815664772356267, -1.9725748007983064, -1.9600052067615492 },
        { -0.14115541793672493, -0.14051387134587928, -0.13961704412363574 } };

    int n = 0;
    for ( int i = cx - 1; i < cx + 2; ++i )
        for ( int j = cy - 1; j < cy + 2; ++j )
            for ( int k = cz - 1; k < cz + 2; ++k, ++n )
                EXPECT_NEAR( gv_host( i, j, k, array_dim ),
                             mathematica[n][test_dim], near_eps );
}

//---------------------------------------------------------------------------//
// Check the grid mass. Computed in mathematica.
template <class GridMass>
void checkGridMass( std::integral_constant<int, 2>, const int cx, const int cy,
                    const int cz, const GridMass& gm_host,
                    const double near_eps )
{
    double mathematica[27] = {
        0.0003507323975679963,  0.0010886054000639896, 0.00007767084236799938,
        0.0016695577904639904,  0.005181983868671976,  0.0003697290608639989,
        0.00021652357196799997, 0.0006720472112640006, 0.000047949856768000124,
        0.0035733899120639796,  0.011091109865471948,  0.0007913389424639975,
        0.017010065244671985,   0.05279594645145601,   0.003766934863872007,
        0.0022060213232640105,  0.00684706272307204,   0.0004885306736640036,
        0.000933667130367998,   0.002897921854463997,  0.0002067636551680001,
        0.004444446084864011,   0.01379469943987205,   0.0009842371952640053,
        0.0005763965447680048,  0.0017890231856640168, 0.0001276449095680014 };

    int n = 0;
    for ( int i = cx - 1; i < cx + 2; ++i )
        for ( int j = cy - 1; j < cy + 2; ++j )
            for ( int k = cz - 1; k < cz + 2; ++k, ++n )
                EXPECT_NEAR( gm_host( i, j, k, 0 ), mathematica[n], near_eps );
}

//---------------------------------------------------------------------------//
// Cubic
//---------------------------------------------------------------------------//
// Check the grid velocity. Computed in mathematica.
template <class GridVelocity>
void checkGridVelocity( std::integral_constant<int, 3>, const int cx,
                        const int cy, const int cz, const GridVelocity& gv_host,
                        const double near_eps, const int test_dim,
                        const int array_dim )
{
    double mathematica[64][3] = {
        { -978.9259866239001, -974.0650036214, -967.2040897101 },
        { -977.2231571579001, -972.3621741554, -965.5012602441 },
        { -975.5019187247001, -970.6409357222, -963.7800218108999 },
        { -973.7622713243002, -968.9012883218, -962.0403744105 },
        { -1016.2663703435001, -1011.4135198706, -1004.5607393312999 },
        { -1014.5217552050001, -1009.6689047321, -1002.8161241928 },
        { -1012.7582793623001, -1007.9054288894, -1001.0526483501 },
        { -1010.9759428154, -1006.1230923425, -999.2703118031999 },
        { -1054.5408038135001, -1049.6961844310001, -1042.8516358245 },
        { -1052.7538965095, -1047.909277127, -1041.0647285204998 },
        { -1050.9476712887001, -1046.1030519062, -1039.2585032997 },
        { -1049.1221281511, -1044.2775087686, -1037.4329601620998 },
        { -1093.7608186475002, -1088.9245289162, -1082.0883108033 },
        { -1091.931112685, -1087.0948229537, -1080.2586048407998 },
        { -1090.0816261175, -1085.2453363862, -1078.4091182732998 },
        { -1088.212358945, -1083.3760692137, -1076.5398511007998 },
        { -1004.9034686363999, -999.9528653024, -993.0023316500001 },
        { -1003.1714065803999, -998.2208032464, -991.2702695940001 },
        { -1001.4206195291999, -996.4700161952001, -989.5194825428001 },
        { -999.6511074828, -994.7005041488001, -987.7499704964001 },
        { -1043.2260542628, -1038.2837230592, -1031.3414623868 },
        { -1041.4514891968, -1036.5091579931998, -1029.5668973207999 },
        { -1039.6577396436, -1034.71540844, -1027.7731477676 },
        { -1037.8448056032, -1032.9024743996, -1025.9602137272 },
        { -1082.507045518, -1077.5730866976, -1070.639199258 },
        { -1080.689462254, -1075.7555034336, -1068.821615994 },
        { -1078.8522294412, -1073.9182706208, -1066.9843831812 },
        { -1076.9953470796, -1072.0613882592, -1065.1275008195998 },
        { -1122.7582722324, -1117.832786048, -1110.907372094 },
        { -1120.8971555824, -1115.971669398, -1109.046255444 },
        { -1119.0159187524, -1114.0904325679999, -1107.165018614 },
        { -1117.1145617424002, -1112.189075558, -1105.2636616040002 },
        { -1031.3347727453001, -1026.2913113106, -1019.2479201478999 },
        { -1029.5732293113, -1024.5297678765999, -1017.4863767138999 },
        { -1027.7926421645002, -1022.7491807298, -1015.7057895670999 },
        { -1025.9930113049002, -1020.9495498701999, -1013.9061587074999 },
        { -1070.6566301369, -1065.6215816213999, -1058.5866042347 },
        { -1068.8518602504003, -1063.8168117349, -1056.7818343482002 },
        { -1067.0275793381002, -1061.9925308226, -1054.9575534359 },
        { -1065.1837874, -1060.1487388845, -1053.1137614978002 },
        { -1110.9616687653001, -1105.9351351281998, -1098.9086734767 },
        { -1109.1131484693003, -1104.0866148322, -1097.0601531807001 },
        { -1107.2446441701002, -1102.2181105329998, -1095.1916488815 },
        { -1105.3561558677002, -1100.3296222305999, -1093.3031605791 },
        { -1152.2620217753, -1147.2441049758002, -1140.2262610187 },
        { -1150.3692271128, -1145.3513103133, -1138.3334663562 },
        { -1148.4559698053001, -1143.4380530058, -1136.4202090487001 },
        { -1146.5222498528, -1141.5043330533001, -1134.4864890962 },
        { -1058.2237845452, -1053.0841429855998, -1045.9445722884 },
        { -1056.4325109452002, -1051.2928693856, -1044.1532986884001 },
        { -1054.6218722252001, -1049.4822306656, -1042.3426599684 },
        { -1052.7918683852001, -1047.6522268255999, -1040.5126561284 },
        { -1098.5621305772002, -1093.4310439136, -1086.3000289764 },
        { -1096.7269009772, -1091.5958143136, -1084.4647993764 },
        { -1094.8718310572, -1089.7407443936, -1082.6097294564 },
        { -1092.9969208172001, -1087.8658341536, -1080.7348192164 },
        { -1139.9088567692, -1134.7864286815998, -1127.6640731844 },
        { -1138.0291383692, -1132.9067102815998, -1125.7843547844 },
        { -1136.1290986892, -1131.0066706015998, -1123.8843151044 },
        { -1134.2087377292, -1129.0863096415999, -1121.9639541444 },
        { -1182.2764047212002, -1177.1627388896, -1170.0491465124 },
        { -1180.3516647212, -1175.2379988895998, -1168.1244065124 },
        { -1178.4061167212, -1173.2924508895999, -1166.1788585124 },
        { -1176.4397607212002, -1171.3260948896, -1164.2125025124 } };

    int n = 0;
    for ( int i = cx - 1; i < cx + 3; ++i )
        for ( int j = cy - 1; j < cy + 3; ++j )
            for ( int k = cz - 1; k < cz + 3; ++k, ++n )
                EXPECT_NEAR( gv_host( i, j, k, array_dim ),
                             mathematica[n][test_dim], near_eps );
}

//---------------------------------------------------------------------------//
// Check particle velocity and affine velocity. Computed in Mathematica.
template <class AffineVelocity>
void checkParticleVelocity( std::integral_constant<int, 3>,
                            const AffineVelocity& pb_host,
                            const double near_eps, const int array_dim )
{
    double mathematica_u[3] = { -1075.4622106732168, -1070.4659603794062,
                                -1063.4697813630962 };

    double mathematica_b[3][3] = {
        { -4.649980486421247, -6.633602112810349, 0.3047536994385708 },
        { -4.634471032846672, -6.635011231192498, 0.3047536994385796 },
        { -4.618961679400126, -6.636420491918657, 0.304753699438543 } };

    EXPECT_NEAR( pb_host( 0, array_dim ), mathematica_u[array_dim], near_eps );

    EXPECT_NEAR( pb_host( 1, array_dim ), mathematica_b[array_dim][0],
                 near_eps );
    EXPECT_NEAR( pb_host( 2, array_dim ), mathematica_b[array_dim][1],
                 near_eps );
    EXPECT_NEAR( pb_host( 3, array_dim ), mathematica_b[array_dim][2],
                 near_eps );
}

//---------------------------------------------------------------------------//
// Check the grid momentum. Computed in mathematica.
template <class GridMomentum>
void checkGridMomentum( std::integral_constant<int, 3>, const int cx,
                        const int cy, const int cz, const GridMomentum& gv_host,
                        const double near_eps, const int test_dim,
                        const int array_dim )
{
    double mathematica[64][3] = {
        { -0.0018334085691580541, -0.0018242775143538393,
          -0.0018113871537709235 },
        { -0.020276068915677896, -0.020174896902789185, -0.020032071812479478 },
        { -0.012599109067265751, -0.012536124675598734, -0.012447209261433377 },
        { -0.00018999100203442195, -0.00018903942457467065,
          -0.0001876960774103205 },
        { -0.033618546169002776, -0.0334579578698969, -0.03323113923491277 },
        { -0.3718222807143902, -0.37004296366775313, -0.36752981503881044 },
        { -0.23105937032158153, -0.22995166080226556, -0.2283871062081323 },
        { -0.0034845684493257, -0.003467833010868095, -0.0034441954897787685 },
        { -0.029869903895024384, -0.02973284448464972, -0.029539159894341933 },
        { -0.33038465827887886, -0.3288660411278194, -0.32672001744412416 },
        { -0.2053230548237571, -0.20437764339322853, -0.20304164154487916 },
        { -0.003096655598189641, -0.0030823721831455078,
          -0.0030621876705793404 },
        { -0.0009981801369304203, -0.0009937741692505194,
          -0.0009875447214025821 },
        { -0.011041355264508085, -0.010992537178486693, -0.01092351494704963 },
        { -0.006862268960448425, -0.00683187737919376, -0.0067889077561286825 },
        { -0.00010350242539818213, -0.00010304326491535914,
          -0.0001023940735255477 },
        { -0.08277872320979247, -0.08237024123492526, -0.08179674611984566 },
        { -0.9155159369497178, -0.9109899599939123, -0.9046356387880357 },
        { -0.5689115530678697, -0.5660939170722915, -0.562138049790604 },
        { -0.008579476780915357, -0.008536907521576457, -0.008477141696434848 },
        { -1.516228374990962, -1.509044147457852, -1.4989527672896712 },
        { -16.77034753945796, -16.690746356882673, -16.578933946779866 },
        { -10.422004912255963, -10.372449388773452, -10.30284084358325 },
        { -0.1571801862209638, -0.15643149407829082, -0.15537983793212592 },
        { -1.3458015684577613, -1.339669749536383, -1.331052392245448 },
        { -14.88629100533356, -14.818350501607899, -14.722870252983983 },
        { -9.251749038266926, -9.20945284216951, -9.150011844564693 },
        { -0.1395399601850086, -0.1389009430289439, -0.1380028996882237 },
        { -0.04493131499416474, -0.044734192538660913, -0.04445702919094041 },
        { -0.4970277603525764, -0.4948436451504989, -0.4917726775143271 },
        { -0.30891863985944884, -0.30755892422855746, -0.3056471007620605 },
        { -0.004659561853888135, -0.004639019070004975, -0.004610134959678722 },
        { -0.11650605303971504, -0.11593618058342019, -0.11514034597850635 },
        { -1.2885970408791105, -1.282282858703074, -1.2734650177600477 },
        { -0.800787907737832, -0.796857028911987, -0.7913675033825908 },
        { -0.012076891557140127, -0.012017503261267343, -0.011934566701585717 },
        { -2.131796633894308, -2.121773614862722, -2.10776966161621 },
        { -23.579961027303362, -23.46890605803745, -23.3137423700816 },
        { -14.654541348044624, -14.585404345813519, -14.488807567557664 },
        { -0.22102365799160156, -0.21997912598586739, -0.21851972764224561 },
        { -1.8903670245974924, -1.881811994314449, -1.8698533709009622 },
        { -20.910771583593437, -20.8159819177095, -20.683480467651183 },
        { -12.996484992638331, -12.937473915646253, -12.85498545872388 },
        { -0.1960285083545994, -0.19513696031777497, -0.19389071256941254 },
        { -0.06305613177293849, -0.0627811004672356, -0.06239646448423626 },
        { -0.6975510902381834, -0.6945037456102081, -0.6902419820501527 },
        { -0.4335675050702909, -0.4316703880503508, -0.42901723746458575 },
        { -0.006539958367692276, -0.006511296444525192, -0.006471212254967764 },
        { -0.00864647090928313, -0.008604532374148202, -0.008546265893609191 },
        { -0.0956374192392317, -0.09517274061138623, -0.09452714848140016 },
        { -0.05943586941667557, -0.05914658486198647, -0.05874467304456607 },
        { -0.0008964101112975921, -0.0008920395576665961,
          -0.0008859674147730544 },
        { -0.15805591816894898, -0.15731827325007944, -0.15629296742465618 },
        { -1.7483441753849738, -1.740171075689557, -1.7288106954915643 },
        { -1.086613260506619, -1.0815251170798799, -1.07445273989328 },
        { -0.01638930428470739, -0.016312431864910687, -0.01620558134941778 },
        { -0.14002830471926708, -0.13939867733950667, -0.13852310751694674 },
        { -1.5490209874992118, -1.5420447212576815, -1.532343414752314 },
        { -0.9627881950161837, -0.9584451373583979, -0.9524056124822058 },
        { -0.014522511961632595, -0.014456896407299732, -0.014365650376534287 },
        { -0.004666902807995712, -0.004646660618648407, -0.004618498433859981 },
        { -0.05162901257255609, -0.051404729279393074, -0.051092692501224404 },
        { -0.03209156819749205, -0.0319519411763663, -0.031757683435618314 },
        { -0.000484089505335131, -0.00048197999974876414,
          -0.0004790451252094456 } };

    int n = 0;
    for ( int i = cx - 1; i < cx + 3; ++i )
        for ( int j = cy - 1; j < cy + 3; ++j )
            for ( int k = cz - 1; k < cz + 3; ++k, ++n )
                EXPECT_NEAR( gv_host( i, j, k, array_dim ),
                             mathematica[n][test_dim], near_eps );
}

//---------------------------------------------------------------------------//
// Check the grid mass. Computed in mathematica.
template <class GridMass>
void checkGridMass( std::integral_constant<int, 3>, const int cx, const int cy,
                    const int cz, const GridMass& gm_host,
                    const double near_eps )
{
    double mathematica[64] = { 1.8797178095437055e-6,   0.000020827258025711575,
                               0.000012965959057119188, 1.9589161784782678e-7,
                               0.00003311632582608277,  0.0003669286205309665,
                               0.0002284305243069647,   3.4511619830967442e-6,
                               0.00002831359202043021,  0.00031371436906658576,
                               0.0001953021208997269,   2.9506531883696727e-6,
                               9.11772742165714e-7,     0.000010102434559847897,
                               6.289246175301007e-6,    9.501885690796734e-8,
                               0.00008250944453604763,  0.0009142039736951518,
                               0.0005691354703606992,   8.598582455211928e-6,
                               0.0014536275791567563,   0.01610618174210809,
                               0.010026864447393296,    0.00015148734388932547,
                               0.001242813542843995,    0.013770363936139954,
                               0.008572706727750885,    0.00012951771502872435,
                               0.00004002189164631248,  0.0004434422335963082,
                               0.0002760638888666562,   4.170813865848138e-6,
                               0.00011298504437458089,  0.001251873371783976,
                               0.0007793507365784994,   0.000011774545638059265,
                               0.0019905379009476495,   0.022055143736207216,
                               0.013730376333241364,    0.00020744054656730962,
                               0.0017018578185459842,   0.018856570773533286,
                               0.011739112479687522,    0.0001773562391808549,
                               0.00005480433457090743,  0.0006072315808471222,
                               0.0003780305504322166,   5.711341196886726e-6,
                               8.16426203005055e-6,     0.00009045995682227375,
                               0.000056315627098795606, 8.508247831031677e-7,
                               0.0001438356119966227,   0.001593697409861213,
                               0.0009921524638619055,   0.000014989585455377672,
                               0.00012297568448471193,  0.0013625695828637832,
                               0.0008482643947694685,   0.000012815703329161863,
                               3.960143134838734e-6,    0.00004387835369185446,
                               0.000027316362852951293, 4.126996305795691e-7 };

    int n = 0;
    for ( int i = cx - 1; i < cx + 3; ++i )
        for ( int j = cy - 1; j < cy + 3; ++j )
            for ( int k = cz - 1; k < cz + 3; ++k, ++n )
                EXPECT_NEAR( gm_host( i, j, k, 0 ), mathematica[n], near_eps );
}

//---------------------------------------------------------------------------//
template <class Location, int Order>
void collocatedTest()
{
    // Test epsilon
    double near_eps = 1.0e-11;

    // Global mesh parameters.
    Kokkos::Array<double, 6> global_box = { -50.0, -50.0, -50.0,
                                            50.0,  50.0,  50.0 };

    // Get inputs for mesh.
    auto inputs = Picasso::parse( "polypic_test.json" );

    // Make mesh.
    int minimum_halo_size = 0;
    UniformMesh<TEST_MEMSPACE> mesh( inputs, global_box, minimum_halo_size,
                                     MPI_COMM_WORLD );
    auto local_mesh =
        Cabana::Grid::createLocalMesh<TEST_EXECSPACE>( *( mesh.localGrid() ) );

    // Particle mass.
    double pm = 0.134;

    // Particle location.
    double px = 9.31;
    double py = -8.28;
    double pz = -3.34;

    // Particle cell location.
    int cx = 120;
    int cy = 85;
    int cz = 95;

    // Particle velocity.
    Kokkos::View<double[4][3], TEST_MEMSPACE> pb( "pb" );

    // Create a grid vector on the entities.
    auto grid_vector = createArray( mesh, Location(), Foo() );

    // Initialize the grid vector.
    auto gv_view = grid_vector->view();
    Cabana::Grid::grid_parallel_for(
        "fill_grid_vector", TEST_EXECSPACE(),
        grid_vector->layout()->indexSpace( Cabana::Grid::Own(),
                                           Cabana::Grid::Local() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int d ) {
            int ic = i - 2;
            int jc = j - 2;
            int kc = k - 2;
            gv_view( i, j, k, d ) =
                0.0000000001 * ( d + 1 ) * pow( ic, 5 ) -
                0.0000000012 * pow( ( d + 1 ) + jc * ic, 3 ) +
                0.0000000001 * pow( ic * jc * kc, 2 ) + pow( ( d + 1 ), 2 );
        } );

    // Check the grid velocity. Computed in Mathematica.
    auto gv_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), gv_view );
    checkGridVelocity( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 0, 0 );
    checkGridVelocity( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 1, 1 );
    checkGridVelocity( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 2, 2 );

    // Do G2P.
    auto gv_wrapper =
        createViewWrapper( FieldLayout<Location, Foo>(), gv_view );
    Kokkos::parallel_for(
        "g2p", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) {
            Vec3<double> x = { px, py, pz };
            auto sd =
                createSpline( Location(), InterpolationOrder<Order>(),
                              local_mesh, x, SplineValue(), SplineDistance() );

            LinearAlgebra::Matrix<double, 4, 3> aff = 0.0;

            APIC::g2p( gv_wrapper, aff, sd );

            for ( int d = 0; d < 3; ++d )
                pb( 0, d ) = aff( 0, d );

            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    pb( i + 1, j ) = aff( i + 1, j );
        } );

    // Check particle velocity. Computed in Mathematica.
    auto pb_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pb );
    checkParticleVelocity( std::integral_constant<int, Order>(), pb_host,
                           near_eps, 0 );
    checkParticleVelocity( std::integral_constant<int, Order>(), pb_host,
                           near_eps, 1 );
    checkParticleVelocity( std::integral_constant<int, Order>(), pb_host,
                           near_eps, 2 );

    // Reset the grid view.
    Kokkos::deep_copy( gv_view, 0.0 );

    // Create grid mass on the entities.
    auto grid_mass = createArray( mesh, Location(), Baz() );
    auto gm_view = grid_mass->view();
    Kokkos::deep_copy( gm_view, 0.0 );

    // Do P2G
    auto gv_sv = Kokkos::Experimental::create_scatter_view( gv_view );
    auto gm_sv = Kokkos::Experimental::create_scatter_view( gm_view );
    Kokkos::parallel_for(
        "p2g", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) {
            Vec3<double> x = { px, py, pz };
            auto sd =
                createSpline( Location(), InterpolationOrder<Order>(),
                              local_mesh, x, SplineValue(), SplineGradient(),
                              SplineDistance(), SplineCellSize() );

            LinearAlgebra::Matrix<double, 4, 3> aff = 0.0;

            for ( int d = 0; d < 3; ++d )
                aff( 0, d ) = pb( 0, d );

            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    aff( i + 1, j ) = pb( i + 1, j );

            APIC::p2g( pm, aff, gm_sv, gv_sv, sd );
        } );
    Kokkos::Experimental::contribute( gv_view, gv_sv );
    Kokkos::Experimental::contribute( gm_view, gm_sv );

    // Check grid momentum. Computed in Mathematica.
    Kokkos::deep_copy( gv_host, gv_view );
    checkGridMomentum( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 0, 0 );
    checkGridMomentum( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 1, 1 );
    checkGridMomentum( std::integral_constant<int, Order>(), cx, cy, cz,
                       gv_host, near_eps, 2, 2 );

    // Check grid mass. Computed in Mathematica.
    auto gm_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), gm_view );
    checkGridMass( std::integral_constant<int, Order>(), cx, cy, cz, gm_host,
                   near_eps );
}

//---------------------------------------------------------------------------//
template <class Location, int Order>
void staggeredTest()
{
    // Test epsilon
    double near_eps = 1.0e-11;

    // Test dimension
    const int Dim = Location::entity_type::dim;

    // Global mesh parameters.
    Kokkos::Array<double, 6> global_box = { -50.0, -50.0, -50.0,
                                            50.0,  50.0,  50.0 };

    // If we are using edges offset the non-test dimensions to align with the
    // mathematica grid.
    if ( Cabana::Grid::isEdge<typename Location::entity_type>::value )
        for ( int d = 0; d < 3; ++d )
        {
            if ( d != Dim )
            {
                global_box[d] -= 0.25;
                global_box[d + 3] -= 0.25;
            }
        }

    // If we are using faces offset that dimension to align with the
    // mathematica grid for the quadratic functions.
    else if ( Cabana::Grid::isFace<typename Location::entity_type>::value )
    {
        global_box[Dim] += 0.25;
        global_box[Dim + 3] += 0.25;
    }

    // Get inputs for mesh.
    auto inputs = Picasso::parse( "polypic_test.json" );

    // Make mesh.
    int minimum_halo_size = 0;
    UniformMesh<TEST_MEMSPACE> mesh( inputs, global_box, minimum_halo_size,
                                     MPI_COMM_WORLD );
    auto local_mesh =
        Cabana::Grid::createLocalMesh<TEST_EXECSPACE>( *( mesh.localGrid() ) );

    // Particle mass.
    double pm = 0.134;

    // Particle location.
    double px = 9.31;
    double py = -8.28;
    double pz = -3.34;

    // Particle cell location.
    int cx = 120;
    int cy = 85;
    int cz = 95;

    // Particle velocity.
    Kokkos::View<double[4][3], TEST_MEMSPACE> pb( "pb" );

    // Create a grid scalar on the faces in the test dimension.
    auto grid_scalar = createArray( mesh, FieldLocation::Face<Dim>(), Bar() );

    // Initialize the grid scalar.
    auto gs_view = grid_scalar->view();
    Cabana::Grid::grid_parallel_for(
        "fill_grid_scalar", TEST_EXECSPACE(),
        grid_scalar->layout()->indexSpace( Cabana::Grid::Own(),
                                           Cabana::Grid::Local() ),
        KOKKOS_LAMBDA( const int i, const int j, const int k, const int ) {
            int ic = i - 2;
            int jc = j - 2;
            int kc = k - 2;
            gs_view( i, j, k, 0 ) =
                0.0000000001 * ( Dim + 1 ) * pow( ic, 5 ) -
                0.0000000012 * pow( ( Dim + 1 ) + jc * ic, 3 ) +
                0.0000000001 * pow( ic * jc * kc, 2 ) + pow( ( Dim + 1 ), 2 );
        } );

    // Check the grid velocity. Computed in Mathematica.
    auto gs_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), gs_view );
    checkGridVelocity( std::integral_constant<int, Order>(), cx, cy, cz,
                       gs_host, near_eps, Dim, 0 );

    // Do G2P.
    auto gs_wrapper = createViewWrapper(
        FieldLayout<FieldLocation::Face<Dim>, Bar>(), gs_view );
    Kokkos::parallel_for(
        "g2p", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) {
            Vec3<double> x = { px, py, pz };
            auto sd = createSpline( FieldLocation::Face<Dim>(),
                                    InterpolationOrder<Order>(), local_mesh, x,
                                    SplineValue(), SplineDistance() );

            LinearAlgebra::Matrix<double, 4, 3> aff = 0.0;

            APIC::g2p( gs_wrapper, aff, sd );

            for ( int d = 0; d < 3; ++d )
                pb( 0, d ) = aff( 0, d );

            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    pb( i + 1, j ) = aff( i + 1, j );
        } );

    // Check particle velocity. Computed in Mathematica.
    auto pb_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), pb );
    checkParticleVelocity( std::integral_constant<int, Order>(), pb_host,
                           near_eps, Dim );

    // Reset the grid view.
    Kokkos::deep_copy( gs_view, 0.0 );

    // Create grid mass on the faces.
    auto grid_mass = createArray( mesh, FieldLocation::Face<Dim>(), Baz() );
    auto gm_view = grid_mass->view();
    Kokkos::deep_copy( gm_view, 0.0 );

    // Do P2G
    auto gs_sv = Kokkos::Experimental::create_scatter_view( gs_view );
    auto gm_sv = Kokkos::Experimental::create_scatter_view( gm_view );
    Kokkos::parallel_for(
        "p2g", Kokkos::RangePolicy<TEST_EXECSPACE>( 0, 1 ),
        KOKKOS_LAMBDA( const int ) {
            Vec3<double> x = { px, py, pz };
            auto sd = createSpline( FieldLocation::Face<Dim>(),
                                    InterpolationOrder<Order>(), local_mesh, x,
                                    SplineValue(), SplineGradient(),
                                    SplineDistance(), SplineCellSize() );

            LinearAlgebra::Matrix<double, 4, 3> aff = 0.0;

            for ( int d = 0; d < 3; ++d )
                aff( 0, d ) = pb( 0, d );

            for ( int i = 0; i < 3; ++i )
                for ( int j = 0; j < 3; ++j )
                    aff( i + 1, j ) = pb( i + 1, j );

            APIC::p2g( pm, aff, gm_sv, gs_sv, sd );
        } );
    Kokkos::Experimental::contribute( gs_view, gs_sv );
    Kokkos::Experimental::contribute( gm_view, gm_sv );

    // Check grid momentum. Computed in Mathematica.
    Kokkos::deep_copy( gs_host, gs_view );
    checkGridMomentum( std::integral_constant<int, Order>(), cx, cy, cz,
                       gs_host, near_eps, Dim, 0 );

    // Check grid mass. Computed in Mathematica.
    auto gm_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), gm_view );
    checkGridMass( std::integral_constant<int, Order>(), cx, cy, cz, gm_host,
                   near_eps );
}

//---------------------------------------------------------------------------//
// RUN TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, linear_collocated_test )
{
    // serial test only.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    if ( comm_size > 1 )
        return;

    // test
    collocatedTest<FieldLocation::Node, 1>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, quadratic_collocated_test )
{
    // serial test only.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    if ( comm_size > 1 )
        return;

    // test
    collocatedTest<FieldLocation::Cell, 2>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, cubic_collocated_test )
{
    // serial test only.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    if ( comm_size > 1 )
        return;

    // test
    collocatedTest<FieldLocation::Node, 3>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, linear_staggered_test )
{
    // serial test only.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    if ( comm_size > 1 )
        return;

    // test
    staggeredTest<FieldLocation::Edge<Dim::I>, 1>();
    staggeredTest<FieldLocation::Edge<Dim::J>, 1>();
    staggeredTest<FieldLocation::Edge<Dim::K>, 1>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, quadratic_staggered_test )
{
    // serial test only.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    if ( comm_size > 1 )
        return;

    // test
    staggeredTest<FieldLocation::Face<Dim::I>, 2>();
    staggeredTest<FieldLocation::Face<Dim::J>, 2>();
    staggeredTest<FieldLocation::Face<Dim::K>, 2>();
}

//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, cubic_staggered_test )
{
    // serial test only.
    int comm_size;
    MPI_Comm_size( MPI_COMM_WORLD, &comm_size );
    if ( comm_size > 1 )
        return;

    // test
    staggeredTest<FieldLocation::Edge<Dim::I>, 3>();
    staggeredTest<FieldLocation::Edge<Dim::J>, 3>();
    staggeredTest<FieldLocation::Edge<Dim::K>, 3>();
}

//---------------------------------------------------------------------------//

} // end namespace Test
