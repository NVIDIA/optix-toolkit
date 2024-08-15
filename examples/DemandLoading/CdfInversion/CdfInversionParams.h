// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

const int SUBFRAME_ID = 0;
const int EMAP_ID = 1;
const int MIP_LEVEL_0_ID = 2;
const int NUM_RIS_SAMPLES = 3;

const int MIP_SCALE_ID = 0;

// Stuffing cdf inversion, alias, and summed area tables in colors
const int EMAP_INVERSION_TABLE_ID = 0; // also 1 and 2
const int EMAP_ALIAS_TABLE_ID = 3;
const int EMAP_SUMMED_AREA_TABLE_ID = 4; // also 5

// Emap sample modes. Uncomment one.
//#define emBIN_SEARCH 1
//#define emLIN_SEARCH 2
#define emDIRECT_LOOKUP 3
//#define emALIAS_TABLE 4
//#define emSUMMED_AREA_TABLE 5
