// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ShaderUtil/AliasTable.h>
#include <OptiXToolkit/ShaderUtil/CdfInversionTable.h>
#include <OptiXToolkit/ShaderUtil/PdfTable.h>
#include <OptiXToolkit/ShaderUtil/ISummedAreaTable.h>

// Emap sample modes. Uncomment one.
//#define emBIN_SEARCH 1
//#define emLIN_SEARCH 2
#define emDIRECT_LOOKUP 3
//#define emALIAS_TABLE 4
//#define emSUMMED_AREA_TABLE 5

struct CdfInversionParams
{
    int emapTextureId;
    bool useMipLevelZero;
    int numRisSamples;
    float mipScale;

    CdfInversionTable emapCdfInversionTable;
    AliasTable emapAliasTable;
    ISummedAreaTable emapSummedAreaTable;
};
