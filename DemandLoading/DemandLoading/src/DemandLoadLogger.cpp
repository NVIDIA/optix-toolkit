// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
#include <cstdio>
#include <OptiXToolkit/DemandLoading/DemandLoadLogger.h>

DemandLoadLogCallback DemandLoadLogger::m_callback = nullptr;
int DemandLoadLogger::m_logLevel = 0;
bool DemandLoadLogger::m_initialized = false;

void standardDemandLoadLogCallback( int level, const char* message )
{
    fprintf( stderr, "[DL%2d] %s\n", level, message );
}
