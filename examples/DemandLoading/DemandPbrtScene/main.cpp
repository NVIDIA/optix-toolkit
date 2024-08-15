// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Application.h"
#include "Sample.h"

#include <optix_function_table_definition.h>

int main( int argc, char* argv[] )
{
    return otk::mainLoop<demandPbrtScene::Application>( argc, argv );
}
