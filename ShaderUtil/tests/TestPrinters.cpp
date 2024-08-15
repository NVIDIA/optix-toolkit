// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/vec_printers.h>

#include <vector_functions.h>

#include <gtest/gtest.h>

#include <sstream>

TEST( TestPrinters, print2 )
{
    std::ostringstream str;

    str << make_int2( 1, 2 );

    EXPECT_EQ( "(1, 2)", str.str() );
}

TEST( TestPrinters, print3 )
{
    std::ostringstream str;

    str << make_int3( 1, 2, 3 );

    EXPECT_EQ( "(1, 2, 3)", str.str() );
}

TEST( TestPrinters, print4 )
{
    std::ostringstream str;

    str << make_int4( 1, 2, 3, 4 );

    EXPECT_EQ( "(1, 2, 3, 4)", str.str() );
}
