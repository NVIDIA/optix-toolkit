// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <mi/mdl_sdk.h>

#include <gtest/gtest.h>

#include <cstring>

TEST( TestMdlSdk, headerProvidesVersionMetadata )
{
    EXPECT_GT( std::strlen( MI_NEURAYLIB_PRODUCT_VERSION_STRING ), 0U );
    EXPECT_GT( MI_NEURAYLIB_API_VERSION, 0 );
}

TEST( TestMdlSdk, headerProvidesNeurayInterfaceId )
{
    const mi::base::Uuid id = mi::neuraylib::INeuray::IID();

    EXPECT_NE( 0U, id.m_id1 | id.m_id2 | id.m_id3 | id.m_id4 );
}
