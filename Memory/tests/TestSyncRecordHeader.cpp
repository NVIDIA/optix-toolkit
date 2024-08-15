// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

// Validate that header stands alone.
#include <OptiXToolkit/OptiXMemory/SyncRecord.h>

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <gtest/gtest.h>

#include <cuda.h>

TEST( SyncRecordHeaderTest, IncludesOK )
{
    OTK_ERROR_CHECK( cuInit( 0 ) );

    otk::SyncRecord<int> record{ 1 };

    ASSERT_TRUE( true );
}
