// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <DemandPbrtScene/Timer.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

using namespace demandPbrtScene;

class TestTimer : public testing::Test
{
};

TEST_F( TestTimer, TestElapsed )
{
    Timer timer;

    using msec = std::chrono::duration<int, std::milli>;
    std::this_thread::sleep_for( msec( 10 ) );

    double seconds = timer.getSeconds();
    EXPECT_GT( seconds, 0.0 );
}

TEST_F( TestTimer, TestAtomic )
{
    std::atomic<Timer::Nanoseconds> duration( Timer::Nanoseconds( 0LL ) );

    Timer timer;

    using msec = std::chrono::duration<int, std::milli>;
    std::this_thread::sleep_for( msec( 10 ) );

    duration += timer.getNanoseconds();
    EXPECT_GT( duration.load(), 0.0 );
}

