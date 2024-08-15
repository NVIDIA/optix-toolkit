// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include "Dependencies.h"

#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>

#include <memory>

namespace demandPbrtScene {

struct LookAtParams;
struct Options;
struct PerspectiveCamera;
struct UserInterfaceStatistics;

using OutputBuffer = otk::CUDAOutputBuffer<uchar4>;

class UserInterface
{
  public:
    virtual ~UserInterface() = default;

    virtual void initialize( const LookAtParams& lookAt, const PerspectiveCamera& camera ) = 0;
    virtual void cleanup()    = 0;

    // returns true if launch is needed
    virtual bool beforeLaunch( OutputBuffer& output ) = 0;
    virtual void afterLaunch( OutputBuffer& output )  = 0;
    virtual bool shouldClose() const                  = 0;

    virtual void setStatistics(const UserInterfaceStatistics &stats) = 0;
};

std::shared_ptr<UserInterface> createUserInterface( Options& options, RendererPtr renderer, ScenePtr scene );

}  // namespace demandPbrtScene
