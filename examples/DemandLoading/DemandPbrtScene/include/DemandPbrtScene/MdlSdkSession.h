// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <mi/mdl_sdk.h>

#include <memory>
#include <string>

namespace demandPbrtScene {

using NeurayHandle = mi::base::Handle<mi::neuraylib::INeuray>;

class MdlSdkSession
{
  public:
    MdlSdkSession();
    ~MdlSdkSession();

    MdlSdkSession( const MdlSdkSession& )            = delete;
    MdlSdkSession& operator=( const MdlSdkSession& ) = delete;

    bool isStarted() const;

    const std::string& error() const;

    const NeurayHandle& handle() const;
    mi::neuraylib::INeuray* neuray() const;

    mi::Sint32 shutdown();
    void       close();

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace demandPbrtScene
