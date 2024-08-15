// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <vector>

namespace demandLoading {

class RequestFilter
{
  public:
    virtual ~RequestFilter() { }
    virtual std::vector<unsigned int> filter( const unsigned int* requests, unsigned int numRequests ) = 0;
};

}  // namespace demandLoading
