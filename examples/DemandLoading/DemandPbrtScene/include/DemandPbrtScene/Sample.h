// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <iostream>
#include <stdexcept>

namespace otk {

class Sample
{
  public:
    virtual ~Sample()         = default;
    virtual void initialize() = 0;
    virtual void run()        = 0;
    virtual void cleanup()    = 0;
};

template <class Application>
int mainLoop( int argc, char* argv[] )
{
    try
    {
        Application app( argc, argv );
        app.initialize();
        app.run();
        app.cleanup();
    }
    catch( const std::exception& e )
    {
        std::cerr << "Caught exception: " << e.what() << '\n';
        return 1;
    }
    catch( ... )
    {
        std::cerr << "Unknown exception\n";
        return 2;
    }
    return 0;
}

}  // namespace otk
