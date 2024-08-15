// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <string>

namespace otk {

class EXRInputFile
{
  public:
    /// The destructor closes the file if necessary.
    ~EXRInputFile();

    /// Open the specified file.  Throws an exception on failure.
    void open( const std::string& filename );

    /// Close the file.
    void close();

    /// Get the image width.
    unsigned int getWidth() const;

    /// Get the image height.
    unsigned int getHeight() const;

    /// Read the image data as half4 pixels.  The given buffer must be sized accordingly.  Throws an
    /// exception on failure.
    void read( void* pixels, size_t size );

  private:
    class EXRInputFileImpl* m_file;
};

}  // namespace otk
