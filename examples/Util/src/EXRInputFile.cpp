// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Util/EXRInputFile.h>

#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>

#include <ImfRgbaFile.h>

namespace otk {

class EXRInputFileImpl : public Imf::RgbaInputFile
{
  public:
    EXRInputFileImpl( const std::string& filename )
        : Imf::RgbaInputFile( filename.c_str() )
    {
    }
    virtual ~EXRInputFileImpl() {}
};

EXRInputFile::~EXRInputFile()
{
    close();
}

void EXRInputFile::open( const std::string& filename )
{
    m_file = new EXRInputFileImpl( filename );
}

void EXRInputFile::close()
{
    delete m_file;
    m_file = nullptr;
}

unsigned int EXRInputFile::getWidth() const
{
    OTK_ASSERT( m_file );
    Imath::Box2i dw = m_file->dataWindow();
    return dw.max.x - dw.min.x + 1;
}

unsigned int EXRInputFile::getHeight() const
{
    OTK_ASSERT( m_file );
    Imath::Box2i dw = m_file->dataWindow();
    return dw.max.y - dw.min.y + 1;
}

void EXRInputFile::read( void* pixels, size_t size )
{
    (void)size;  // silence unused variable warning
    OTK_ASSERT( m_file );
    OTK_ASSERT( size >= 4 * sizeof( half ) * getWidth() * getHeight() );
    m_file->setFrameBuffer( reinterpret_cast<Imf::Rgba*>( pixels ), 1, getWidth() );
    Imath::Box2i dw = m_file->dataWindow();
    m_file->readPixels( dw.min.y, dw.max.y );
}

} // namespace otk
