//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <OptiXToolkit/Util/EXRInputFile.h>
#include <OptiXToolkit/Util/Exception.h>

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
