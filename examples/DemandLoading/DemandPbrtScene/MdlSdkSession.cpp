// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlSdkSession.h"

#ifdef _WIN32
#include <mi/base/miwindows.h>
#else
#include <dlfcn.h>
#endif

#include <sstream>
#include <stdexcept>

namespace demandPbrtScene {
namespace {

#ifdef _WIN32

using MdlLibraryHandle = HMODULE;

std::string lastLibraryError()
{
    std::ostringstream out;
    out << "Windows error " << GetLastError();
    return out.str();
}

MdlLibraryHandle loadMdlSdkLibrary( std::string& error )
{
    const char* const libraryName = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
    MdlLibraryHandle  handle      = LoadLibraryA( libraryName );
    if( handle )
        return handle;

    const std::string fallback = std::string( "../../../bin/" ) + libraryName;
    handle                     = LoadLibraryA( fallback.c_str() );
    if( handle )
        return handle;

    error = "Failed to load " + std::string( libraryName ) + ": " + lastLibraryError();
    return nullptr;
}

void* loadMdlFactorySymbol( MdlLibraryHandle handle, std::string& error )
{
    void* symbol = GetProcAddress( handle, "mi_factory" );
    if( !symbol )
        error = "Failed to find mi_factory: " + lastLibraryError();
    return symbol;
}

void unloadMdlSdkLibrary( MdlLibraryHandle handle )
{
    if( handle )
        FreeLibrary( handle );
}

#else

using MdlLibraryHandle = void*;

MdlLibraryHandle loadMdlSdkLibrary( std::string& error )
{
    const char* const libraryName = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;
    MdlLibraryHandle  handle      = dlopen( libraryName, RTLD_LAZY );
    if( !handle )
        error = dlerror();
    return handle;
}

void* loadMdlFactorySymbol( MdlLibraryHandle handle, std::string& error )
{
    void* symbol = dlsym( handle, "mi_factory" );
    if( !symbol )
        error = dlerror();
    return symbol;
}

void unloadMdlSdkLibrary( MdlLibraryHandle handle )
{
    if( handle )
        dlclose( handle );
}

#endif

}  // namespace

struct MdlSdkSession::Impl
{
    MdlLibraryHandle library{};
    NeurayHandle     neuray;
    std::string      error;
    bool             started{};
};

MdlSdkSession::MdlSdkSession()
    : m_impl( std::make_unique<Impl>() )
{
    m_impl->library = loadMdlSdkLibrary( m_impl->error );
    if( !m_impl->library )
        return;

    void* symbol = loadMdlFactorySymbol( m_impl->library, m_impl->error );
    if( !symbol )
        return;

    m_impl->neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>( symbol );
    if( !m_impl->neuray.is_valid_interface() )
    {
        VersionHandle version( mi::neuraylib::mi_factory<mi::neuraylib::IVersion>( symbol ) );
        m_impl->error = version.is_valid_interface() ? "MDL SDK library version does not match header version "
                                                          + std::string( MI_NEURAYLIB_PRODUCT_VERSION_STRING ) :
                                                      "MDL SDK library is incompatible with this header";
        return;
    }

    const mi::Sint32 startResult = m_impl->neuray->start( true );
    if( startResult != 0 )
    {
        std::ostringstream out;
        out << "Failed to start MDL SDK: " << startResult;
        m_impl->error = out.str();
        return;
    }

    m_impl->started = true;
}

MdlSdkSession::~MdlSdkSession()
{
    shutdown();
    unloadMdlSdkLibrary( m_impl->library );
}

bool MdlSdkSession::isStarted() const
{
    return m_impl->started;
}

const std::string& MdlSdkSession::error() const
{
    return m_impl->error;
}

const NeurayHandle& MdlSdkSession::handle() const
{
    return m_impl->neuray;
}

mi::neuraylib::INeuray* MdlSdkSession::neuray() const
{
    return m_impl->neuray.get();
}

mi::Sint32 MdlSdkSession::shutdown()
{
    mi::Sint32 result = 0;
    if( m_impl->started )
    {
        result          = m_impl->neuray->shutdown( true );
        m_impl->started = false;
    }
    m_impl->neuray.reset();
    return result;
}

void MdlSdkSession::close()
{
    if( shutdown() != 0 )
    {
        throw std::runtime_error( "Failed to shut down MDL SDK" );
    }
}

}  // namespace demandPbrtScene
