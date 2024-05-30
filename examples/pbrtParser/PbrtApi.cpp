/*
* Copyright (c) 2017 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from NVIDIA Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES
*/

#include <OptiXToolkit/PbrtApi/PbrtApi.h>

#include <parser.h>

#include <api.h>
#include <stringprint.h>

#include <glog/logging.h>

#include <cstdarg>

static ::otk::pbrt::Api* g_api{};

namespace otk {
namespace pbrt {

void setApi( Api* api )
{
    g_api = api;
}

Api* getApi()
{
    return g_api;
}

}  // namespace pbrt
}  // namespace otk

// pbrt global function API and global data needed by the parser code
namespace pbrt {

void pbrtObjectBegin( const std::string& name )
{
    g_api->objectBegin( name );
}

void pbrtObjectEnd()
{
    g_api->objectEnd();
}

void pbrtShape( const std::string& name, const ParamSet& params )
{
    g_api->shape( name, params );
}

void pbrtAttributeBegin()
{
    g_api->attributeBegin();
}

void pbrtAttributeEnd()
{
    g_api->attributeEnd();
}

void pbrtConcatTransform( float transform[16] )
{
    g_api->concatTransform( transform );
}

void pbrtIdentity()
{
    g_api->identity();
}

void pbrtTranslate( float dx, float dy, float dz )
{
    g_api->translate( dx, dy, dz );
}

void pbrtRotate( float angle, float ax, float ay, float az )
{
    g_api->rotate( angle, ax, ay, az );
}

void pbrtScale( float sx, float sy, float sz )
{
    g_api->scale( sx, sy, sz );
}

void pbrtLookAt( float ex, float ey, float ez, float lx, float ly, float lz, float ux, float uy, float uz )
{
    g_api->lookAt( ex, ey, ez, lx, ly, lz, ux, uy, uz );
}

void pbrtTransform( float transform[16] )
{
    g_api->transform( transform );
}

void pbrtCoordinateSystem( const std::string& name )
{
    g_api->coordinateSystem( name );
}

void pbrtCoordSysTransform( const std::string& name )
{
    g_api->coordSysTransform( name );
}

void pbrtActiveTransformAll()
{
    g_api->activeTransformAll();
}

void pbrtActiveTransformEndTime()
{
    g_api->activeTransformEndTime();
}

void pbrtActiveTransformStartTime()
{
    g_api->activeTransformStartTime();
}

void pbrtTransformTimes( float start, float end )
{
    g_api->transformTimes( start, end );
}

void pbrtPixelFilter( const std::string& name, const ParamSet& params )
{
    g_api->pixelFilter( name, params );
}

void pbrtFilm( const std::string& type, const ParamSet& params )
{
    g_api->film( type, params );
}

void pbrtSampler( const std::string& name, const ParamSet& params )
{
    g_api->sampler( name, params );
}

void pbrtAccelerator( const std::string& name, const ParamSet& params )
{
    g_api->accelerator( name, params );
}

void pbrtIntegrator( const std::string& name, const ParamSet& params )
{
    g_api->integrator( name, params );
}

void pbrtCamera( const std::string& name, const ParamSet& cameraParams )
{
    g_api->camera( name, cameraParams );
}

void pbrtMakeNamedMedium( const std::string& name, const ParamSet& params )
{
    g_api->makeNamedMedium( name, params );
}

void pbrtMediumInterface( const std::string& insideName, const std::string& outsideName )
{
    g_api->mediumInterface( insideName, outsideName );
}

void pbrtWorldBegin()
{
    g_api->worldBegin();
}

void pbrtTransformBegin()
{
    g_api->transformBegin();
}

void pbrtTransformEnd()
{
    g_api->transformEnd();
}

void pbrtTexture( const std::string& name, const std::string& type, const std::string& texname, const ParamSet& params )
{
    g_api->texture( name, type, texname, params );
}

void pbrtMaterial( const std::string& name, const ParamSet& params )
{
    g_api->material( name, params );
}

void pbrtMakeNamedMaterial( const std::string& name, const ParamSet& params )
{
    g_api->makeNamedMaterial( name, params );
}

void pbrtNamedMaterial( const std::string& name )
{
    g_api->namedMaterial( name );
}

void pbrtLightSource( const std::string& name, const ParamSet& params )
{
    g_api->lightSource( name, params );
}

void pbrtAreaLightSource( const std::string& name, const ParamSet& params )
{
    g_api->areaLightSource( name, params );
}

void pbrtReverseOrientation()
{
    g_api->reverseOrientation();
}

void pbrtObjectInstance( const std::string& name )
{
    g_api->objectInstance( name );
}

void pbrtWorldEnd()
{
    g_api->worldEnd();
}

// pbrt global function API and global data needed by the parser code
static int  g_pbrtErrorCount{};
static int  g_pbrtWarningCount{};
Options     PbrtOptions;
int         catIndentCount = 0;
int         line_num       = 0;
std::string current_file;

template <typename... Args>
std::string StringVaprintf( const std::string& fmt, va_list args )
{
    // Figure out how much space we need to allocate; add an extra
    // character for '\0'.
    va_list argsCopy;
    va_copy( argsCopy, args );
    size_t      size = vsnprintf( nullptr, 0, fmt.c_str(), args ) + 1;
    std::string str;
    str.resize( size );
    vsnprintf( &str[0], size, fmt.c_str(), argsCopy );
    str.pop_back();  // remove trailing NUL
    return str;
}

enum class Severity
{
    warning = 1,
    error   = 2,
};

static void processError( const char* format, va_list args, Severity severity )
{
    // Build up an entire formatted error string and print it all at once;
    // this way, if multiple threads are printing messages at once, they
    // don't get jumbled up...
    std::string errorString;

    // Print line and position in input file, if available
    if( line_num != 0 )
    {
        errorString += current_file;
        errorString += StringPrintf( "(%d): ", line_num );
    }

    errorString += StringVaprintf( format, args );

    // Report the error message (but not more than one time).
    static std::string lastError;
    if( errorString != lastError )
    {
        if( severity == Severity::warning )
            g_api->warning( errorString, __FILE__, __LINE__ );
        else
            g_api->error( errorString, __FILE__, __LINE__ );
        lastError = errorString;
    }
}

void Warning( const char* format, ... )
{
    ++g_pbrtWarningCount;
    va_list args;
    va_start( args, format );
    processError( format, args, Severity::warning );
    va_end( args );
}

void Error( const char* format, ... )
{
    ++g_pbrtErrorCount;
    va_list args;
    va_start( args, format );
    processError( format, args, Severity::error );
    va_end( args );
}

}  // namespace pbrt
