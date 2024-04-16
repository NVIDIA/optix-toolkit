//
// Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
//
// NVIDIA Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from NVIDIA Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES
//

#pragma once

#include <core/paramset.h>
#include <core/pbrt.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace otk {
namespace pbrt {

struct SceneDescription;
using SceneDescriptionPtr = std::shared_ptr<SceneDescription>;

class Api
{
  public:
    using ParamSet = ::pbrt::ParamSet;

    virtual ~Api() = default;

    virtual SceneDescriptionPtr parseFile( std::string filename ) = 0;
    virtual SceneDescriptionPtr parseString( std::string str )    = 0;

    virtual void identity()                                                                                         = 0;
    virtual void translate( float dx, float dy, float dz )                                                          = 0;
    virtual void rotate( float angle, float ax, float ay, float az )                                                = 0;
    virtual void scale( float sx, float sy, float sz )                                                              = 0;
    virtual void lookAt( float ex, float ey, float ez, float lx, float ly, float lz, float ux, float uy, float uz ) = 0;
    virtual void concatTransform( float transform[16] )                                                             = 0;
    virtual void transform( float transform[16] )                                                                   = 0;
    virtual void coordinateSystem( const std::string& name )                                                        = 0;
    virtual void coordSysTransform( const std::string& name )                                                       = 0;
    virtual void activeTransformAll()                                                                               = 0;
    virtual void activeTransformEndTime()                                                                           = 0;
    virtual void activeTransformStartTime()                                                                         = 0;
    virtual void transformTimes( float start, float end )                                                           = 0;
    virtual void pixelFilter( const std::string& name, const ParamSet& params )                                     = 0;
    virtual void film( const std::string& type, const ParamSet& params )                                            = 0;
    virtual void sampler( const std::string& name, const ParamSet& params )                                         = 0;
    virtual void accelerator( const std::string& name, const ParamSet& params )                                     = 0;
    virtual void integrator( const std::string& name, const ParamSet& params )                                      = 0;
    virtual void camera( const std::string& name, const ParamSet& cameraParams )                                    = 0;
    virtual void makeNamedMedium( const std::string& name, const ParamSet& params )                                 = 0;
    virtual void mediumInterface( const std::string& insideName, const std::string& outsideName )                   = 0;
    virtual void worldBegin()                                                                                       = 0;
    virtual void attributeBegin()                                                                                   = 0;
    virtual void attributeEnd()                                                                                     = 0;
    virtual void transformBegin()                                                                                   = 0;
    virtual void transformEnd()                                                                                     = 0;
    virtual void texture( const std::string& name, const std::string& type, const std::string& tex_type, const ParamSet& params ) = 0;
    virtual void material( const std::string& name, const ParamSet& params )          = 0;
    virtual void makeNamedMaterial( const std::string& name, const ParamSet& params ) = 0;
    virtual void namedMaterial( const std::string& name )                             = 0;
    virtual void lightSource( const std::string& name, const ParamSet& params )       = 0;
    virtual void areaLightSource( const std::string& name, const ParamSet& params )   = 0;
    virtual void shape( const std::string& name, const ParamSet& params )             = 0;
    virtual void reverseOrientation()                                                 = 0;
    virtual void objectBegin( const std::string& name )                               = 0;
    virtual void objectEnd()                                                          = 0;
    virtual void objectInstance( const std::string& name )                            = 0;
    virtual void worldEnd()                                                           = 0;

    // Error handling during parse
    virtual void info( std::string text, const char* file, int line ) const    = 0;
    virtual void warning( std::string text, const char* file, int line ) const = 0;
    virtual void error( std::string text, const char* file, int line ) const   = 0;
};

// Global instance of the API class that is used by the global API functions.
void setApi( Api* api );
Api* getApi();

}  // namespace pbrt
}  // namespace otk
