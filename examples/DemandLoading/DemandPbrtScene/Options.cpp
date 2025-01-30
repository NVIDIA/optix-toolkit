// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/Options.h"

#include "DemandPbrtScene/Params.h"

#include <OptiXToolkit/Gui/Window.h>

#include <iostream>
#include <sstream>

namespace demandPbrtScene {

[[noreturn]] static void programUsage( const char* program, const char* message )
{
    // clang-format off
    std::cerr <<
        message << '\n' <<
        "Usage:\n" <<
        program << " [options] <scene-file>\n"
        "\n"
        "<scene-file>                   Path to a .pbrt scene file; required.\n"
        "\n"
        "Options:\n"
        "   --file <file> | -f <file>   Render to <file>\n"
        "   --dim=<width>x<height>      Set image dimensions; defaults to 768x512\n"
        "   --bg=<red>/<green>/<blue>   Set image background color; defaults to black\n"
        "   --warmup=<count>            Render <count> frames before saving to file\n"
        "   --face-forward              Flip the direction of back face normals\n"
        "   --render-mode=<mode>        Specify the initial rendering mode, where <mode> is one of:\n"
        "                               primary     Use primary ray only (default)\n"
        "                               near        Use near ambient occlusion\n"
        "                               distant     Use distant ambient occlusion\n"
        "                               path        Use path tracing\n"
        "   --proxy-granularity=<mode>  Set the granularity of proxy geometry, where <mode> is one of:\n"
        "                               coarse      Use a single proxy for all Shapes in an Object\n"
        "                               fine        Use a proxy for each Shape in an Object\n"
        "   --oneshot-geometry          Enable one-shot proxy geometry resolution by keystroke\n"
        "   --oneshot-material          Enable one-shot proxy material resolution by keystroke\n"
        "   --proxy-resolution          Enable verbose logging of resolution of proxy geometries and materials\n"
        "   --proxy-geometry            Enable verbose logging of resolution of proxy geometries\n"
        "   --proxy-material            Enable verbose logging of resolution of proxy materials\n"
        "   --scene-decomposition       Enable verbose logging of scene hierarchy decomposition\n"
        "   --texture-creation          Enable verbose logging of texture creation\n"
        "   --verbose-loading           Enable verbose logging of mesh reading\n"
        "   --verbose                   Enables all verbose logging\n"
        "   --sort-proxies              Sort proxies before resolving\n"
        "   --sync                      Enable extra synchronization for debugging (off in release build)\n"
        "   --debug=<x>/<y>             Enable debug output for pixel at (x,y)\n"
        "   --oneshot-debug             Enable one-shot debug output\n"
        "\n"
        "Interactive keys (case insensitive):\n"
        "   <Space>                     Toggle pause\n"
        "   D                           Toggle debug mode\n"
        "   G                           Resolve one proxy geometry\n"
        "   M                           Resolve one proxy material\n"
        "   Q or <Esc>                  Quit\n"
        ;
    // clang-format on
    exit( 1 );
}

inline bool beginsWith( const std::string& text, const std::string& prefix )
{
    return text.substr( 0, prefix.length() ) == prefix;
}

inline std::string extractValue( const std::string& text )
{
    return text.substr( text.find_first_of( '=' ) + 1 );
}

Options parseOptions( int argc, char* argv[], const std::function<UsageFn>& usage )
{
    if( argc < 2 )
    {
        usage( argv[0], "missing scene file argument" );
    }

    Options options;
    options.program = argv[0];
#ifdef NDEBUG
    options.sync = false;
#else
    options.sync = true;
#endif
    options.proxyGranularity = ProxyGranularity::FINE;

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg{ argv[i] };
        if( arg == "-f" || arg == "--file" )
        {
            if( i + 1 < argc )
            {
                ++i;
                options.outFile = argv[i];
            }
            else
            {
                usage( argv[0], "missing filename argument" );
            }
        }
        else if( beginsWith( arg, "--dim" ) )
        {
            otk::parseDimensions( extractValue( arg ).c_str(), options.width, options.height );
        }
        else if( beginsWith( arg, "--bg" ) )
        {
            std::istringstream str( extractValue( arg ) );
            char               sep;
            float3             value;
            str >> value.x >> sep >> value.y >> sep >> value.z;
            if( !str || value.x < 0.0f || value.y < 0.0f || value.z < 0.0f )
            {
                usage( argv[0], "bad background color value" );
            }
            options.background = value;
        }
        else if( beginsWith( arg, "--warmup" ) )
        {
            std::istringstream str( extractValue( arg ) );
            int                warmup{};
            str >> warmup;
            if( !str || warmup < 0 )
            {
                usage( argv[0], "bad warmup frame count value" );
            }
            options.warmupFrames = warmup;
        }
        else if( beginsWith( arg, "--debug" ) )
        {
            std::istringstream str( extractValue( arg ) );
            char               sep;
            int2               value{};
            str >> value.x >> sep >> value.y;
            if( !str || value.x < 0 || value.y < 0 )
            {
                usage( argv[0], "bad debug pixel value" );
            }
            options.debug      = true;
            options.debugPixel = value;
        }
        else if( arg == "--oneshot-debug" )
        {
            options.oneShotDebug = true;
        }
        else if( arg == "--oneshot-geometry" )
        {
            options.oneShotGeometry = true;
        }
        else if( arg == "--oneshot-material" )
        {
            options.oneShotMaterial = true;
        }
        else if( arg == "--proxy-resolution" )
        {
            options.verboseProxyGeometryResolution = true;
            options.verboseProxyMaterialResolution = true;
        }
        else if( arg == "--proxy-geometry" )
        {
            options.verboseProxyGeometryResolution = true;
        }
        else if( arg == "--proxy-material" )
        {
            options.verboseProxyMaterialResolution = true;
        }
        else if( arg == "--scene-decomposition" )
        {
            options.verboseSceneDecomposition = true;
        }
        else if( arg == "--texture-creation" )
        {
            options.verboseTextureCreation = true;
        }
        else if( arg == "--verbose" )
        {
            options.verboseLoading                 = true;
            options.verboseProxyGeometryResolution = true;
            options.verboseProxyMaterialResolution = true;
            options.verboseSceneDecomposition      = true;
            options.verboseTextureCreation         = true;
        }
        else if( arg == "--verbose-loading" )
        {
            options.verboseLoading = true;
        }
        else if( arg == "--sort-proxies" )
        {
            options.sortProxies = true;
        }
        else if( arg == "--sync" )
        {
            options.sync = true;
        }
        else if( arg == "--face-forward" )
        {
            options.faceForward = true;
        }
        else if( beginsWith( arg, "--render-mode=" ) )
        {
            const std::string value{ extractValue( arg ) };
            if( value.empty() )
            {
                usage( argv[0], "missing render mode value" );
            }
            else if( value == "primary" )
            {
                options.renderMode = RenderMode::PRIMARY_RAY;
            }
            else if( value == "near" )
            {
                options.renderMode = RenderMode::NEAR_AO;
            }
            else if( value == "distant" )
            {
                options.renderMode = RenderMode::DISTANT_AO;
            }
            else if( value == "path" )
            {
                options.renderMode = RenderMode::PATH_TRACING;
            }
            else
            {
                usage( argv[0], ( "bad render mode value: " + value ).c_str() );
            }
        }
        else if( beginsWith( arg, "--proxy-granularity=" ) )
        {
            const std::string value{ extractValue( arg ) };
            if( value.empty() )
            {
                usage( argv[0], "missing proxy granularity value" );
            }
            else if( value == "fine" )
            {
                options.proxyGranularity = ProxyGranularity::FINE;
            }
            else if( value == "coarse" )
            {
                options.proxyGranularity = ProxyGranularity::COARSE;
            }
            else
            {
                usage( argv[0], ( "bad proxy granularity value: " + value ).c_str() );
            }
        }
        else if( arg[0] == '-' )
        {
            usage( argv[0], ( "unknown option: " + arg ).c_str() );
        }
        else if( i + 1 <= argc )
        {
            options.sceneFile = argv[i];
        }
        else
        {
            usage( argv[0], "missing scene file argument" );
        }
    }

    // Dimensions and debug pixel can come in any order, so post validate after parsing.
    if( options.debug && ( options.debugPixel.x >= options.width || options.debugPixel.y >= options.height ) )
    {
        usage( argv[0], "bad debug pixel value" );
    }

    return options;
}

Options parseOptions( int argc, char* argv[] )
{
    return parseOptions( argc, argv, programUsage );
}

}  // namespace demandPbrtScene
