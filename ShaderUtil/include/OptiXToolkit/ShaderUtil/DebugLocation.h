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

#pragma once

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <optix.h>

#include <vector_functions.h>

namespace otk {

/// DebugLocation
///
/// A simple structure to hold flags and a debug launch index at which debug information
/// should be obtained.
///
struct DebugLocation
{
    bool  enabled;         // when true, debug location checking is enabled
    bool  dumpSuppressed;  // when true, the dump function is NOT called
    bool  debugIndexSet;   // when true, the debugIndex location is set to something meaningful
    uint3 debugIndex;      // the launchIndex at which to emit debugging information
};

#ifdef __CUDACC__

namespace debugDetail {

static __forceinline__ __device__ bool outsideWindow( unsigned int lhs, unsigned int rhs, unsigned int width )
{
    const unsigned int bigger  = max( lhs, rhs );
    const unsigned int smaller = min( lhs, rhs );
    return bigger - smaller > width;
}

static __forceinline__ __device__ bool inDebugWindow( const uint3& launchIndex, const uint3& debugIndex, unsigned int width )
{
    return !( outsideWindow( launchIndex.x, debugIndex.x, width ) || outsideWindow( launchIndex.y, debugIndex.y, width )
              || outsideWindow( launchIndex.z, debugIndex.z, width ) );
}

}  // namespace debugDetail

/// atDebugIndex
///
/// Returns true if debugging enabled, debug pixel set and it matches the given launchIndex.
/// This is useful when dumping debug information from raygen programs.
///
__forceinline__ __device__ bool atDebugIndex( const DebugLocation& debug, const uint3 launchIndex = optixGetLaunchIndex() )
{
    return debug.enabled && debug.debugIndexSet && debug.debugIndex == launchIndex;
}

/// debugInfoDump
///
/// Invoke a functor at a specific launch index for debugging purposes.  The functor takes the OptiX launch index
/// at which debugging information is desired.  Generally the functor is a lambda that captures application
/// state (or references or pointers to such state) to be dumped via printf at the specific launch index.
/// A red pixel is drawn at the debug launch index, with a 2 pixel black border around the debug launch index
/// and a further 2 pixel white border around that.  This makes a single red pixel easy to spot in the output.
/// Pixels are set via the given function pointer, which generally just sets the ray payload to the appropriate
/// values.  If no visible output is desired, then simply supply a lambda that does nothing with the 3 float values.
/// The dump function is taken as a template parameter to allow the use of lambdas that capture application state
/// as we don't have access to std::function<> to make a generic callable.  The set color routine is taken as a
/// simple function pointer because it is assumed to not need to capture any application state in order to set
/// a ray payload.
///
/// A "one-shot" debug information dump mechanism can be implemented as follows:
/// - Enable the debug dump mechanism and set the debug launch index and the set flag.
/// - Launch with dumpSuppressed set to false and the debug output will be printed
/// - Set dumpSuppressed to true and subsequent launches will display the debug location,
///   but will not emit debug output.
///
/// This allows you to get visual indication of the debug location in the rendered image without repeated
/// output of the same debug information over and over again as frames are rendered.  This is most useful in
/// an application where the rendered image only changes due to user interaction and therefore the debug output
/// is static across many frames.
///
/// @param  debug       The DebugLocation structure containing enable flags and debug launchIndex.
/// @param  dumpFn      The function to be invoked when the launch index matches.
///         setColor    The function invoked to set colors for the debug pixel and its border.
///
/// @returns            Returns true if a debug color was set at the current launch index.
///
template <typename Dump>
static __forceinline__ __device__ bool debugInfoDump( const DebugLocation& debug,
                                                      const Dump&          dumpFn,
                                                      void( setColor )( float red, float green, float blue ) )
{
    if( !debug.enabled || !debug.debugIndexSet )
    {
        return false;
    }

    const uint3 launchIndex = optixGetLaunchIndex();
    if( debug.debugIndex == launchIndex )
    {
        if( !debug.dumpSuppressed )
        {
            dumpFn( launchIndex );
        }
        setColor( 1.0f, 0.0f, 0.0f );  // red
        return true;
    }
    if( debugDetail::inDebugWindow( launchIndex, debug.debugIndex, 2 ) )
    {
        setColor( 0.0f, 0.0f, 0.0f );  // black
        return true;
    }
    if( debugDetail::inDebugWindow( launchIndex, debug.debugIndex, 4 ) )
    {
        setColor( 1.0f, 1.0f, 1.0f );  // white
        return true;
    }
    return false;
}

#endif

}  // namespace otk
