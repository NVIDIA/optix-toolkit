#!/usr/bin/env python3

#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import optix
import os
import cupy as cp   # CUDA bindings
import numpy as np  # Packing of structures in C-compatible format

import array
import ctypes       # C interop helpers
from PIL import Image, ImageOps # Image IO
from pynvrtc.compiler import Program

import path_util


#-------------------------------------------------------------------------------
#
# Util 
#
#-------------------------------------------------------------------------------
pix_width  = 1024
pix_height =  768

class Logger:
    def __init__( self ):
        self.num_mssgs = 0

    def __call__( self, level, tag, mssg ):
        print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )
        self.num_mssgs += 1


def log_callback( level, tag, mssg ):
    print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )


def round_up( val, mult_of ):
    return val if val % mult_of == 0 else val + mult_of - val % mult_of 


def  get_aligned_itemsize( formats, alignment ):
    names = []
    for i in range( len(formats ) ):
        names.append( 'x'+str(i) )

    temp_dtype = np.dtype( { 
        'names'   : names,
        'formats' : formats, 
        'align'   : True
        } )
    return round_up( temp_dtype.itemsize, alignment )


def optix_version_gte( version ):
    if optix.version()[0] >  version[0]:
        return True
    if optix.version()[0] == version[0] and optix.version()[1] >= version[1]:
        return True
    return False


def array_to_device_memory( numpy_array, stream=cp.cuda.Stream() ):

    byte_size = numpy_array.size*numpy_array.dtype.itemsize

    h_ptr = ctypes.c_void_p( numpy_array.ctypes.data )
    d_mem = cp.cuda.memory.alloc( byte_size )
    d_mem.copy_from_async( h_ptr, byte_size, stream )
    return d_mem


def compile_cuda( cuda_file ):
    with open( cuda_file, 'rb' ) as f:
        src = f.read()
    nvrtc_dll = os.environ.get('NVRTC_DLL')
    if nvrtc_dll is None:
        nvrtc_dll = ''
    print("NVRTC_DLL = {}".format(nvrtc_dll))
    prog = Program( src.decode(), cuda_file,
                    lib_name= nvrtc_dll )
    compile_options = [
        '-use_fast_math', 
        '-lineinfo',
        '-default-device',
        '-std=c++11',
        '-rdc',
        'true',
        #'-IC:\\Program Files\\NVIDIA GPU Computing Toolkit\CUDA\\v11.1\include'
        f'-I{path_util.cuda_tk_path}',
        f'-I{path_util.include_path}'
    ]
    # Optix 7.0 compiles need path to system stddef.h
    # the value of optix.stddef_path is compiled in constant. When building
    # the module, the value can be specified via an environment variable, e.g.
    #   export PYOPTIX_STDDEF_DIR="/usr/include/linux"
    if (optix.version()[1] == 0):
        compile_options.append( f'-I{path_util.stddef_path}' )

    ptx  = prog.compile( compile_options )
    return ptx


#-------------------------------------------------------------------------------
#
# Optix setup
#
#-------------------------------------------------------------------------------


def create_ctx():
    print( "Creating optix device context ..." )

    # Note that log callback data is no longer needed.  We can
    # instead send a callable class instance as the log-function
    # which stores any data needed
    global logger
    logger = Logger()
    
    # OptiX param struct fields can be set with optional
    # keyword constructor arguments.
    ctx_options = optix.DeviceContextOptions( 
            logCallbackFunction = logger,
            logCallbackLevel    = 4
            )

    # They can also be set and queried as properties on the struct
    if optix.version()[1] >= 2:
        ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL 

    cu_ctx = 0 
    return optix.deviceContextCreate( cu_ctx, ctx_options )


def create_accel( ctx ):

    accel_options = optix.AccelBuildOptions(
        buildFlags = int( optix.BUILD_FLAG_ALLOW_COMPACTION ),
        operation = optix.BUILD_OPERATION_BUILD
        )

    aabb = cp.array( [ 
        -1.5, -1.5, -1.5, 
        1.5, 1.5, 1.5 ], 
        dtype = 'f4' )

    aabb_input_flags = [ optix.GEOMETRY_FLAG_NONE ]
    aabb_input = optix.BuildInputCustomPrimitiveArray(
        aabbBuffers = [ aabb.data.ptr ],
        numPrimitives = 1,                  
        flags = aabb_input_flags,
        numSbtRecords = 1,                  
        )

    gas_buffer_sizes = ctx.accelComputeMemoryUsage( [accel_options], [aabb_input] )

    d_temp_buffer_gas = cp.cuda.alloc( gas_buffer_sizes.tempSizeInBytes )
    d_gas_output_buffer = cp.cuda.alloc( gas_buffer_sizes.outputSizeInBytes )

    gas_handle = ctx.accelBuild(
        0,  # CUDA stream
        [ accel_options ],
        [ aabb_input ],
        d_temp_buffer_gas.ptr,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer.ptr,
        gas_buffer_sizes.outputSizeInBytes,
        [] # emitted properties
        )

    return ( gas_handle, d_gas_output_buffer )


def set_pipeline_options():
    if optix.version()[1] >= 2:
        return optix.PipelineCompileOptions(
            usesMotionBlur      = False,
            traversableGraphFlags = int( optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS ),
            numPayloadValues = 3,
            numAttributeValues = 4,
            exceptionFlags = int( optix.EXCEPTION_FLAG_NONE ),
            pipelineLaunchParamsVariableName = "params",
            usesPrimitiveTypeFlags = optix.PRIMITIVE_TYPE_FLAGS_CUSTOM
        )
    else:
        return optix.PipelineCompileOptions(
            usesMotionBlur      = False,
            traversableGraphFlags = int( optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS ),
            numPayloadValues = 3,
            numAttributeValues = 4,
            exceptionFlags = int( optix.EXCEPTION_FLAG_NONE ),
            pipelineLaunchParamsVariableName = "params",
        )


def create_module( ctx, pipeline_options, sphere_ptx ):
    print( "Creating OptiX module ..." )

    module_options = optix.ModuleCompileOptions(
        maxRegisterCount = optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT,
        debugLevel       = optix.COMPILE_DEBUG_LEVEL_DEFAULT
        )

    module, log = ctx.moduleCreateFromPTX(
        module_options,
        pipeline_options,
        sphere_ptx
        )
    print( "\tModule create log: <<<{}>>>".format( log ) )
    return module


def create_program_groups( ctx, module ):
    print( "Creating program groups ... " )

    raygen_program_desc                         = optix.ProgramGroupDesc()
    raygen_program_desc.raygenModule            = module
    raygen_program_desc.raygenEntryFunctionName = "__raygen__rg"
    raygen_prog_group, log = ctx.programGroupCreate(
        [ raygen_program_desc ]
        )
    print( "\tProgramGroup raygen create log: <<<{}>>>".format( log ) )

    miss_prog_group_desc                        = optix.ProgramGroupDesc()
    miss_prog_group_desc.missModule             = module
    miss_prog_group_desc.missEntryFunctionName  = "__miss__ms"
    miss_prog_group, log = ctx.programGroupCreate(
        [ miss_prog_group_desc ]
        )
    print( "\tProgramGroup mis create log: <<<{}>>>".format( log ) )

    hitgroup_prog_group_desc                             = optix.ProgramGroupDesc()
    hitgroup_prog_group_desc.hitgroupModuleCH            = module
    hitgroup_prog_group_desc.hitgroupEntryFunctionNameCH = "__closesthit__ch"
    hitgroup_prog_group_desc.hitgroupModuleIS            = module
    hitgroup_prog_group_desc.hitgroupEntryFunctionNameIS = "__intersection__sphere"
    hitgroup_prog_group, log = ctx.programGroupCreate(
        [ hitgroup_prog_group_desc ]
        )
    print( "\tProgramGroup hitgroup create log: <<<{}>>>".format( log ) )

    return [ raygen_prog_group, miss_prog_group, hitgroup_prog_group ]


def create_pipeline( ctx, program_groups, pipeline_compile_options ):
    print( "Creating pipeline ... " )

    max_trace_depth = 1
    pipeline_link_options                   = optix.PipelineLinkOptions()
    pipeline_link_options.maxTraceDepth     = max_trace_depth
    pipeline_link_options.debugLevel        = optix.COMPILE_DEBUG_LEVEL_FULL

    log = ""
    pipeline = ctx.pipelineCreate(
        pipeline_compile_options,
        pipeline_link_options,
        program_groups,
        log)

    stack_sizes = optix.StackSizes()
    for prog_group in program_groups:
        optix.util.accumulateStackSizes( prog_group, stack_sizes )

    ( dc_stack_size_from_trav, dc_stack_size_from_state, cc_stack_size ) = \
        optix.util.computeStackSizes(
            stack_sizes,
            max_trace_depth,
            0,  # maxCCDepth
            0   # maxDCDepth
            )

    pipeline.setStackSize(
            dc_stack_size_from_trav,
            dc_stack_size_from_state,
            cc_stack_size,
            1   # maxTraversableDepth
            )

    return pipeline


def create_sbt( prog_groups ):
    print( "Creating sbt ... " )

    (raygen_prog_group, miss_prog_group, hitgroup_prog_group ) = prog_groups

    header_format = '{}B'.format( optix.SBT_RECORD_HEADER_SIZE )

    #
    # raygen record
    #
    formats = [ header_format, 'f4','f4','f4',
                               'f4','f4','f4',
                               'f4','f4','f4',
                               'f4','f4','f4' ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( {
        'names'     : ['header', 'eye_x','eye_y','eye_z',
                                 'u_x', 'u_y', 'u_z',
                                 'v_x', 'v_y', 'v_z',
                                 'w_x', 'w_y', 'w_z'
                                 ],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'align'     : True  
        })
    h_raygen_sbt = np.array( [ ( 0, 0.0, 0.0, 3.0,
                                    2.31, -0.0, 0.0,
                                    0.0, 1.73, 0.0,
                                    0.0, 0.0, -3.0  
                               ) ], dtype = dtype )
    optix.sbtRecordPackHeader( raygen_prog_group, h_raygen_sbt )
    global d_raygen_sbt
    d_raygen_sbt = array_to_device_memory( h_raygen_sbt )

    #
    # miss record
    #
    formats = [ header_format, 'f4', 'f4', 'f4' ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( {
        'names'     : ['header', 'r', 'g', 'b' ],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'align'     : True  
        })
    h_miss_sbt = np.array( [ (0, 0.3, 0.1, 0.2) ], dtype = dtype )
    optix.sbtRecordPackHeader( miss_prog_group, h_miss_sbt )
    global d_miss_sbt
    d_miss_sbt = array_to_device_memory( h_miss_sbt )

    #
    # hitgroup record
    #
    formats = [ header_format, 'f4', 'f4', 'f4', 'f4' ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( {
        'names'     : ['header', 'center_x', 'center_y', 'center_z', 'radius' ],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'align'     : True  
        } )
    h_hitgroup_sbt = np.array( [ ( 0, 0.0, 0.0, 0.0, 1.5) ], dtype=dtype )
    optix.sbtRecordPackHeader( hitgroup_prog_group, h_hitgroup_sbt )
    global d_hitgroup_sbt
    d_hitgroup_sbt = array_to_device_memory( h_hitgroup_sbt )

    return optix.ShaderBindingTable(
        raygenRecord                = d_raygen_sbt.ptr,
        missRecordBase              = d_miss_sbt.ptr,
        missRecordStrideInBytes     = h_miss_sbt.dtype.itemsize,
        missRecordCount             = 1,
        hitgroupRecordBase          = d_hitgroup_sbt.ptr,
        hitgroupRecordStrideInBytes = h_hitgroup_sbt.dtype.itemsize,
        hitgroupRecordCount         = 1
        )


def launch( pipeline, sbt, trav_handle ):
    print( "Launching ... " )

    pix_bytes = pix_width*pix_height*4

    h_pix = np.zeros( (pix_width, pix_height, 4 ), 'B' )
    h_pix[0:pix_width, 0:pix_height] = [255, 128, 0, 255]
    d_pix = cp.array( h_pix )

    params = [
        ( 'u8', 'image',        d_pix.data.ptr ),
        ( 'u4', 'image_width',  pix_width      ),
        ( 'u4', 'image_height', pix_height     ),
        ( 'u4', 'origin_x',     pix_width / 2  ),
        ( 'u4', 'origin_y',     pix_height / 2 ),
        ( 'u8', 'trav_handle',  trav_handle    )
    ]

    formats = [ x[0] for x in params ]
    names   = [ x[1] for x in params ]
    values  = [ x[2] for x in params ]
    itemsize = get_aligned_itemsize( formats, 8 )
    params_dtype = np.dtype( {
        'names'   : names,
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True 
        } )
    h_params = np.array( [ tuple(values) ], dtype=params_dtype )
    d_params = array_to_device_memory( h_params )

    stream = cp.cuda.Stream()
    optix.launch(
        pipeline,
        stream.ptr,
        d_params.ptr,
        h_params.dtype.itemsize,
        sbt,
        pix_width,
        pix_height,
        1 # depth
        )
    
    stream.synchronize()

    h_pix = cp.asnumpy( d_pix )
    return h_pix


#-------------------------------------------------------------------------------
#
# main
#
#-------------------------------------------------------------------------------


def main():
    sphere_cu = os.path.join(os.path.dirname(__file__), 'sphere.cu')
    sphere_ptx = compile_cuda( sphere_cu )

    ctx                             = create_ctx()
    gas_handle, d_gas_output_buffer = create_accel(ctx)
    pipeline_options                = set_pipeline_options()
    module                          = create_module( ctx, pipeline_options, sphere_ptx )
    prog_groups                     = create_program_groups( ctx, module )
    pipeline                        = create_pipeline( ctx, prog_groups, pipeline_options )
    sbt                             = create_sbt( prog_groups )
    pix                             = launch( pipeline, sbt, gas_handle )

    print( "Total number of log messages: {}".format( logger.num_mssgs ) )

    pix = pix.reshape( ( pix_height, pix_width, 4 ) )   # PIL expects [ y, x ] resolution
    img = ImageOps.flip( Image.fromarray(pix , 'RGBA' ) ) # PIL expects y = 0 at bottom
    img.show()
    img.save( 'my.png' )

if __name__ == "__main__":
    main()
