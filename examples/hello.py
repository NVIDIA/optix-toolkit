#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.



import optix
import os
import cupy  as cp    # CUDA bindings
import numpy as np    # Packing of structures in C-compatible format

import array
import ctypes         # C interop helpers
from PIL import Image # Image IO
from pynvrtc.compiler import Program

import path_util


#-------------------------------------------------------------------------------
#
# Util 
#
#-------------------------------------------------------------------------------
pix_width  = 512
pix_height = 512

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
        f'-I{path_util.include_path}',
        f'-I{path_util.cuda_tk_path}'
    ]
    print("pynvrtc compile options = {}".format(compile_options))

    # Optix 7.0 compiles need path to system stddef.h
    # the value of optix.stddef_path is compiled in constant. When building
    # the module, the value can be specified via an environment variable, e.g.
    #   export PYOPTIX_STDDEF_DIR="/usr/include/linux"
    if not optix_version_gte( (7,1) ):
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


def set_pipeline_options():
    return optix.PipelineCompileOptions(
        usesMotionBlur        = False,
        traversableGraphFlags = 
            optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        numPayloadValues      = 2,
        numAttributeValues    = 2,
        exceptionFlags        = optix.EXCEPTION_FLAG_NONE,
        pipelineLaunchParamsVariableName = "params"
        )


def create_module( ctx, pipeline_options, hello_ptx ):
    print( "Creating optix module ..." )
    
    formats = ['u8', 'u4']
    itemsize = get_aligned_itemsize( formats, 16 )
    params_dtype = np.dtype( { 
        'names'   : ['image', 'image_width' ],
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True
        } )

    if optix_version_gte( (7,2) ):
        bound_value = array.array( 'i', [pix_width] )
        bound_value_entry = optix.ModuleCompileBoundValueEntry(
            pipelineParamOffsetInBytes = params_dtype.fields['image_width'][1],
            boundValue  = bound_value,
            annotation  = "my_bound_value"
        )

        module_options = optix.ModuleCompileOptions(
            maxRegisterCount = optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT,
            boundValues      = [ bound_value_entry ],
            debugLevel       = optix.COMPILE_DEBUG_LEVEL_DEFAULT
        )
    else:
        module_options = optix.ModuleCompileOptions(
            maxRegisterCount = optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT,
            debugLevel       = optix.COMPILE_DEBUG_LEVEL_DEFAULT
        )

    module, log = ctx.moduleCreateFromPTX(
            module_options,
            pipeline_options,
            hello_ptx
            )
    print( "\tModule create log: <<<{}>>>".format( log ) )
    return module


def create_program_groups( ctx, module ):
    print( "Creating program groups ... " )

    raygen_prog_group_desc                          = optix.ProgramGroupDesc()
    raygen_prog_group_desc.raygenModule             = module
    raygen_prog_group_desc.raygenEntryFunctionName  = "__raygen__hello"
    
    log = None
    raygen_prog_group = None
    if optix_version_gte( (7,4) ):
        #  ProgramGroupOptions introduced in OptiX 7.4
        program_group_options = optix.ProgramGroupOptions() 
        raygen_prog_group, log = ctx.programGroupCreate(
                [ raygen_prog_group_desc ],
                program_group_options,
                )
    else:
        raygen_prog_group, log = ctx.programGroupCreate(
                [ raygen_prog_group_desc ]
                )
    print( "\tProgramGroup raygen create log: <<<{}>>>".format( log ) )

    miss_prog_group_desc  = optix.ProgramGroupDesc( missEntryFunctionName = "")
    program_group_options = optix.ProgramGroupOptions() 
    miss_prog_group, log = ctx.programGroupCreate(
            [ miss_prog_group_desc ]
            # Even in 7.4+, the OptixProgramGroupOptions param is optional
            )
    print( "\tProgramGroup miss create log: <<<{}>>>".format( log ) )

    return ( raygen_prog_group, miss_prog_group )


def create_pipeline( ctx, raygen_prog_group, pipeline_compile_options ):
    print( "Creating pipeline ... " )
    max_trace_depth  = 0
    program_groups = [ raygen_prog_group ]

    pipeline_link_options               = optix.PipelineLinkOptions() 
    pipeline_link_options.maxTraceDepth = max_trace_depth
    pipeline_link_options.debugLevel    = optix.COMPILE_DEBUG_LEVEL_FULL

    log = ""
    pipeline = ctx.pipelineCreate(
            pipeline_compile_options,
            pipeline_link_options,
            program_groups,
            log
            )

    stack_sizes = optix.StackSizes()
    for prog_group in program_groups:
        optix.util.accumulateStackSizes( prog_group, stack_sizes )

    (dc_stack_size_from_trav, dc_stack_size_from_state, cc_stack_size) = \
        optix.util.computeStackSizes( 
            stack_sizes, 
            0,  # maxTraceDepth
            0,  # maxCCDepth
            0   # maxDCDepth
            )
    
    pipeline.setStackSize( 
            dc_stack_size_from_trav,
            dc_stack_size_from_state, 
            cc_stack_size,
            2  # maxTraversableDepth
            )

    return pipeline


def create_sbt( raygen_prog_group, miss_prog_group ):
    print( "Creating sbt ... " )

    global d_raygen_sbt
    global d_miss_sbt

    header_format = '{}B'.format( optix.SBT_RECORD_HEADER_SIZE )

    #
    # raygen record
    #
    formats  = [ header_format, 'f4', 'f4', 'f4' ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( { 
        'names'   : ['header', 'r', 'g', 'b' ],
        'formats' : formats, 
        'itemsize': itemsize,
        'align'   : True
        } )
    h_raygen_sbt = np.array( [ (0, 0.462, 0.725, 0.0 ) ], dtype=dtype )
    optix.sbtRecordPackHeader( raygen_prog_group, h_raygen_sbt )
    d_raygen_sbt = array_to_device_memory( h_raygen_sbt )
    
    #
    # miss record
    #
    formats  = [ header_format, 'i4']
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( { 
        'names'   : ['header', 'x' ],
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True
        } )
    h_miss_sbt = np.array( [ (0, 127 ) ], dtype=dtype )
    optix.sbtRecordPackHeader( miss_prog_group, h_miss_sbt )
    d_miss_sbt = array_to_device_memory( h_miss_sbt )
    
    sbt = optix.ShaderBindingTable()
    sbt.raygenRecord                = d_raygen_sbt.ptr
    sbt.missRecordBase              = d_miss_sbt.ptr
    sbt.missRecordStrideInBytes     = h_miss_sbt.dtype.itemsize
    sbt.missRecordCount             = 1
    return sbt


def launch( pipeline, sbt ):
    print( "Launching ... " )

    pix_bytes  = pix_width*pix_height*4
    
    h_pix = np.zeros( (pix_width,pix_height,4), 'B' )
    h_pix[0:256, 0:256] = [255, 128, 0, 255]
    d_pix = cp.array( h_pix )

    formats = ['u8', 'u4']
    itemsize = get_aligned_itemsize( formats, 8 )
    params_dtype = np.dtype( { 
        'names'   : ['image', 'image_width' ],
        'formats' : formats,
        'itemsize': itemsize,
        'align'   : True
        } )
    h_params = np.array( [ ( d_pix.data.ptr, pix_width ) ], dtype=params_dtype )
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
    ctx              = create_ctx()

    hello_cu = os.path.join(os.path.dirname(__file__), 'hello.cu')
    hello_ptx = compile_cuda(hello_cu)
    pipeline_options = set_pipeline_options()
    module           = create_module( ctx, pipeline_options, hello_ptx )

    raygen_prog_group, miss_prog_group = create_program_groups( ctx, module )
    pipeline         = create_pipeline( ctx, raygen_prog_group, pipeline_options )
    sbt              = create_sbt( raygen_prog_group, miss_prog_group ) 
    pix              = launch( pipeline, sbt ) 

    print( "Total number of log messages: {}".format( logger.num_mssgs ) )

    img = Image.fromarray( pix, 'RGBA' )
    img.save( 'my.png' )
    img.show()


if __name__ == "__main__":
    main()
