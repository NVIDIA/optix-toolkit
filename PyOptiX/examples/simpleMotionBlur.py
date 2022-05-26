#!/usr/bin/env python3

#
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import cupy  as cp    # CUDA bindings
import numpy as np    # Packing of structures in C-compatible format

import array
import ctypes         # C interop helpers
from PIL import Image, ImageOps # Image IO
from pynvrtc.compiler import Program

import path_util



class State:
    def __init__( self ):
        self.context                        = None

        self.tri_gas_handle                 = 0 
        self.d_tri_gas_output_buffer        = 0 # Triangle AS memory

        self.sphere_gas_handle              = 0 # Traversable handle for sphere
        self.d_sphere_gas_output_buffer     = 0 # Sphere AS memory
        self.sphere_motion_transform_handle = 0
        self.d_sphere_motion_transform      = 0

        self.ias_handle                     = 0 # Traversable handle for instance AS
        self.d_ias_output_buffer            = 0 # Instance AS memory

        self.ptx_module                     = None 
        self.pipeline_compile_options       = None
        self.pipeline                       = None 

        self.raygen_prog_group              = None
        self.miss_group                     = None
        self.tri_hit_group                  = None
        self.sphere_hit_group               = None

        self.stream                         = stream=cp.cuda.Stream()
        self.params                         = None
        self.d_params                       = 0

        self.sbt                            = None
        self.d_raygen_record                = 0                 
        self.d_miss_records                 = 0                 
        self.d_hitgroup_records             = 0                 


#-------------------------------------------------------------------------------
#
# Util 
#
#-------------------------------------------------------------------------------


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
        'aligned'   : True
        } )
    return round_up( temp_dtype.itemsize, alignment )


def array_to_device_memory( numpy_array, stream=cp.cuda.Stream() ):

    byte_size = numpy_array.size*numpy_array.dtype.itemsize

    h_ptr = ctypes.c_void_p( numpy_array.ctypes.data )
    d_mem = cp.cuda.memory.alloc( byte_size )
    d_mem.copy_from_async( h_ptr, byte_size, stream )
    return d_mem


def optix_version_gte( version ):
    if optix.version()[0] >  version[0]:
        return True
    if optix.version()[0] == version[0] and optix.version()[1] >= version[1]:
        return True
    return False


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

pix_width = 768
pix_height = 768


def create_context( state ):
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
    if optix_version_gte( (7,2) ):
        ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL 

    cu_ctx = 0 
    state.context = optix.deviceContextCreate( cu_ctx, ctx_options )


def build_triangle_gas( state ):

    NUM_KEYS = 3

    motion_options           = optix.MotionOptions()
    motion_options.numKeys   = NUM_KEYS
    motion_options.timeBegin = 0.0
    motion_options.timeEnd   = 1.0
    motion_options.flags     = optix.MOTION_FLAG_NONE

    accel_options = optix.AccelBuildOptions(
        buildFlags = optix.BUILD_FLAG_ALLOW_COMPACTION,
        operation  = optix.BUILD_OPERATION_BUILD,
        motionOptions = motion_options
        )    

    #
    # Copy triangle mesh data to device 
    #
    NUM_VERTS = 3
    vertices_0 = cp.array( [ 
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.5, 1.0, 0.0, 0.0,
        ], dtype = 'f4' )

    vertices_1 = cp.array( [
            0.5, 0.0, 0.0, 0.0,
            1.5, 0.0, 0.0, 0.0,
            1.0, 1.0, 0.0, 0.0,
        ], 
        dtype = 'f4' 
    )
    vertices_2 = cp.array( [
            0.5, -0.5, 0.0, 0.0,
            1.5, -0.5, 0.0, 0.0,
            1.0,  0.5, 0.0, 0.0
        ], 
        dtype = 'f4' 
    )


    triangle_input                      = optix.BuildInputTriangleArray()
    triangle_input.vertexFormat         = optix.VERTEX_FORMAT_FLOAT3
    triangle_input.vertexStrideInBytes  = np.dtype( 'f4' ).itemsize*4 # four floats per vert 
    triangle_input.numVertices          = NUM_VERTS
    triangle_input.vertexBuffers        = [ vertices_0.data.ptr, vertices_1.data.ptr, vertices_2.data.ptr ]
    triangle_input.flags                = [ optix.GEOMETRY_FLAG_DISABLE_ANYHIT ]
    triangle_input.numSbtRecords        = 1
    triangle_input.sbtIndexOffsetBuffer = 0

    gas_buffer_sizes = state.context.accelComputeMemoryUsage( 
        [ accel_options  ], 
        [ triangle_input ]
    )

    d_temp_buffer    = cp.cuda.alloc( gas_buffer_sizes.tempSizeInBytes   )
    d_output_buffer  = cp.cuda.alloc( gas_buffer_sizes.outputSizeInBytes ) 
    d_result         = cp.array( [ 0 ], dtype = 'u8' )

    emit_property = optix.AccelEmitDesc(
        type = optix.PROPERTY_TYPE_COMPACTED_SIZE,
        result = d_result.data.ptr
        )

    state.tri_gas_handle = state.context.accelBuild(
        0,  # CUDA stream
        [ accel_options ],
        [ triangle_input ],
        d_temp_buffer.ptr,
        gas_buffer_sizes.tempSizeInBytes,
        d_output_buffer.ptr,
        gas_buffer_sizes.outputSizeInBytes,
        [ emit_property ]
    )

    compacted_gas_size = cp.asnumpy( d_result )[0]

    if compacted_gas_size < gas_buffer_sizes.outputSizeInBytes and False:

        state.d_tri_gas_output_buffer = cp.cuda.alloc( compacted_gas_size )
        state.tri_gas_handle = state.context.accelCompact( 
            0,  #CUDA stream
            state.tri_gas_handle,
            state.d_tri_gas_output_buffer.ptr,
            compacted_gas_size
        )
    else:
        state.d_tri_gas_output_buffer = d_output_buffer 


def build_sphere_gas( state ):

    accel_options = optix.AccelBuildOptions(
        buildFlags = optix.BUILD_FLAG_ALLOW_COMPACTION,
        operation  = optix.BUILD_OPERATION_BUILD
    )  

    aabb = cp.array( [ 
        -1.5, -1.0, -0.5,
        -0.5,  0.0,  0.5
        #-1.0, -1.0, -1.0,
        # 1.0,  1.0,  1.0
        ], dtype = 'f4'
    )

    sphere_input = optix.BuildInputCustomPrimitiveArray(
        aabbBuffers    = [ aabb.data.ptr ],
        numPrimitives  = 1,
        #flags          = [ optix.GEOMETRY_FLAG_DISABLE_ANYHIT ],
        flags          = [ optix.GEOMETRY_FLAG_NONE],
        numSbtRecords  = 1
    )

    gas_buffer_sizes = state.context.accelComputeMemoryUsage( 
        [ accel_options ], 
        [ sphere_input  ] 
    )

    d_temp_buffer    = cp.cuda.alloc( gas_buffer_sizes.tempSizeInBytes )
    d_output_buffer  = cp.cuda.alloc( gas_buffer_sizes.outputSizeInBytes )
    d_result         = cp.array( [ 0 ], dtype = 'u8' )

    emit_property = optix.AccelEmitDesc(
        type = optix.PROPERTY_TYPE_COMPACTED_SIZE,
        result = d_result.data.ptr
    )

    state.sphere_gas_handle = state.context.accelBuild(
        0,  # CUDA stream
        [ accel_options ],
        [ sphere_input ],
        d_temp_buffer.ptr,
        gas_buffer_sizes.tempSizeInBytes,
        d_output_buffer.ptr,
        gas_buffer_sizes.outputSizeInBytes,
        [ emit_property ]
    )

    compacted_gas_size = cp.asnumpy( d_result )[0]

    if compacted_gas_size < gas_buffer_sizes.outputSizeInBytes and False:
        state.d_sphere_gas_output_buffer = cp.cuda.alloc( compacted_gas_size )
        state.sphere_gas_handle = state.context.accelCompact( 
            0,  #CUDA stream
            state.sphere_gas_handle,
            state.d_sphere_gas_output_buffer,
            compacted_gas_size
        )
    else:
        state.d_sphere_gas_output_buffer = d_output_buffer 


def create_sphere_xform( state ):

    motion_keys = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,

        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.5,
        0.0, 0.0, 1.0, 0.0
    ]

    motion_options           = optix.MotionOptions()
    motion_options.numKeys   = 2
    motion_options.timeBegin = 0.0
    motion_options.timeEnd   = 1.0
    motion_options.flags     = optix.MOTION_FLAG_NONE

    motion_transform  = optix.MatrixMotionTransform( 
        child         = state.sphere_gas_handle,
        motionOptions = motion_options,
        transform     = motion_keys
        )

    xform_bytes = optix.getDeviceRepresentation( motion_transform )
    state.d_sphere_motion_transform = cp.array( np.frombuffer( xform_bytes, dtype='B' ) )

    state.sphere_motion_transform_handle = optix.convertPointerToTraversableHandle(
        state.context,
        state.d_sphere_motion_transform.data.ptr,
        optix.TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM
    )


def build_ias( state ):

    instance_xform = [ 
        1.0, 0.0, 0.0, 0.0, 
        0.0, 1.0, 0.0, 0.0, 
        0.0, 0.0, 1.0, 0.0 
    ]

    sphere_instance = optix.Instance( 
        transform         = instance_xform,
        flags             = optix.INSTANCE_FLAG_NONE,
        instanceId        = 0,
        sbtOffset         = 0,
        visibilityMask    = 1,
        traversableHandle = state.sphere_motion_transform_handle
    )

    triangle_instance = optix.Instance(
        transform         = instance_xform,
        flags             = optix.INSTANCE_FLAG_NONE,
        instanceId        = 1,
        sbtOffset         = 1,
        visibilityMask    = 1,
        traversableHandle = state.tri_gas_handle 
    )

    instances       = [ sphere_instance, triangle_instance ]
    instances_bytes = optix.getDeviceRepresentation( instances ) 
    d_instances     = cp.array( np.frombuffer( instances_bytes, dtype='B' ) )

    instance_input = optix.BuildInputInstanceArray(
        instances    = d_instances.data.ptr,
        numInstances = len( instances ) 
    )
    
    motion_options           = optix.MotionOptions()
    motion_options.numKeys   = 2 
    motion_options.timeBegin = 0.0
    motion_options.timeEnd   = 1.0
    motion_options.flags     = optix.MOTION_FLAG_NONE

    accel_options = optix.AccelBuildOptions(
        buildFlags    = optix.BUILD_FLAG_NONE,
        operation     = optix.BUILD_OPERATION_BUILD,
        motionOptions = motion_options
        )    

    ias_buffer_sizes    = state.context.accelComputeMemoryUsage( 
        [ accel_options  ], 
        [ instance_input ]
    )
    d_temp_buffer             = cp.cuda.alloc( ias_buffer_sizes.tempSizeInBytes   ) 
    state.d_ias_output_buffer = cp.cuda.alloc( ias_buffer_sizes.outputSizeInBytes )

    state.ias_handle = state.context.accelBuild(
        0,    # CUDA stream
        [ accel_options  ], 
        [ instance_input ],   
        d_temp_buffer.ptr,
        ias_buffer_sizes.tempSizeInBytes,
        state.d_ias_output_buffer.ptr,
        ias_buffer_sizes.outputSizeInBytes,
        [] # emitted properties
        )


def create_module( state ):

    module_compile_options                  = optix.ModuleCompileOptions()
    module_compile_options.maxRegisterCount = optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT
    module_compile_options.optLevel         = optix.COMPILE_OPTIMIZATION_DEFAULT
    module_compile_options.debugLevel       = optix.COMPILE_DEBUG_LEVEL_DEFAULT

    state.pipeline_compile_options = optix.PipelineCompileOptions(
        traversableGraphFlags            = optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
        numPayloadValues                 = 3,
        numAttributeValues               = 3,
        usesMotionBlur                   = True,
        exceptionFlags                   = optix.EXCEPTION_FLAG_NONE,
        pipelineLaunchParamsVariableName = "params"
    )

    simple_motion_blur_cu = os.path.join(os.path.dirname(__file__), 'simpleMotionBlur.cu')
    simple_motion_blur_ptx = compile_cuda( simple_motion_blur_cu )

    state.ptx_module, log = state.context.moduleCreateFromPTX(
        module_compile_options,
        state.pipeline_compile_options,
        simple_motion_blur_ptx
    )


def create_program_groups( state ):

    
    raygen_program_group_desc = optix.ProgramGroupDesc(
        raygenModule            = state.ptx_module,
        raygenEntryFunctionName = "__raygen__rg"
    )
    state.raygen_prog_group, log = state.context.programGroupCreate(
        [ raygen_program_group_desc ]
    )
    print( "\tProgramGroup raygen create log: <<<{}>>>".format( log ) )

    miss_prog_group_desc = optix.ProgramGroupDesc(
        missModule             = state.ptx_module,
        missEntryFunctionName  = "__miss__camera"
    )
    state.miss_group, log = state.context.programGroupCreate(
        [ miss_prog_group_desc ]
    )
    print( "\tProgramGroup miss create log: <<<{}>>>".format( log ) )

    hitgroup_prog_group_desc = optix.ProgramGroupDesc(
        hitgroupModuleCH            = state.ptx_module,
        hitgroupEntryFunctionNameCH = "__closesthit__camera",
    )
    state.tri_hit_group, log = state.context.programGroupCreate(
        [ hitgroup_prog_group_desc ]
    )
    print( "\tProgramGroup triangle hit create log: <<<{}>>>".format( log ) )
        
    hitgroup_prog_group_desc.hitgroupModuleIS            = state.ptx_module
    hitgroup_prog_group_desc.hitgroupEntryFunctionNameIS = "__intersection__sphere"
    state.sphere_hit_group, log = state.context.programGroupCreate(
        [ hitgroup_prog_group_desc ]
    )
    print( "\tProgramGroup sphere hit create log: <<<{}>>>".format( log ) )


def create_pipeline( state ):

    program_groups = [
        state.raygen_prog_group,
        state.miss_group,
        state.sphere_hit_group,
        state.tri_hit_group
    ]

    pipeline_link_options = optix.PipelineLinkOptions(
        maxTraceDepth = 2,
        debugLevel = optix.COMPILE_DEBUG_LEVEL_FULL
    )

    log = ""
    state.pipeline = state.context.pipelineCreate(
        state.pipeline_compile_options,
        pipeline_link_options,
        program_groups,
        log
    )

    stack_sizes = optix.StackSizes()
    for prog_group in program_groups:
        optix.util.accumulateStackSizes( prog_group, stack_sizes )

    ( dc_stack_size_from_trav, dc_stack_size_from_state, cc_stack_size ) = \
        optix.util.computeStackSizes(
            stack_sizes,
            1,  # maxTraceDepth
            0,  # maxCCDepth
            0   # maxDCDepth
        )

    state.pipeline.setStackSize(
        1024, #dc_stack_size_from_trav,
        1024, #dc_stack_size_from_state,
        1024, #cc_stack_size,
        3   # maxTraversableDepth ( 3 since largest depth is IAS->MT->GAS )
    )


def create_sbt( state ):
    print( "Creating sbt ... " )

    header_format = '{}V'.format( optix.SBT_RECORD_HEADER_SIZE )

    #
    # raygen record
    #
    formats = [ header_format ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( {
        'names'     : ['header'],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'aligned'     : True 
        } )
    h_raygen_record = np.array( 
        [ optix.sbtRecordGetHeader( state.raygen_prog_group) ], 
        dtype = dtype 
    )
    optix.sbtRecordPackHeader( state.raygen_prog_group, h_raygen_record )
    state.d_raygen_record = array_to_device_memory( h_raygen_record )

    #
    # miss records
    #
    formats = [ header_format, 'f4','f4','f4', 'u4' ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    dtype = np.dtype( {
        'names'     : [ 'header', 'r', 'g', 'b', 'pad' ],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'aligned'     : True 
        } )
    h_miss_record = np.array( [ (
            optix.sbtRecordGetHeader( state.miss_group ), 
            0.1, 0.1, 0.1, 
            0 
        ) ], 
        dtype=dtype 
    )
    optix.sbtRecordPackHeader( state.miss_group, h_miss_record )
    state.d_miss_records = array_to_device_memory( h_miss_record )

    #
    # hit group records
    #
    formats = [ 
        header_format, 
        'f4','f4','f4',
        'f4','f4','f4',
        'f4',
        'u4'
    ]
    itemsize = get_aligned_itemsize( formats, optix.SBT_RECORD_ALIGNMENT )
    hit_record_dtype = np.dtype( {
        'names' : [ 
            'header',
            'r','g','b',
            'x','y','z',
            'rad',
            'pad' 
        ],
        'formats'   : formats,
        'itemsize'  : itemsize,
        'aligned'      : True
        } )

    sphere_record_header = optix.sbtRecordGetHeader( state.sphere_hit_group )
    triangle_record_header   = optix.sbtRecordGetHeader( state.tri_hit_group )

    h_hitgroup_records = np.array( [ 
        ( 
            sphere_record_header, 
            0.9,  0.1, 0.1,
            -1.0, -0.5, 0.1,
            0.5,
            0.0
        ), 
        (
            triangle_record_header, 
            0.1, 0.1, 0.9,
            0.0, 0.0, 0.0,    # unused
            0.0,              # unused 
            0.0   
        ) ],
        dtype=hit_record_dtype 
    )

    state.d_hitgroup_records = array_to_device_memory( h_hitgroup_records )

    state.sbt = optix.ShaderBindingTable(
        raygenRecord                = state.d_raygen_record.ptr,
        missRecordBase              = state.d_miss_records.ptr,
        missRecordStrideInBytes     = h_miss_record.dtype.itemsize,
        missRecordCount             = 1,
        hitgroupRecordBase          = state.d_hitgroup_records.ptr,
        hitgroupRecordStrideInBytes = h_hitgroup_records.dtype.itemsize,
        hitgroupRecordCount         = 2 
    )


def launch( state ):
    print( "Launching ... " )

    pix_bytes = pix_width * pix_height * 4

    h_accum = np.zeros( (pix_width, pix_height, 4 ), 'f4' )
    h_accum[0:pix_width, 0:pix_height] = [255, 128, 0, 255]
    d_accum = cp.array( h_accum )

    h_frame = np.zeros( (pix_width, pix_height, 4 ), 'B' )
    h_frame[0:pix_width, 0:pix_height] = [255, 128, 0, 255]
    d_frame = cp.array( h_frame )

    params = [
        ( 'u4', 'image_width',    pix_width ),
        ( 'u4', 'image_height',   pix_height ),
        ( 'u8', 'accum',          d_accum.data.ptr ),
        ( 'u8', 'frame',          d_frame.data.ptr ),
        ( 'u4', 'subframe index', 0 ),
        ( 'f4', 'cam_eye_x',      0 ),
        ( 'f4', 'cam_eye_y',      0 ),
        ( 'f4', 'cam_eye_z',      5.0 ),
        ( 'f4', 'cam_U_x',        1.10457 ),
        ( 'f4', 'cam_U_y',        0 ),
        ( 'f4', 'cam_U_z',        0 ),
        ( 'f4', 'cam_V_x',        0 ),
        ( 'f4', 'cam_V_y',        0.828427 ),
        ( 'f4', 'cam_V_z',        0 ),
        ( 'f4', 'cam_W_x',        0 ),
        ( 'f4', 'cam_W_y',        0 ),
        ( 'f4', 'cam_W_z',        -2.0 ),
        ( 'u8', 'trav_handle',   state.ias_handle )
        #( 'u8', 'trav_handle',   state.tri_gas_handle)
    ]

    formats = [ x[0] for x in params ] 
    names   = [ x[1] for x in params ] 
    values  = [ x[2] for x in params ] 
    itemsize = get_aligned_itemsize( formats, 8 )
    params_dtype = np.dtype( { 
        'names'   : names, 
        'formats' : formats,
        'itemsize': itemsize,
        'aligned'   : True
        } )
    h_params = np.array( [ tuple(values) ], dtype=params_dtype )
    d_params = array_to_device_memory( h_params )

    stream = cp.cuda.Stream()
    optix.launch( 
        state.pipeline, 
        stream.ptr, 
        d_params.ptr, 
        h_params.dtype.itemsize, 
        state.sbt,
        pix_width,
        pix_height,
        1 # depth
        )

    stream.synchronize()

    h_pix = cp.asnumpy( d_frame )
    return h_pix



#-------------------------------------------------------------------------------
#
# main
#
#-------------------------------------------------------------------------------


def main():
    state = State()
    create_context       ( state )
    build_triangle_gas   ( state )
    build_sphere_gas     ( state )
    create_sphere_xform  ( state )
    build_ias            ( state )
    create_module        ( state )
    create_program_groups( state )
    create_pipeline      ( state )
    create_sbt           ( state )

    pix = launch( state ) 

    print( "Total number of log messages: {}".format( logger.num_mssgs ) )

    pix = pix.reshape( ( pix_height, pix_width, 4 ) )     # PIL expects [ y, x ] resolution
    img = ImageOps.flip( Image.fromarray( pix, 'RGBA' ) ) # PIL expects y = 0 at bottom
    img.show()
    img.save( 'my.png' )


if __name__ == "__main__":
    main()
