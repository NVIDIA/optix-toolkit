#!/usr/bin/env python3
# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.



import optix
import Imath 
import OpenEXR
import cupy as cp 
import cupy.cuda.runtime as cuda
import numpy as np

import ctypes



#-------------------------------------------------------------------------------
#
# Helpers 
#
#-------------------------------------------------------------------------------

class Logger:
    def __init__( self ):
        self.num_mssgs = 0

    def __call__( self, level, tag, mssg ):
        print( "[{:>2}][{:>12}]: {}".format( level, tag, mssg ) )
        self.num_mssgs += 1


class State:
    def __init__( self ):
        self.tile_size    = (0, 0)
        self.exposure     = 0.0
        self.layer        = optix.DenoiserLayer()
        self.guide_layer  = optix.DenoiserGuideLayer()

        #self.scratch_size = 0 
        self.overlap      = 0 

        self.d_intensity  = 0  # 
        self.d_scratch    = 0  # CUPY RAII memory pointers
        self.d_state      = 0  # 


    def __str__( self ):
        return  (
            "w     : {}\n".format( self.layer.input.width  ) +   
            "h     : {}\n".format( self.layer.input.height ) +   
            "tile  : {}\n".format( self.tile_size          ) +   
            "expos : {}"  .format( self.exposure           )
            )


def optix_version_gte( version ):
    if optix.version()[0] >  version[0]:
        return True
    if optix.version()[0] == version[0] and optix.version()[1] >= version[1]:
        return True
    return False


def create_optix_image_2D( w, h, image ):
    oi = optix.Image2D()
    byte_size = w*h*4*4 
    d_mem = cuda.malloc( byte_size )
    if image is not None:
        cuda.memcpy( 
            d_mem, 
            image.ctypes.data, 
            byte_size, 
            cuda.memcpyHostToDevice
        )
    oi.data               = d_mem
    oi.width              = w
    oi.height             = h
    oi.rowStrideInBytes   = w*4*4
    oi.pixelStrideInBytes = 4*4
    oi.format             = optix.PIXEL_FORMAT_FLOAT4
    return oi


def free_optix_image_2D( optix_image ):
    cuda.free( optix_imae.data )
    oi.data = 0


def load_exr( filename ):
    exr_file     = OpenEXR.InputFile( filename )
    exr_header   = exr_file.header()
    r,g,b = exr_file.channels("RGB", pixel_type=Imath.PixelType(Imath.PixelType.FLOAT) )

    dw = exr_header[ "dataWindow" ]
    w  = dw.max.x - dw.min.x + 1
    h  = dw.max.y - dw.min.y + 1

    image = np.ones( (h, w, 4), dtype = np.float32 )
    image[:, :, 0] = np.core.multiarray.frombuffer( r, dtype = np.float32 ).reshape(h, w)
    image[:, :, 1] = np.core.multiarray.frombuffer( g, dtype = np.float32 ).reshape(h, w)
    image[:, :, 2] = np.core.multiarray.frombuffer( b, dtype = np.float32 ).reshape(h, w)
    return create_optix_image_2D( w, h, image.flatten() )


def write_exr( filename, optix_image ):
    w = optix_image.width
    h = optix_image.height
    data = np.zeros( (h*w*4), dtype = np.float32 )
    cuda.memcpy( 
        data.ctypes.data, 
        optix_image.data,
        w*h*4*4, 
        cuda.memcpyDeviceToHost
    )
    exr = OpenEXR.OutputFile( filename, OpenEXR.Header( w, h ) )
    exr.writePixels( { 
        'R' : data[0::4].tobytes(),
        'G' : data[1::4].tobytes(),
        'B' : data[2::4].tobytes()
    } )


def parse_args():

    import argparse
    parser = argparse.ArgumentParser( 
        description = 'Apply OptiX denoiser to input images'
    )
    parser.add_argument( 
        '-n', '--normal',   
        metavar = 'normal.exr', 
        type    = str, 
        help    = 'Screen space normals input' 
    ) 
    parser.add_argument( 
        '-a', '--albedo',
        metavar = 'albedo.exr', 
        type    = str, 
        help    = 'Albedo input'
    ) 
    parser.add_argument( 
        '-o', '--out',
        metavar = 'out.exr',
        type    = str, 
        help="Output filename, default 'denoised.exr'" ,
        default='denoised.exr' 
    )
    parser.add_argument( 
        '-t', '--tilesize', 
        metavar='INT',
        type    = int,
        nargs   = 2,
        help="Output image name.", 
        default = ( 0, 0 )
    )
    parser.add_argument( 
        '-e', '--exposure', 
        metavar = 'FLOAT',
        type    = float,
        help    = "Exposure to be applied to output",
        default = 1.0
    ) 
    parser.add_argument( 
        'color',
        metavar = 'color.exr',
        type    = str, 
        help    = "Noisy color image name."
    ) 
    return parser.parse_args()


def load_state( args, state ):


    print( "Loading color file '{}'".format( args.color) )
    state.layer.input = load_exr( args.color )
    state.layer.output = create_optix_image_2D( state.layer.input.width, state.layer.input.height, None )
    print( " ... success" )

    if args.normal:
        print( "Loading normal file '{}'".format( args.normal) )
        state.guide_layer.normal = load_exr( args.normal )
        w = state.guide_layer.normal.width
        h = state.guide_layer.normal.height
        if w != state.layer.input.width or h != state.layer.input.height:
            print( "ERROR: Normal image dims do not match color image dims" )
            sys.exit(0)
        print( " ... success" )

    if args.albedo:
        print( "Loading albedo file '{}'".format( args.albedo) )
        state.guide_layer.albedo = load_exr( args.albedo )
        w = state.guide_layer.albedo.width
        h = state.guide_layer.albedo.height
        if w != state.layer.input.width or h != state.layer.input.height:
            print( "ERROR: Albedo image dims do not match color image dims" )
            sys.exit(0) 
        print( " ... success" )

    if args.tilesize[0] <= 0 or args.tilesize[1] <= 0:
        state.tile_size = (
            state.layer.input.width,
            state.layer.input.height
        )
    else:
        state.tile_size = args.tilesize
    state.exposure  = args.exposure




#-------------------------------------------------------------------------------
#
# Denoising 
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
    if optix_version_gte( (7,2) ):
        ctx_options.validationMode = optix.DEVICE_CONTEXT_VALIDATION_MODE_ALL 

    cu_ctx = 0 
    return optix.deviceContextCreate( cu_ctx, ctx_options )


def denoiser_init( ctx, state ):
    options = optix.DenoiserOptions()
    options.guideAlbedo = 0 if state.guide_layer.albedo.width == 0 else 1
    options.guideNormal = 0 if state.guide_layer.normal.width == 0 else 1
    denoiser = ctx.denoiserCreate( optix.DENOISER_MODEL_KIND_HDR, options )
    
    sizes = denoiser.computeMemoryResources(
        state.tile_size[0],
        state.tile_size[1]
        )

    if state.tile_size[0] == state.layer.input.width and state.tile_size[0] == state.layer.input.width:
        state.scratch_size = sizes.withoutOverlapScratchSizeInBytes
    else:
        state.scratch_size = sizes.withOverlapScratchSizeInBytes
        state.overlap      = sizes.overlapWindowSizeInPixels

    state.d_state     = cp.empty( ( sizes.stateSizeInBytes ), dtype='B' )
    state.d_intensity = cp.empty( ( 1 ), 'f4' )
    state.d_scratch   = cp.empty( ( state.scratch_size ), dtype='B' )

    denoiser.setup(
        0,
        state.tile_size[0] + 2*state.overlap,
        state.tile_size[1] + 2*state.overlap,
        state.d_state.data.ptr,
        state.d_state.nbytes, 
        state.d_scratch.data.ptr,
        state.d_scratch.nbytes
        )

    return denoiser


def denoiser_exec( denoiser, state ):

    params = optix.DenoiserParams()
    params.denoiseAlpha    = 0 
    params.hdrIntensity    = state.d_intensity
    params.hdrAverageColor = 0 
    params.blendFactor     = 0.0

        
    denoiser.computeIntensity(
        0, 
        state.layer.input,
        state.d_intensity.data.ptr,
        state.d_scratch.data.ptr,
        state.d_scratch.nbytes
        )

    denoiser.invokeTiled(
        0, # CUDA stream
        params,
        state.d_state.data.ptr,
        state.d_state.nbytes,
        state.guide_layer,
        [ state.layer ],
        state.d_scratch.data.ptr,
        state.d_scratch.nbytes,
        state.overlap,
        state.tile_size[0],
        state.tile_size[1]
        )


#-------------------------------------------------------------------------------
#
# Main
#
#-------------------------------------------------------------------------------

def main():
    args  = parse_args()
    state = State()
    load_state( args, state )
    print( "\n-------- State loaded --------" )
    print( state )
    print( "------------------------------\n" )

    ctx = create_ctx()
    denoiser = denoiser_init( ctx, state )
    denoiser_exec( denoiser, state )
    write_exr( args.out, state.layer.output )
    



if __name__ == "__main__":
    main()

