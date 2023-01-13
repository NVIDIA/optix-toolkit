# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import cupy as cp
import optix as ox
import pytest 

import tutil 


class Logger:
    def __init__(self):
        self.num_mssgs = 0
    
    def __call__(self, level, tag, mssg):
        print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))
        self.num_mssgs += 1
    

def log_callback(level, tag, mssg):
    print("[{:>2}][{:>12}]: {}".format(level, tag, mssg))


class TestDeviceContextOptions:
    def test_default_ctor(self):
        options = ox.DeviceContextOptions()
        assert options.logCallbackFunction is None
        assert options.logCallbackLevel == 0
        if tutil.optix_version_gte( (7,2) ): 
            assert options.validationMode == ox.DEVICE_CONTEXT_VALIDATION_MODE_OFF

    def test_ctor0(self):
        options = ox.DeviceContextOptions(log_callback)
        assert options.logCallbackFunction == log_callback
   
    def test_ctor1(self):
        logger = Logger()
        if tutil.optix_version_gte( (7,2) ): 
            options = ox.DeviceContextOptions(
                logCallbackFunction = logger,
                logCallbackLevel    = 3,
                validationMode      = ox.DEVICE_CONTEXT_VALIDATION_MODE_ALL
            )
        else:
            options = ox.DeviceContextOptions(
                logCallbackFunction = logger,
                logCallbackLevel    = 3
            )
        assert options.logCallbackFunction == logger
        assert options.logCallbackLevel    == 3
        if tutil.optix_version_gte( (7,2) ): 
            assert options.validationMode == ox.DEVICE_CONTEXT_VALIDATION_MODE_ALL
        else:
            assert options.validationMode == ox.DEVICE_CONTEXT_VALIDATION_MODE_OFF

    def test_context_options_props(self):
        options = ox.DeviceContextOptions()
        options.logCallbackLevel = 1
        assert options.logCallbackLevel == 1

        options.logCallbackFunction = log_callback
        assert options.logCallbackFunction == log_callback 


class TestContext:
    def test_create_destroy( self ):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        ctx.destroy()

    def test_get_property( self ):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        v = ctx.getProperty( ox.DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK )
        assert type( v ) is int
        assert v > 1 and v <= 16  # at time of writing, was 8
        ctx.destroy()
    
    def test_set_log_callback( self ):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        logger = Logger()
        ctx.setLogCallback( logger, 3 )
        ctx.setLogCallback( None, 2 )
        ctx.setLogCallback( log_callback, 1 )
        ctx.destroy()

    def test_cache_default(self):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        assert ctx.getCacheEnabled()
        ctx.destroy()

    def test_cache_enable_disable(self):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        ctx.setCacheEnabled(False);
        assert not ctx.getCacheEnabled()
        ctx.setCacheEnabled(True);
        assert ctx.getCacheEnabled()
        ctx.destroy()

    def test_cache_database_sizes(self):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())
        db_sizes = ( 1024, 1024*1024 )
        ctx.setCacheDatabaseSizes( *db_sizes )
        assert ctx.getCacheDatabaseSizes() == db_sizes 
        ctx.destroy()
        
    def test_set_get_cache( self ):
        ctx = ox.deviceContextCreate(0, ox.DeviceContextOptions())

        v = ctx.getCacheLocation() 
        assert type(v) is str

        loc =  "/dev/null"
        with pytest.raises( RuntimeError ):
            ctx.setCacheLocation( loc ) # not valid dir
        ctx.destroy()



