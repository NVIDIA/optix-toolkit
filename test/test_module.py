# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import optix 
import cupy as cp

import array
import pytest

import sample_ptx
import tutil 


if tutil.optix_version_gte( (7,2) ): 
    class TestModuleCompileBoundValueEntry:
        def test_compile_bound_value_entry( self ):
            bound_value_entry_default = optix.ModuleCompileBoundValueEntry(
            )

            bound_value = array.array( 'f', [0.1, 0.2, 0.3] )
            bound_value_entry = optix.ModuleCompileBoundValueEntry(
                pipelineParamOffsetInBytes = 4,
                boundValue  = bound_value,
                annotation  = "my_bound_value"
            )

            assert bound_value_entry.pipelineParamOffsetInBytes == 4
            with pytest.raises( AttributeError ):
                print( bound_value_entry.boundValue )
            assert bound_value_entry.annotation == "my_bound_value"

            bound_value_entry.pipelineParamOffsetInBytes = 8
            assert bound_value_entry.pipelineParamOffsetInBytes == 8
            bound_value_entry.annotation = "new_bound_value"
            assert bound_value_entry.annotation == "new_bound_value"


if tutil.optix_version_gte( (7,4) ): 
    class TestModuleCompilePayloadType:
        def test_compile_payload_type( self ):
            payload_semantics = [ 0, 1 ]
            payload_type_default = optix.PayloadType(
            )
            payload_type_default.payloadSemantics = payload_semantics
            
            payload_type = optix.PayloadType(
                payloadSemantics = payload_semantics
            )


class TestModule:
    if tutil.optix_version_gte( (7,2) ): 
        def test_options( self ):
            mod_opts = optix.ModuleCompileOptions(
                maxRegisterCount = 64,
                optLevel         = optix.COMPILE_OPTIMIZATION_LEVEL_1,
                debugLevel       = tutil.default_debug_level(),
                boundValues      = []
            )
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == tutil.default_debug_level() 
            # optix.ModuleCompileOptions.boundValues is write-only
            with pytest.raises( AttributeError ):
                print( mod_opts.boundValues )

            mod_opts = optix.ModuleCompileOptions()
            assert mod_opts.maxRegisterCount == optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT
            assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_DEFAULT
            assert mod_opts.debugLevel       == tutil.default_debug_level()
            mod_opts.maxRegisterCount = 64
            mod_opts.optLevel         = optix.COMPILE_OPTIMIZATION_LEVEL_1
            mod_opts.debugLevel       = tutil.default_debug_level()
            mod_opts.boundValues = [ optix.ModuleCompileBoundValueEntry() ];
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == tutil.default_debug_level()
    elif tutil.optix_version_gte( (7,1) ): 
        def test_options( self ):
            mod_opts = optix.ModuleCompileOptions(
                maxRegisterCount = 64,
                optLevel         = optix.COMPILE_OPTIMIZATION_LEVEL_1,
                debugLevel       = tutil.default_debug_level()
            )
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == tutil.default_debug_level()

            mod_opts = optix.ModuleCompileOptions()
            assert mod_opts.maxRegisterCount == optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT
            assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_DEFAULT
            assert mod_opts.debugLevel       == optix.COMPILE_DEBUG_LEVEL_DEFAULT
            mod_opts.maxRegisterCount = 64
            mod_opts.optLevel         = optix.COMPILE_OPTIMIZATION_LEVEL_1
            mod_opts.debugLevel       = tutil.default_debug_level()
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == tutil.default_debug_level()
    else:
        def test_options( self ):
            mod_opts = optix.ModuleCompileOptions(
                maxRegisterCount = 64,
                optLevel         = optix.COMPILE_OPTIMIZATION_LEVEL_1,
                debugLevel       = tutil.default_debug_level()
            )
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == tutil.default_debug_level()

            mod_opts = optix.ModuleCompileOptions()
            assert mod_opts.maxRegisterCount == optix.COMPILE_DEFAULT_MAX_REGISTER_COUNT
            assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_DEFAULT
            assert mod_opts.debugLevel       == tutil.default_debug_level()
            mod_opts.maxRegisterCount = 64
            mod_opts.optLevel         = optix.COMPILE_OPTIMIZATION_LEVEL_1
            mod_opts.debugLevel       = optix.COMPILE_DEBUG_LEVEL_FULL
            assert mod_opts.maxRegisterCount == 64
            assert mod_opts.optLevel         == optix.COMPILE_OPTIMIZATION_LEVEL_1
            assert mod_opts.debugLevel       == optix.COMPILE_DEBUG_LEVEL_FULL

    def test_create_destroy( self ):
        ctx = optix.deviceContextCreate(0, optix.DeviceContextOptions())
        module_opts   = optix.ModuleCompileOptions()
        pipeline_opts = optix.PipelineCompileOptions()
        mod, log = ctx.moduleCreateFromPTX(
            module_opts,
            pipeline_opts,
            sample_ptx.hello_ptx,
            )
        assert type(mod) is optix.Module
        assert type(log) is str

        mod.destroy()
        ctx.destroy()
            

if tutil.optix_version_gte( (7,4) ): 
    def test_payload_semantics_use( self ):
        ctx = optix.deviceContextCreate(0, optix.DeviceContextOptions())
        module_opts   = optix.ModuleCompileOptions()
        pipeline_opts = optix.PipelineCompileOptions()

        payload_sem = ( 
            optix.PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE | 
            optix.PAYLOAD_SEMANTICS_CH_READ_WRITE | 
            optix.PAYLOAD_SEMANTICS_MS_READ_WRITE | 
            optix.PAYLOAD_SEMANTICS_AH_READ_WRITE | 
            optix.PAYLOAD_SEMANTICS_IS_READ_WRITE
            )
        
        payload_type = optix.PayloadType( [ payload_sem, payload_sem, payload_sem ] )
        module_opts.payloadTypes = [ payload_type ]
        mod, log = ctx.moduleCreateFromPTX(
            module_opts,
            pipeline_opts,
            sample_ptx.triangle_ptx,
            )
        mod.destroy()
        ctx.destroy()


    def test_bound_values_use( self ):
        ctx = optix.deviceContextCreate(0, optix.DeviceContextOptions())
        module_opts   = optix.ModuleCompileOptions()
        pipeline_opts = optix.PipelineCompileOptions()

        bound_value = array.array( 'f', [0.1, 0.2, 0.3] )
        bound_value_entry = optix.ModuleCompileBoundValueEntry(
            pipelineParamOffsetInBytes = 4,
            boundValue  = bound_value,
            annotation  = "my_bound_value"
        )
        module_opts.boundValues = [ bound_value_entry ]

        mod, log = ctx.moduleCreateFromPTX(
            module_opts,
            pipeline_opts,
            sample_ptx.hello_ptx,
            )
        mod.destroy()
        ctx.destroy()


    if tutil.optix_version_gte( (7,1) ): 
        def test_builtin_is_module_get( self ):
            ctx = optix.deviceContextCreate(0, optix.DeviceContextOptions())
            module_opts     = optix.ModuleCompileOptions()
            pipeline_opts   = optix.PipelineCompileOptions()
            builtin_is_opts = optix.BuiltinISOptions()
            builtin_is_opts.builtinISModuleType = optix.PRIMITIVE_TYPE_TRIANGLE

            is_mod = ctx.builtinISModuleGet(
                module_opts,
                pipeline_opts,
                builtin_is_opts
            )
            assert type( is_mod ) is optix.Module
            is_mod.destroy()
            ctx.destroy()
