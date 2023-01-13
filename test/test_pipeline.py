# Copyright (c) 2022 NVIDIA CORPORATION All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.



import optix
import pytest 
import cupy as cp

import tutil 




class TestPipeline:

    def test_pipeline_options( self ):

        pipeline_options = optix.PipelineCompileOptions()
        pipeline_options.usesMotionBlur        = False
        pipeline_options.traversableGraphFlags = optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
        pipeline_options.numPayloadValues      = 2
        pipeline_options.numAttributeValues    = 2
        pipeline_options.exceptionFlags        = optix.EXCEPTION_FLAG_NONE
        pipeline_options.pipelineLaunchParamsVariableName = "params1"
        assert pipeline_options.pipelineLaunchParamsVariableName == "params1"


        pipeline_options = optix.PipelineCompileOptions(
            usesMotionBlur        = False,
            traversableGraphFlags = optix.TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
            numPayloadValues      = 3,
            numAttributeValues    = 4,
            exceptionFlags        = optix.EXCEPTION_FLAG_NONE,
            pipelineLaunchParamsVariableName = "params2"
            )
        assert pipeline_options.pipelineLaunchParamsVariableName == "params2"


