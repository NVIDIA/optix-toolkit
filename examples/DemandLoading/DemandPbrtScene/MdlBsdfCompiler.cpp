// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlBsdfCompiler.h"

#ifdef OTK_USE_MDL

#include "DemandPbrtScene/MdlHandleTypes.h"

#include <stdexcept>

namespace demandPbrtScene {
namespace {

std::string mdlBsdfVisibleFunctions( const std::string& baseFunctionName )
{
    return baseFunctionName + "_init," + baseFunctionName + "_sample," + baseFunctionName + "_evaluate,"
           + baseFunctionName + "_pdf";
}

void captureBsdfCallableName( MdlBsdfCallablePtx& result, const std::string& functionName, const std::string& baseFunctionName )
{
    if( functionName == baseFunctionName + "_init" )
    {
        result.initFunctionName = functionName;
        return;
    }
    if( functionName == baseFunctionName + "_sample" )
    {
        result.sampleFunctionName = functionName;
        return;
    }
    if( functionName == baseFunctionName + "_evaluate" )
    {
        result.evaluateFunctionName = functionName;
        return;
    }
    if( functionName == baseFunctionName + "_pdf" )
    {
        result.pdfFunctionName = functionName;
        return;
    }

    throw std::runtime_error( "MDL generated unexpected BSDF callable function name " + functionName );
}

}  // namespace

MdlBsdfCallablePtx compileMdlBsdfCallablesToPtx( mi::neuraylib::INeuray*                  neuray,
                                                 mi::neuraylib::ITransaction*             transaction,
                                                 const mi::neuraylib::ICompiled_material* compiledMaterial,
                                                 mi::neuraylib::IMdl_execution_context*   context,
                                                 const std::string&                       expressionPath,
                                                 const std::string&                       baseFunctionName )
{
    requireMdl( neuray != nullptr, "Cannot compile MDL BSDF callables without an MDL SDK instance" );
    requireMdl( transaction != nullptr, "Cannot compile MDL BSDF callables without an MDL transaction" );
    requireMdl( compiledMaterial != nullptr, "Cannot compile MDL BSDF callables without a compiled material" );
    requireMdl( context != nullptr, "Cannot compile MDL BSDF callables without an execution context" );
    requireMdl( !expressionPath.empty(), "Cannot compile MDL BSDF callables without an expression path" );
    requireMdl( !baseFunctionName.empty(), "Cannot compile MDL BSDF callables without a base function name" );

    BackendApiHandle backendApi( neuray->get_api_component<mi::neuraylib::IMdl_backend_api>() );
    requireMdl( backendApi.is_valid_interface(), "Failed to get MDL backend API" );

    BackendHandle ptxBackend( backendApi->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX ) );
    requireMdl( ptxBackend.is_valid_interface(), "Failed to get MDL CUDA PTX backend" );
    requireMdl( ptxBackend->set_option( "sm_version", "50" ) == 0, "Failed to set MDL CUDA PTX target architecture" );
    requireMdl( ptxBackend->set_option( "df_handle_slot_mode", "none" ) == 0,
                "Failed to set MDL BSDF handle slot mode" );
    const std::string visibleFunctions{ mdlBsdfVisibleFunctions( baseFunctionName ) };
    requireMdl( ptxBackend->set_option( "visible_functions", visibleFunctions.c_str() ) == 0,
                "Failed to restrict MDL BSDF visible functions" );

    context->clear_messages();
    TargetCodeHandle targetCode( ptxBackend->translate_material_df(
        transaction, compiledMaterial, expressionPath.c_str(), baseFunctionName.c_str(), context ) );
    requireMdl( targetCode.is_valid_interface(), "Failed to translate MDL BSDF to PTX", context );
    requireMdl( targetCode->get_code_size() > 0U, "MDL generated empty BSDF PTX target code" );
    requireMdl( targetCode->get_callable_function_count() == 4U,
                "MDL generated unexpected BSDF callable function count" );

    MdlBsdfCallablePtx result;
    result.ptx.assign( targetCode->get_code(), static_cast<std::size_t>( targetCode->get_code_size() ) );
    for( mi::Size i = 0; i < targetCode->get_callable_function_count(); ++i )
    {
        const char* const functionName = targetCode->get_callable_function( i );
        requireMdl( functionName != nullptr, "MDL generated a null BSDF callable function name" );
        captureBsdfCallableName( result, functionName, baseFunctionName );
    }

    requireMdl( !result.initFunctionName.empty(), "MDL did not generate a BSDF init callable" );
    requireMdl( !result.sampleFunctionName.empty(), "MDL did not generate a BSDF sample callable" );
    requireMdl( !result.evaluateFunctionName.empty(), "MDL did not generate a BSDF evaluate callable" );
    requireMdl( !result.pdfFunctionName.empty(), "MDL did not generate a BSDF PDF callable" );
    result.argumentBlock = captureMdlTargetArgumentBlock( targetCode.get(), compiledMaterial, context );
    return result;
}

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
