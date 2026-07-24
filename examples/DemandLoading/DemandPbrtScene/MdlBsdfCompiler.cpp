// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlBsdfCompiler.h"

#ifdef OTK_USE_MDL

#include <mi/mdl_sdk.h>

#include <sstream>
#include <stdexcept>

namespace demandPbrtScene {
namespace {

std::string describeContextMessages( const mi::neuraylib::IMdl_execution_context* context )
{
    if( !context )
        return {};

    std::ostringstream out;
    for( mi::Size i = 0; i < context->get_messages_count(); ++i )
    {
        mi::base::Handle<const mi::neuraylib::IMessage> message( context->get_message( i ) );
        if( message.is_valid_interface() )
            out << message->get_string() << '\n';
    }
    return out.str();
}

[[noreturn]] void failMdlBsdfCompile( const std::string& message, const mi::neuraylib::IMdl_execution_context* context = nullptr )
{
    const std::string contextMessages{ describeContextMessages( context ) };
    throw std::runtime_error( contextMessages.empty() ? message : message + ":\n" + contextMessages );
}

void requireMdlBsdfCompile( bool condition, const std::string& message, const mi::neuraylib::IMdl_execution_context* context = nullptr )
{
    if( !condition )
    {
        failMdlBsdfCompile( message, context );
    }
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
    requireMdlBsdfCompile( neuray != nullptr, "Cannot compile MDL BSDF callables without an MDL SDK instance" );
    requireMdlBsdfCompile( transaction != nullptr, "Cannot compile MDL BSDF callables without an MDL transaction" );
    requireMdlBsdfCompile( compiledMaterial != nullptr,
                           "Cannot compile MDL BSDF callables without a compiled material" );
    requireMdlBsdfCompile( context != nullptr, "Cannot compile MDL BSDF callables without an execution context" );
    requireMdlBsdfCompile( !expressionPath.empty(), "Cannot compile MDL BSDF callables without an expression path" );
    requireMdlBsdfCompile( !baseFunctionName.empty(), "Cannot compile MDL BSDF callables without a base function "
                                                      "name" );

    mi::base::Handle<mi::neuraylib::IMdl_backend_api> backendApi( neuray->get_api_component<mi::neuraylib::IMdl_backend_api>() );
    requireMdlBsdfCompile( backendApi.is_valid_interface(), "Failed to get MDL backend API" );

    mi::base::Handle<mi::neuraylib::IMdl_backend> ptxBackend( backendApi->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX ) );
    requireMdlBsdfCompile( ptxBackend.is_valid_interface(), "Failed to get MDL CUDA PTX backend" );
    requireMdlBsdfCompile( ptxBackend->set_option( "sm_version", "50" ) == 0,
                           "Failed to set MDL CUDA PTX target architecture" );

    context->clear_messages();
    mi::base::Handle<const mi::neuraylib::ITarget_code> targetCode( ptxBackend->translate_material_df(
        transaction, compiledMaterial, expressionPath.c_str(), baseFunctionName.c_str(), context ) );
    requireMdlBsdfCompile( targetCode.is_valid_interface(), "Failed to translate MDL BSDF to PTX", context );
    requireMdlBsdfCompile( targetCode->get_code_size() > 0U, "MDL generated empty BSDF PTX target code" );
    requireMdlBsdfCompile( targetCode->get_callable_function_count() == 4U,
                           "MDL generated unexpected BSDF callable function count" );

    MdlBsdfCallablePtx result;
    result.ptx.assign( targetCode->get_code(), static_cast<std::size_t>( targetCode->get_code_size() ) );
    for( mi::Size i = 0; i < targetCode->get_callable_function_count(); ++i )
    {
        const char* const functionName = targetCode->get_callable_function( i );
        requireMdlBsdfCompile( functionName != nullptr, "MDL generated a null BSDF callable function name" );
        captureBsdfCallableName( result, functionName, baseFunctionName );
    }

    requireMdlBsdfCompile( !result.initFunctionName.empty(), "MDL did not generate a BSDF init callable" );
    requireMdlBsdfCompile( !result.sampleFunctionName.empty(), "MDL did not generate a BSDF sample callable" );
    requireMdlBsdfCompile( !result.evaluateFunctionName.empty(), "MDL did not generate a BSDF evaluate callable" );
    requireMdlBsdfCompile( !result.pdfFunctionName.empty(), "MDL did not generate a BSDF PDF callable" );
    return result;
}

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
