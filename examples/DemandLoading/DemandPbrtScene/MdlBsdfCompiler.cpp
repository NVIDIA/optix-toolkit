// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlBsdfCompiler.h"

#ifdef OTK_USE_MDL

#include "DemandPbrtScene/MdlHandleTypes.h"

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
        MessageHandle message( context->get_message( i ) );
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

MdlTargetArgumentBlock captureTargetArgumentBlock( const mi::neuraylib::ITarget_code*       targetCode,
                                                   const mi::neuraylib::ICompiled_material* compiledMaterial,
                                                   mi::neuraylib::IMdl_execution_context*   context )
{
    requireMdlBsdfCompile( targetCode != nullptr, "Cannot capture MDL argument block without target code", context );
    requireMdlBsdfCompile( compiledMaterial != nullptr, "Cannot capture MDL argument block without a compiled material", context );
    requireMdlBsdfCompile( targetCode->get_callable_function_count() > 0U,
                           "Cannot capture MDL argument block without callable functions", context );

    const mi::Size argumentBlockIndex{ targetCode->get_callable_function_argument_block_index( 0U ) };
    if( argumentBlockIndex == ~mi::Size( 0 ) )
    {
        return MdlTargetArgumentBlock{};
    }

    TargetArgumentBlockHandle argumentBlock( targetCode->get_argument_block( argumentBlockIndex ) );
    requireMdlBsdfCompile( argumentBlock.is_valid_interface(), "MDL target code did not expose an argument block", context );

    TargetValueLayoutHandle layout( targetCode->get_argument_block_layout( argumentBlockIndex ) );
    requireMdlBsdfCompile( layout.is_valid_interface(), "MDL target code did not expose an argument block layout", context );

    MdlTargetArgumentBlock result;
    result.data.assign( argumentBlock->get_data(), argumentBlock->get_data() + argumentBlock->get_size() );

    const mi::Size parameterCount{ compiledMaterial->get_parameter_count() };
    requireMdlBsdfCompile( layout->get_num_elements() >= parameterCount,
                           "MDL argument block layout has fewer entries than the compiled material", context );
    for( mi::Size i = 0; i < parameterCount; ++i )
    {
        const char* const name = compiledMaterial->get_parameter_name( i );
        requireMdlBsdfCompile( name != nullptr, "MDL compiled material exposed a null parameter name", context );

        const mi::neuraylib::Target_value_layout_state state{ layout->get_nested_state( i ) };
        requireMdlBsdfCompile( state.m_state_offs != ~mi::Uint32( 0 ),
                               "MDL argument block layout did not expose parameter state for " + std::string{ name }, context );

        mi::neuraylib::IValue::Kind kind{};
        mi::Size                    size{};
        const mi::Size              offset{ layout->get_layout( kind, size, state ) };
        requireMdlBsdfCompile( offset != ~mi::Size( 0 ),
                               "MDL argument block layout did not expose parameter offset for " + std::string{ name }, context );
        requireMdlBsdfCompile( offset + size <= result.data.size(),
                               "MDL argument block layout parameter exceeds block size for " + std::string{ name }, context );

        result.parameters.push_back( MdlTargetArgumentBlockParameter{
            name, static_cast<unsigned int>( kind ), static_cast<std::size_t>( offset ), static_cast<std::size_t>( size ) } );
    }
    return result;
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
    requireMdlBsdfCompile( !baseFunctionName.empty(),
                           "Cannot compile MDL BSDF callables without a base function "
                           "name" );

    BackendApiHandle backendApi( neuray->get_api_component<mi::neuraylib::IMdl_backend_api>() );
    requireMdlBsdfCompile( backendApi.is_valid_interface(), "Failed to get MDL backend API" );

    BackendHandle ptxBackend( backendApi->get_backend( mi::neuraylib::IMdl_backend_api::MB_CUDA_PTX ) );
    requireMdlBsdfCompile( ptxBackend.is_valid_interface(), "Failed to get MDL CUDA PTX backend" );
    requireMdlBsdfCompile( ptxBackend->set_option( "sm_version", "50" ) == 0,
                           "Failed to set MDL CUDA PTX target architecture" );
    requireMdlBsdfCompile( ptxBackend->set_option( "df_handle_slot_mode", "none" ) == 0,
                           "Failed to set MDL BSDF handle slot mode" );
    const std::string visibleFunctions{ mdlBsdfVisibleFunctions( baseFunctionName ) };
    requireMdlBsdfCompile( ptxBackend->set_option( "visible_functions", visibleFunctions.c_str() ) == 0,
                           "Failed to restrict MDL BSDF visible functions" );

    context->clear_messages();
    TargetCodeHandle targetCode( ptxBackend->translate_material_df(
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
    result.argumentBlock = captureTargetArgumentBlock( targetCode.get(), compiledMaterial, context );
    return result;
}

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
