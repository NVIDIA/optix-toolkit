// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "DemandPbrtScene/MdlUtils.h"

#ifdef OTK_USE_MDL

#include "DemandPbrtScene/MdlHandleTypes.h"

#include <sstream>
#include <stdexcept>

namespace demandPbrtScene {

std::string describeMdlContextMessages( const mi::neuraylib::IMdl_execution_context* context )
{
    if( !context )
    {
        return {};
    }

    std::ostringstream out;
    for( mi::Size i = 0; i < context->get_messages_count(); ++i )
    {
        MessageHandle message( context->get_message( i ) );
        if( message.is_valid_interface() )
        {
            out << message->get_string() << '\n';
        }
    }
    return out.str();
}

[[noreturn]] void failMdl( const std::string& message, const mi::neuraylib::IMdl_execution_context* context )
{
    const std::string contextMessages{ describeMdlContextMessages( context ) };
    throw std::runtime_error( contextMessages.empty() ? message : message + ":\n" + contextMessages );
}

void requireMdl( bool condition, const std::string& message, const mi::neuraylib::IMdl_execution_context* context )
{
    if( !condition )
    {
        failMdl( message, context );
    }
}

MdlTargetArgumentBlock captureMdlTargetArgumentBlock( const mi::neuraylib::ITarget_code*       targetCode,
                                                      const mi::neuraylib::ICompiled_material* compiledMaterial,
                                                      mi::neuraylib::IMdl_execution_context*   context )
{
    requireMdl( targetCode != nullptr, "Cannot capture MDL argument block without target code", context );
    requireMdl( compiledMaterial != nullptr, "Cannot capture MDL argument block without a compiled material", context );
    requireMdl( targetCode->get_callable_function_count() > 0U,
                "Cannot capture MDL argument block without callable functions", context );

    const mi::Size argumentBlockIndex{ targetCode->get_callable_function_argument_block_index( 0U ) };
    if( argumentBlockIndex == ~mi::Size( 0 ) )
    {
        return MdlTargetArgumentBlock{};
    }

    TargetArgumentBlockHandle argumentBlock( targetCode->get_argument_block( argumentBlockIndex ) );
    requireMdl( argumentBlock.is_valid_interface(), "MDL target code did not expose an argument block", context );

    TargetValueLayoutHandle layout( targetCode->get_argument_block_layout( argumentBlockIndex ) );
    requireMdl( layout.is_valid_interface(), "MDL target code did not expose an argument block layout", context );

    MdlTargetArgumentBlock result;
    result.data.assign( argumentBlock->get_data(), argumentBlock->get_data() + argumentBlock->get_size() );

    const mi::Size parameterCount{ compiledMaterial->get_parameter_count() };
    requireMdl( layout->get_num_elements() >= parameterCount,
                "MDL argument block layout has fewer entries than the compiled material", context );
    for( mi::Size i = 0; i < parameterCount; ++i )
    {
        const char* const name = compiledMaterial->get_parameter_name( i );
        requireMdl( name != nullptr, "MDL compiled material exposed a null parameter name", context );

        const mi::neuraylib::Target_value_layout_state state{ layout->get_nested_state( i ) };
        requireMdl( state.m_state_offs != ~mi::Uint32( 0 ),
                    "MDL argument block layout did not expose parameter state for " + std::string{ name }, context );

        mi::neuraylib::IValue::Kind kind{};
        mi::Size                    size{};
        const mi::Size              offset{ layout->get_layout( kind, size, state ) };
        requireMdl( offset != ~mi::Size( 0 ),
                    "MDL argument block layout did not expose parameter offset for " + std::string{ name }, context );
        requireMdl( offset + size <= result.data.size(),
                    "MDL argument block layout parameter exceeds block size for " + std::string{ name }, context );

        result.parameters.push_back( MdlTargetArgumentBlockParameter{
            name, static_cast<unsigned int>( kind ), static_cast<std::size_t>( offset ), static_cast<std::size_t>( size ) } );
    }
    return result;
}

}  // namespace demandPbrtScene

#endif  // OTK_USE_MDL
