// SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

//------------------------------------------------------------------------------
//
// GL error-checking
//
//------------------------------------------------------------------------------

#define DO_GL_CHECK
#ifdef DO_GL_CHECK

#define GL_CHECK( call )                                                       \
    do                                                                         \
    {                                                                          \
        call;                                                                  \
        ::otk::glCheck( #call, __FILE__, __LINE__ );                         \
    } while( false )


#define GL_CHECK_ERRORS() ::otk::glCheckErrors( __FILE__, __LINE__ )

#else
#define GL_CHECK( call )                                                       \
    do                                                                         \
    {                                                                          \
        call;                                                                  \
    } while( 0 )
#define GL_CHECK_ERRORS()                                                      \
    do                                                                         \
    {                                                                          \
        ;                                                                      \
    } while( 0 )
#endif


namespace otk {

void glCheck( const char* call, const char* file, unsigned int line );

void glCheckErrors( const char* file, unsigned int line );

}  // end namespace otk
    
