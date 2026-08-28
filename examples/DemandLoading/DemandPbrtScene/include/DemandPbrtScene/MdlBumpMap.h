// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <OptiXToolkit/ShaderUtil/vec_math.h>

#include <vector_functions.h>
#include <vector_types.h>

#include <cmath>

namespace demandPbrtScene {

struct MdlBumpDifferentialGeometry
{
    float3 dpdu;
    float3 dpdv;
    float3 dndu;
    float3 dndv;
};

OTK_INLINE OTK_HOSTDEVICE float mdlBumpOffset( float dx, float dy )
{
    const float offset{ 0.5f * ( fabsf( dx ) + fabsf( dy ) ) };
    return offset == 0.0f ? 0.0005f : offset;
}

OTK_INLINE OTK_HOSTDEVICE MdlBumpDifferentialGeometry makeMdlBumpDifferentialGeometry( const float3 ( &vertices )[3],
                                                                                       const float2 ( &uvs )[3],
                                                                                       const float3 ( &normals )[3] )
{
    using namespace otk;

    const float3 dp02{ vertices[0] - vertices[2] };
    const float3 dp12{ vertices[1] - vertices[2] };
    const float2 duv02{ uvs[0] - uvs[2] };
    const float2 duv12{ uvs[1] - uvs[2] };
    const float  determinant{ duv02.x * duv12.y - duv02.y * duv12.x };

    if( fabsf( determinant ) < 1.0e-8f )
    {
        const float3 normal{ normalize( cross( vertices[1] - vertices[0], vertices[2] - vertices[0] ) ) };
        const float3 tangentU{ fabsf( normal.x ) > fabsf( normal.y )
                                   ? normalize( make_float3( -normal.z, 0.0f, normal.x ) )
                                   : normalize( make_float3( 0.0f, normal.z, -normal.y ) ) };
        return MdlBumpDifferentialGeometry{ tangentU, normalize( cross( normal, tangentU ) ),
                                            make_float3( 0.0f, 0.0f, 0.0f ), make_float3( 0.0f, 0.0f, 0.0f ) };
    }

    const float  invDeterminant{ 1.0f / determinant };
    const float3 dn02{ normals[0] - normals[2] };
    const float3 dn12{ normals[1] - normals[2] };
    return MdlBumpDifferentialGeometry{
        ( duv12.y * dp02 - duv02.y * dp12 ) * invDeterminant,
        ( -duv12.x * dp02 + duv02.x * dp12 ) * invDeterminant,
        ( duv12.y * dn02 - duv02.y * dn12 ) * invDeterminant,
        ( -duv12.x * dn02 + duv02.x * dn12 ) * invDeterminant,
    };
}

OTK_INLINE OTK_HOSTDEVICE MdlBumpDifferentialGeometry applyMdlBumpMap( const MdlBumpDifferentialGeometry& geometry,
                                                                      const float3&                     shadingNormal,
                                                                      float                             height,
                                                                      float                             heightU,
                                                                      float                             heightV,
                                                                      float                             du,
                                                                      float                             dv )
{
    const float3 dpdu{ geometry.dpdu + ( ( heightU - height ) / du ) * shadingNormal + height * geometry.dndu };
    const float3 dpdv{ geometry.dpdv + ( ( heightV - height ) / dv ) * shadingNormal + height * geometry.dndv };
    return MdlBumpDifferentialGeometry{ dpdu, dpdv, geometry.dndu, geometry.dndv };
}

OTK_INLINE OTK_HOSTDEVICE float3 mdlBumpNormal( const MdlBumpDifferentialGeometry& geometry, const float3& shadingNormal )
{
    using namespace otk;

    float3 normal{ normalize( cross( geometry.dpdu, geometry.dpdv ) ) };
    if( dot( normal, shadingNormal ) < 0.0f )
    {
        normal = -normal;
    }
    return normal;
}

}  // namespace demandPbrtScene
