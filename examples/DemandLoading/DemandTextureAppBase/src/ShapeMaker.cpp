//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/DemandTextureAppBase/ShapeMaker.h>

using namespace otk;  // for vec_math operators


void ShapeMaker::makeAxisPlane( float3 A, float3 B, std::vector<Vert>& shape )
{
    if( A.x == B.x )
    {
        shape.push_back( Vert{ float3{A.x, A.y, A.z}, float3{1.0f, 0.0f, 0.0f}, float2{0.0f, 1.0f} } );
        shape.push_back( Vert{ float3{A.x, B.y, A.z}, float3{1.0f, 0.0f, 0.0f}, float2{1.0f, 1.0f} } );
        shape.push_back( Vert{ float3{A.x, B.y, B.z}, float3{1.0f, 0.0f, 0.0f}, float2{1.0f, 0.0f} } );

        shape.push_back( Vert{ float3{A.x, A.y, A.z}, float3{1.0f, 0.0f, 0.0f}, float2{0.0f, 1.0f} } );
        shape.push_back( Vert{ float3{A.x, B.y, B.z}, float3{1.0f, 0.0f, 0.0f}, float2{1.0f, 0.0f} } );
        shape.push_back( Vert{ float3{A.x, A.y, B.z}, float3{1.0f, 0.0f, 0.0f}, float2{0.0f, 0.0f} } );
    }
    else if ( A.y == B.y )
    {
        shape.push_back( Vert{ float3{A.x, A.y, A.z}, float3{0.0f, 1.0f, 0.0f}, float2{0.0f, 1.0f} } );
        shape.push_back( Vert{ float3{B.x, A.y, B.z}, float3{0.0f, 1.0f, 0.0f}, float2{1.0f, 0.0f} } );
        shape.push_back( Vert{ float3{B.x, A.y, A.z}, float3{0.0f, 1.0f, 0.0f}, float2{1.0f, 1.0f} } );

        shape.push_back( Vert{ float3{A.x, A.y, A.z}, float3{0.0f, 1.0f, 0.0f}, float2{0.0f, 1.0f} } );
        shape.push_back( Vert{ float3{A.x, A.y, B.z}, float3{0.0f, 1.0f, 0.0f}, float2{0.0f, 0.0f} } );
        shape.push_back( Vert{ float3{B.x, A.y, B.z}, float3{0.0f, 1.0f, 0.0f}, float2{1.0f, 0.0f} } );
    }
    else if ( A.z == B.z )
    {
        shape.push_back( Vert{ float3{A.x, A.y, A.z}, float3{0.0f, 0.0f, 1.0f}, float2{0.0f, 0.0f} } );
        shape.push_back( Vert{ float3{B.x, A.y, A.z}, float3{0.0f, 0.0f, 1.0f}, float2{1.0f, 0.0f} } );
        shape.push_back( Vert{ float3{B.x, B.y, A.z}, float3{0.0f, 0.0f, 1.0f}, float2{1.0f, 1.0f} } );

        shape.push_back( Vert{ float3{A.x, A.y, A.z}, float3{0.0f, 0.0f, 1.0f}, float2{0.0f, 0.0f} } );
        shape.push_back( Vert{ float3{B.x, B.y, A.z}, float3{0.0f, 0.0f, 1.0f}, float2{1.0f, 1.0f} } );
        shape.push_back( Vert{ float3{A.x, B.y, A.z}, float3{0.0f, 0.0f, 1.0f}, float2{0.0f, 1.0f} } );
    }
}

void ShapeMaker::makeCircle( float3 center, float radius, int numSegments, std::vector<Vert>& shape )
{
    std::vector<Vert> silhouette;
    Vert a{float3{0.0f, 0.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f}, float2{0.0f, 0.0f}};
    silhouette.push_back(a);
    
    Vert b{float3{radius, 0.0f, 0.0f}, float3{0.0f, 0.0f, 0.0f}, float2{0.0f, 1.0f}};
    silhouette.push_back(b);
    ShapeMaker::spinZaxis( silhouette, numSegments, center, shape );
}

void ShapeMaker::makeSphere( float3 center, float radius, int numSegments, std::vector<Vert>& shape, float beginAngle, float endAngle )
{
    std::vector<Vert> silhouette;
    int vsegs = numSegments / 2;
    for( int i=0; i <= vsegs; ++i )
    {
        float theta = beginAngle + ( endAngle * i ) / vsegs;
        Vert v;
        v.p = float3{ radius * sinf( theta ), 0.0f, -radius * cosf( theta ) };
        v.n = float3{ sinf( theta ), 0.0f, -cosf( theta ) };
        v.t = float2{ 0.0f, float(i) / float(numSegments) };
        silhouette.push_back( v );
    }
    ShapeMaker::spinZaxis( silhouette, numSegments, center, shape );
}

void ShapeMaker::makeCylinder( float3 basePoint, float radius, float height, int numSegments, std::vector<Vert>& shape )
{
    std::vector<Vert> silhouette;
    int vsegs = numSegments / 2;
    for( int i=0; i <= vsegs; ++i )
    {
        Vert v;
        v.p = float3{ radius, 0.0f, (height * i) / vsegs };
        v.n = float3{ 1, 0.0f, 0 };
        v.t = float2{ 0.0f, float(i) / float(vsegs) };
        silhouette.push_back( v );
    }
    ShapeMaker::spinZaxis( silhouette, numSegments, basePoint, shape );
}

void ShapeMaker::makeCone( float3 basePoint, float radius, float height, int numSegments, std::vector<Vert>& shape )
{
    std::vector<Vert> silhouette;
    int vsegs = numSegments / 2;
    for( int i=0; i <= vsegs; ++i )
    {
        Vert v;
        v.p = float3{ (radius * (vsegs - i)) / vsegs, 0.0f, (height * i) / vsegs };
        float nscale = 1.0f / ( ( height * height ) + ( radius * radius ) );
        v.n = float3{ height * nscale, 0.0f, radius * nscale };
        v.t = float2{ 0.0f, float(i) / float(vsegs) };
        silhouette.push_back( v );
    }
    ShapeMaker::spinZaxis( silhouette, numSegments, basePoint, shape );
}

void ShapeMaker::makeTorus( float3 center, float radius1, float radius2, int numSegments, std::vector<Vert>& shape )
{
    float centerPoint = (radius2 + radius1) / 2.0f;
    float radius = (radius2 - radius1) / 2.0f;
    std::vector<Vert> silhouette;
    int vsegs = numSegments / 2;
    for( int i=0; i <= vsegs; ++i )
    {
        float theta = ( 2.0f * M_PIf * i ) / vsegs;
        Vert v;
        v.p = float3{ centerPoint + radius * sinf( theta ), 0.0f, radius * cosf( theta ) };
        v.n = float3{ sinf( theta ), 0.0f, cosf( theta ) };
        v.t = float2{ 0.0f, float(i) / float(numSegments) };
        silhouette.push_back( v );
    }
    ShapeMaker::spinZaxis( silhouette, numSegments, center, shape );
}

void ShapeMaker::makeVase( float3 basePoint, float radius1, float radius2, float height, int numSegments, std::vector<Vert>& shape )
{
    const float maxTheta = 5.5f;
    std::vector<Vert> silhouette;
    int vsegs = numSegments / 2;
    float offset = (radius1 + radius2) / 2.0f;
    float scale = (radius2 - radius1) / 2.0f;
    for( int i=0; i <= vsegs; ++i )
    {
        float theta = ( maxTheta * i ) / vsegs;
        Vert v;
        v.p = float3{ offset + scale * sinf( theta ), 0.0f, (height * i) / vsegs };
        v.n = normalize( float3{ height / (2.0f * M_PIf), 0.0f, -cosf( theta ) * scale * maxTheta / (2.0f * M_PIf) } );
        v.t = float2{ 0.0f, float(i) / float(numSegments) };
        silhouette.push_back( v );
    }
    ShapeMaker::spinZaxis( silhouette, numSegments, basePoint, shape );
}

void ShapeMaker::spinZaxis( std::vector<Vert>& silhouette, int numSegments, float3 translation, std::vector<Vert>& shape )
{
    // Create surface verts
    shape.clear();
    for( unsigned int j = 0; j < silhouette.size() - 1; ++j)
    {
        for( int i = 0; i < numSegments; ++i )
        {
            // Triangle 1
            if( silhouette[j].p.x != 0.0f ) // skip degenerate triangles
            {
                shape.push_back( rotateSilhouettePoint( silhouette[j],   2.0f * M_PIf * (i + 0.0f) / numSegments ) );
                shape.push_back( rotateSilhouettePoint( silhouette[j],   2.0f * M_PIf * (i + 1.0f) / numSegments ) );
                shape.push_back( rotateSilhouettePoint( silhouette[j+1], 2.0f * M_PIf * (i + 0.0f) / numSegments ) );
            }

            // Triangle 2
            if( silhouette[j+1].p.x != 0.0f ) // skip degenerate triangles
            {
                shape.push_back( rotateSilhouettePoint( silhouette[j+1], 2.0f * M_PIf * (i + 0.0f) / numSegments ) );
                shape.push_back( rotateSilhouettePoint( silhouette[j],   2.0f * M_PIf * (i + 1.0f) / numSegments ) );
                shape.push_back( rotateSilhouettePoint( silhouette[j+1], 2.0f * M_PIf * (i + 1.0f) / numSegments ) );
            }
        }
    }

    for( unsigned int i = 0; i< shape.size(); ++i )
        shape[i].p += translation;
}

Vert ShapeMaker::rotateSilhouettePoint( const Vert& p, float angle )
{
    float cosA = cosf( angle );
    float sinA = sinf( angle );
    float u = angle / (2.0f * M_PIf);
    
    Vert P;
    P.p = float3{ p.p.x * cosA, p.p.x * sinA, p.p.z };
    P.n = float3{ p.n.x * cosA, p.n.x * sinA, p.n.z };
    P.t = float2{ u, p.t.y };
    return P;
}
