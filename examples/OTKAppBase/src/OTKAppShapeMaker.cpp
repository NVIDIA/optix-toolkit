// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/ShaderUtil/vec_math.h>
#include <OptiXToolkit/OTKAppBase/OTKAppShapeMaker.h>

using namespace otk;  // for vec_math operators


void OTKAppShapeMaker::makeAxisPlane( float3 A, float3 B, std::vector<Vert>& shape )
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

void OTKAppShapeMaker::makeCircle( float3 center, float radius, int numSegments, std::vector<Vert>& shape )
{
    std::vector<Vert> silhouette;
    Vert a{float3{0.0f, 0.0f, 0.0f}, float3{0.0f, 0.0f, 1.0f}, float2{0.0f, 0.0f}};
    silhouette.push_back(a);
    
    Vert b{float3{radius, 0.0f, 0.0f}, float3{0.0f, 0.0f, 0.0f}, float2{0.0f, 1.0f}};
    silhouette.push_back(b);
    OTKAppShapeMaker::spinZaxis( silhouette, numSegments, center, shape );
}

void OTKAppShapeMaker::makeSphere( float3 center, float radius, int numSegments, std::vector<Vert>& shape, float beginAngle, float endAngle )
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
    OTKAppShapeMaker::spinZaxis( silhouette, numSegments, center, shape );
}

void OTKAppShapeMaker::makeCylinder( float3 basePoint, float radius, float height, int numSegments, std::vector<Vert>& shape )
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
    OTKAppShapeMaker::spinZaxis( silhouette, numSegments, basePoint, shape );
}

void OTKAppShapeMaker::makeCone( float3 basePoint, float radius, float height, int numSegments, std::vector<Vert>& shape )
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
    OTKAppShapeMaker::spinZaxis( silhouette, numSegments, basePoint, shape );
}

void OTKAppShapeMaker::makeTorus( float3 center, float radius1, float radius2, int numSegments, std::vector<Vert>& shape )
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
    OTKAppShapeMaker::spinZaxis( silhouette, numSegments, center, shape );
}

void OTKAppShapeMaker::makeVase( float3 basePoint, float radius1, float radius2, float height, int numSegments, std::vector<Vert>& shape )
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
    OTKAppShapeMaker::spinZaxis( silhouette, numSegments, basePoint, shape );
}

void OTKAppShapeMaker::makeBox( float3 corner, float3 dim, std::vector<Vert>& shape )
{
    float normals[18] = { -1,0,0, 1,0,0, 0,-1,0, 0,1,0, 0,0,-1, 0,0,1 };
    float texcos[12] = { 0,0, 1,0, 1,1, 0,0, 1,1, 0,1 };
    float pos[108] = { 0,0,0, 0,1,1, 0,1,0,  0,0,0, 0,0,1, 0,1,1, 
                       1,0,0, 1,1,0, 1,1,1,  1,0,0, 1,1,1, 1,0,1,
                       0,0,0, 1,0,0, 1,0,1,  0,0,0, 1,0,1, 0,0,1, 
                       0,1,0, 1,1,1, 1,1,0,  0,1,0, 0,1,1, 1,1,1,   
                       0,0,0, 1,0,0, 1,1,0,  0,0,0, 1,1,0, 0,1,0,
                       0,0,1, 1,0,1, 1,1,1,  0,0,1, 1,1,1, 0,1,1 };

    for( int face = 0; face < 6; ++face )
    {
        for( int vertex = 0; vertex < 6; ++vertex )
        {
            float* p = &pos[ face*6*3 + vertex*3 ];
            float* t = &texcos[ vertex*2 ];
            float* n = &normals[ face*3 ];
            Vert v = Vert{ float3{p[0], p[1], p[2]} * dim + corner, float3{n[0], n[1], n[2]}, float2{t[0], t[1]} };
            shape.push_back( v );
        }
    }
}


void OTKAppShapeMaker::spinZaxis( std::vector<Vert>& silhouette, int numSegments, float3 translation, std::vector<Vert>& shape )
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

Vert OTKAppShapeMaker::rotateSilhouettePoint( const Vert& p, float angle )
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
