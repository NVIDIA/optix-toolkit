//
// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

/// \file PdfTable.h
/// Create a pdf table from a texture

#include <math.h>

enum PdfBrightnessType { pbLUMINANCE, pbRGBSUM };
enum PdfAngleType { paNONE, paLATLONG, paCUBEMAP };

#define LUMINANCE( c ) ( 0.299f * float(c.x) + 0.587f * float(c.y) + 0.114f * float(c.z) )
#define RGBSUM( c ) ( float(c.x) + float(c.y) + float(c.z) )

/// Make a PDF array from an RGB image
template <class TYPE> 
void makePdfTable( float* pdfTable, TYPE* srcArray, float* aveBrightness,
                   int width, int height, PdfBrightnessType brightnessType, PdfAngleType angleType )
{
    double sumBrightnessAngle = 0.0f;
    double sumAngleTerm = 0.0f;

    for( int j = 0; j < height; ++j )
    {
        float angleTerm = 1.0f;
        if ( angleType == paLATLONG )
        {
            angleTerm = sinf( ( j + 0.5f ) * float( M_PI ) / height );
        }

        for( int i = 0; i < width; ++i )
        {
            if( angleType == paCUBEMAP )
            {
                float x = ( 2.0f * i + 1.0f - width ) / float( width );
                float y = ( 2.0f * j + 1.0f - height ) / float( height );
                float d = sqrtf( x * x + y * y + 1.0f );
                angleTerm = 1.0f / ( d * d * d );
            }

            TYPE c = srcArray[j * width + i];
            float brightnessTerm;
            if( brightnessType == pbRGBSUM )
                brightnessTerm = RGBSUM( c );
            else
                brightnessTerm = LUMINANCE( c );

            pdfTable[j * width + i] = brightnessTerm * angleTerm;
            sumBrightnessAngle += brightnessTerm * angleTerm;
            sumAngleTerm += angleTerm;
        }
    }

    *aveBrightness = static_cast<float>( sumBrightnessAngle / sumAngleTerm );
}