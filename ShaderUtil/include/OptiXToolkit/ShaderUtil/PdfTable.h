// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

/// \file PdfTable.h
/// Create a pdf table from a texture

#include <OptiXToolkit/ShaderUtil/vec_math.h>

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
            angleTerm = sinf( ( j + 0.5f ) * float( M_PIf ) / height );
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
