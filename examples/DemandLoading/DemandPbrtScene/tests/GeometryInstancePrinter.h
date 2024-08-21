// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <DemandPbrtScene/SceneProxy.h>

#include <OptiXToolkit/DemandGeometry/Mocks/OptixCompare.h>

#include "ParamsPrinters.h"

#include <iostream>

namespace demandPbrtScene {

inline std::ostream& operator<<( std::ostream& str, GeometryPrimitive value )
{
    switch( value )
    {
        case GeometryPrimitive::NONE:
            return str << "NONE";
        case GeometryPrimitive::TRIANGLE:
            return str << "TRIANGLE";
        case GeometryPrimitive::SPHERE:
            return str << "SPHERE";
    }
    return str << "? (" << static_cast<int>( value ) << ')';
}

inline std::ostream& operator<<( std::ostream& str, const MaterialGroup& value )
{
    return str << "MaterialGroup{ "                         //
               << value.material                            //
               << ", '" << value.diffuseMapFileName << "'"  //
               << ", '" << value.alphaMapFileName << "'"    //
               << ", " << value.primitiveIndexEnd << " }";
}

inline std::ostream& operator<<( std::ostream& str, const GeometryInstance& lhs )
{
    return str << "GeometryInstance{ " << lhs.accelBuffer  //
               << ", " << lhs.primitive                    //
               << ", " << lhs.instance                     //
               << ", " << lhs.groups                       //
               << ", " << lhs.devNormals                   //
               << ", " << lhs.devUVs                       //
               << " }";
}

inline bool operator==( const MaterialGroup& lhs, const MaterialGroup& rhs )
{
    return lhs.material == rhs.material                         //
           && lhs.diffuseMapFileName == rhs.diffuseMapFileName  //
           && lhs.alphaMapFileName == rhs.alphaMapFileName      //
           && lhs.primitiveIndexEnd == rhs.primitiveIndexEnd;
}

inline bool operator!=( const MaterialGroup& lhs, const MaterialGroup& rhs )
{
    return !( lhs == rhs );
}

inline bool operator==( const GeometryInstance& lhs, const GeometryInstance& rhs )
{
    return lhs.accelBuffer == rhs.accelBuffer   //
           && lhs.primitive == rhs.primitive    //
           && lhs.instance == rhs.instance      //
           && lhs.groups == rhs.groups          //
           && lhs.devNormals == rhs.devNormals  //
           && lhs.devUVs == rhs.devUVs;
}

inline bool operator!=( const GeometryInstance& lhs, const GeometryInstance& rhs )
{
    return !( lhs == rhs );
}

}  // namespace demandPbrtScene
