// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#pragma once

#include <SceneProxy.h>

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

inline std::ostream& operator<<( std::ostream& str, const GeometryInstance& lhs )
{
    return str << "GeometryInstance{ " << lhs.accelBuffer  //
               << ", " << lhs.primitive                    //
               << ", " << lhs.instance                     //
               << ", " << lhs.material                     //
               << ", '" << lhs.diffuseMapFileName << "'"   //
               << ", '" << lhs.alphaMapFileName << "'"     //
               << ", " << lhs.devNormals                   //
               << ", " << lhs.devUVs                       //
               << " }";
}

inline bool operator==( const GeometryInstance& lhs, const GeometryInstance& rhs )
{
    return lhs.accelBuffer == rhs.accelBuffer                   //
           && lhs.primitive == rhs.primitive                    //
           && lhs.instance == rhs.instance                      //
           && lhs.material == rhs.material                      //
           && lhs.diffuseMapFileName == rhs.diffuseMapFileName  //
           && lhs.alphaMapFileName == rhs.alphaMapFileName      //
           && lhs.devNormals == rhs.devNormals                  //
           && lhs.devUVs == rhs.devUVs;
}

}  // namespace demandPbrtScene
