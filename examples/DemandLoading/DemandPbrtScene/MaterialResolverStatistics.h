#pragma once

namespace demandPbrtScene {

struct MaterialResolverStats
{
    unsigned int numPartialMaterialsRealized;
    unsigned int numMaterialsRealized;
    unsigned int numMaterialsReused;
    unsigned int numProxyMaterialsCreated;
};

}
