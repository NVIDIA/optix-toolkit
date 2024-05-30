#
#  Copyright (c) 2023 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

include(ImageTest)

image_test_get_resource_dir(resourceDir DemandPbrtScene)

function(pbrtImageTest name warmup scene)
    cmake_parse_arguments(pbrtImageTest "" "" "RESOURCES" ${ARGN})
    if(NOT pbrtImageTest_RESOURCES)
        set(pbrtImageTest_RESOURCES "")
    endif()
    add_image_test(DemandPbrtScene ${name}
        FOLDER Examples/DemandLoading/Tests
        RESOURCES ${scene} ${pbrtImageTest_RESOURCES}
        ARGS --bg=0.3,0.3,0.3 --warmup=${warmup} ${resourceDir}/${scene})
endfunction()

source_group("Scene Files" REGULAR_EXPRESSION ".*\\.pbrt")
source_group("Mesh Files" REGULAR_EXPRESSION ".*\\.ply")

# Render proxy geometry
pbrtImageTest(cube0 0 cube.pbrt RESOURCES light-sources.pbrt)
# Render realized geometry with proxy material
pbrtImageTest(cube1 1 cube.pbrt)
# Render realized geometry with realized material
pbrtImageTest(cube2 2 cube.pbrt)

# Render scene proxy
pbrtImageTest(two-cubes0 0 two-cubes.pbrt RESOURCES cube-geom.pbrt)
# Render decomposed proxies
pbrtImageTest(two-cubes1 1 two-cubes.pbrt)
# Render realized geometry with proxy material
pbrtImageTest(two-cubes2 2 two-cubes.pbrt)
# Render realized geometry with real material for cube 1, proxy material for cube 2
pbrtImageTest(two-cubes3 3 two-cubes.pbrt)
# Render realized geometry with real material for cube 1, cube 2
pbrtImageTest(two-cubes4 4 two-cubes.pbrt)

# Render PLY mesh
pbrtImageTest(cube-ply0 0 cube-mesh.pbrt RESOURCES cube.ply)
pbrtImageTest(cube-ply1 1 cube-mesh.pbrt)
pbrtImageTest(cube-ply2 2 cube-mesh.pbrt)

# Render sphere proxy geometry
# warmup 0: load sphere proxy
# warmup 1: load sphere geometry with proxy material
# warmup 2: load real material
pbrtImageTest(sphere0 0 sphere.pbrt)
pbrtImageTest(sphere1 1 sphere.pbrt)
pbrtImageTest(sphere2 2 sphere.pbrt)

# Render instance proxy geometry
# warmup 0: load instance proxy
# warmup 1: load instance geometry with proxy material
# warmup 2: load real material
pbrtImageTest(instance-cube0 0 instance-cube.pbrt)
pbrtImageTest(instance-cube1 1 instance-cube.pbrt)
pbrtImageTest(instance-cube2 2 instance-cube.pbrt)

# PNG textures require OpenImageIO
if(OTK_USE_OIIO)
    # Render cube with opaque everywhere alpha map
    # warmup 0: load scene proxy
    # warmup 1: load real geometry from proxy
    # warmup 2: load alpha texture sampler
    # warmup 3: load alpha texture tile
    # warmup 4: sample alpha texture
    pbrtImageTest(cutout-white0 0 cutout-white.pbrt RESOURCES cube-textured-geom.pbrt cutout-white.png)
    pbrtImageTest(cutout-white1 1 cutout-white.pbrt)
    pbrtImageTest(cutout-white2 2 cutout-white.pbrt)
    pbrtImageTest(cutout-white3 3 cutout-white.pbrt)
    pbrtImageTest(cutout-white4 4 cutout-white.pbrt)

    # Render cube with cut out alpha map
    # warmup 0: load scene proxy
    # warmup 1: load real geometry from proxy
    # warmup 2: load alpha texture sampler
    # warmup 3: load alpha texture tile
    # warmup 4: sample alpha texture
    pbrtImageTest(cutout2 2 cutout.pbrt RESOURCES cutout.png)
    pbrtImageTest(cutout3 3 cutout.pbrt)
    pbrtImageTest(cutout4 4 cutout.pbrt)

    # Render cube with diffuse map
    # warmup 0: load scene proxy
    # warmup 1: load real geometry from proxy
    # warmup 2: load diffuse texture sampler
    # warmup 3: load diffuse texture tile
    # warmup 4: sample diffuse texture
    pbrtImageTest(diffuse2 2 diffuse.pbrt RESOURCES mandelbrot-hue.png)
    pbrtImageTest(diffuse3 3 diffuse.pbrt)
    pbrtImageTest(diffuse4 4 diffuse.pbrt)

    # Render cube with diffuse map and alpha map
    # warmup 0: load scene proxy
    # warmup 1: load real geometry from proxy
    # warmup 2: load alpha texture sampler
    # warmup 3: load alpha texture tile
    # warmup 4: sample alpha texture, load diffuse texture sampler
    # warmup 5: sample alpha texture, load diffuse texture tile
    # warmup 6: sample alpha texture, sample diffuse texture
    pbrtImageTest(diffuse-alpha4 4 diffuse-alpha.pbrt RESOURCES circle.png)
    pbrtImageTest(diffuse-alpha5 5 diffuse-alpha.pbrt)
    pbrtImageTest(diffuse-alpha6 6 diffuse-alpha.pbrt)
endif()
