// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include "Config.h"
#include "ImageSourceTestConfig.h"

#include <OptiXToolkit/ImageSource/ImageSourceCache.h>
#include <OptiXToolkit/ImageSource/TiledImageSource.h>

#include <gtest/gtest.h>

using namespace imageSource;

class TestImageSourceCache : public testing::Test
{
  protected:
    ImageSourceCache m_cache;
    std::string      m_directoryPrefix{ getSourceDir() + "/Textures/" };
    std::string      m_exrPath{ m_directoryPrefix + "TiledMipMappedFloat.exr" };
    std::string      m_jpgPath{ m_directoryPrefix + "level0.jpg" };
};

TEST_F( TestImageSourceCache, findMissing )
{
    EXPECT_EQ( std::shared_ptr<ImageSource>(), m_cache.find( "missing-file.foo" ) );
}

#if OTK_USE_OPENEXR
TEST_F( TestImageSourceCache, get )
{
    std::shared_ptr<ImageSource> imageSource1( m_cache.get( m_exrPath ) );
    std::shared_ptr<ImageSource> imageSource2( m_cache.get( m_exrPath ) );
    const CacheStatistics        stats{ m_cache.getStatistics() };

    EXPECT_TRUE( imageSource1 );
    EXPECT_TRUE( imageSource2 );
    EXPECT_EQ( imageSource1, imageSource2 );
    EXPECT_EQ( 1, stats.numImageSources );
}

TEST_F( TestImageSourceCache, findPresent )
{
    std::shared_ptr<ImageSource> image = m_cache.get( m_exrPath );

    EXPECT_EQ( image, m_cache.find( m_exrPath ) );
}
#endif

#if OTK_USE_OIIO
TEST_F( TestImageSourceCache, setAdaptedReturnsAdapted )
{
    std::shared_ptr<ImageSource> adapted = std::make_shared<TiledImageSource>( createImageSource( m_jpgPath ) );
    m_cache.set( m_jpgPath, adapted );

    std::shared_ptr<ImageSource> image = m_cache.get( m_jpgPath );

    EXPECT_EQ( adapted, image );
}
#endif
