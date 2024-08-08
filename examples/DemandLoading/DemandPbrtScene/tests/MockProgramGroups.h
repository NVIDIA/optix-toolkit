#pragma once

#include <ProgramGroups.h>

#include <gmock/gmock.h>

#include <memory>

namespace demandPbrtScene {
namespace testing {

class MockProgramGroups : public ::testing::StrictMock<ProgramGroups>
{
  public:
    ~MockProgramGroups() override = default;

    MOCK_METHOD( void, cleanup, (), ( override ) );
    MOCK_METHOD( uint_t, getRealizedMaterialSbtOffset, (const GeometryInstance&), ( override ) );
    MOCK_METHOD( void, initialize, (), ( override ) );
};

using MockProgramGroupsPtr = std::shared_ptr<MockProgramGroups>;

inline MockProgramGroupsPtr createMockProgramGroups()
{
    return std::make_shared<MockProgramGroups>();
}

}  // namespace testing
}  // namespace demandPbrtScene
