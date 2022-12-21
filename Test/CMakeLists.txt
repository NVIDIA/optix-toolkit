include(BuildConfig)

find_package( CUDAToolkit 10.0 REQUIRED )

add_executable( test
  test.cpp
)
target_link_libraries( test
  OptiXToolkit::Util
  CUDA::cudart
  )
