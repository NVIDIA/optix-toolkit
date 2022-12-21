#include <OptiXToolkit/Util/Exception.h>

#include <cuda_runtime.h>


int main()
{
    CUDA_CHECK( cudaFree( nullptr ) );

    cudaMemset( nullptr, 0, 128 );
    CUDA_SYNC_CHECK();

    return 0;
}
