#include <libgpu/context.h>
#include <libgpu/work_size.h>
#include <libgpu/shared_device_buffer.h>

#include <libgpu/cuda/cu/common.cu>

#include "../defines.h"

#define WARP_SIZE 32

__global__ void sum_04_local_reduction(
    const unsigned int* a,
    unsigned int* b,
    unsigned int  n)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ unsigned int local_data[GROUP_SIZE];

    if (index < n) {
        local_data[threadIdx.x] = a[index];
    } else {
        local_data[threadIdx.x] = 0;
    }

    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        unsigned int my_sum = 0;
        for (int i = 0; i < GROUP_SIZE / WARP_SIZE; ++i) {
            my_sum += local_data[i * WARP_SIZE + threadIdx.x];
        }
        local_data[threadIdx.x] = my_sum;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        unsigned int master_sum = 0;
        for (unsigned int i = 0; i < WARP_SIZE; ++i) {
            master_sum += local_data[i];
        }
        b[index / GROUP_SIZE] = master_sum;
    }
}

namespace cuda {
void sum_04_local_reduction(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &a, gpu::gpu_mem_32u &sum, unsigned int n)
{
    gpu::Context context;
    rassert(context.type() == gpu::Context::TypeCUDA, 6573652345243, context.type());
    cudaStream_t stream = context.cudaStream();
    ::sum_04_local_reduction<<<workSize.cuGridSize(), workSize.cuBlockSize(), 0, stream>>>(a.cuptr(), sum.cuptr(), n);
    CUDA_CHECK_KERNEL(stream);
}
} // namespace cuda
