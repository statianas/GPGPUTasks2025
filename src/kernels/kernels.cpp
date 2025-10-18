#include "kernels.h"

#include "cl/generated_kernels/aplusb.h"
#include "cl/generated_kernels/fill_buffer_with_zeros.h"
#include "cl/generated_kernels/radix_sort_01_local_counting.h"
#include "cl/generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction.h"
#include "cl/generated_kernels/radix_sort_03_global_prefixes_scan_accumulation.h"
#include "cl/generated_kernels/radix_sort_04_scatter.h"

#include "vk/generated_kernels/aplusb_comp.h"
#include "vk/generated_kernels/fill_buffer_with_zeros_comp.h"
#include "vk/generated_kernels/radix_sort_01_local_counting_comp.h"
#include "vk/generated_kernels/radix_sort_02_global_prefixes_scan_sum_reduction_comp.h"
#include "vk/generated_kernels/radix_sort_03_global_prefixes_scan_accumulation_comp.h"
#include "vk/generated_kernels/radix_sort_04_scatter_comp.h"

#ifndef CUDA_SUPPORT
namespace cuda {
void aplusb(const gpu::WorkSize& workSize,
    const gpu::gpu_mem_32u& a, const gpu::gpu_mem_32u& b, gpu::gpu_mem_32u& c, unsigned int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
void merge_sort(const gpu::WorkSize &workSize,
    const gpu::gpu_mem_32u &input_data, gpu::gpu_mem_32u &output_data, int sorted_k, int n)
{
    // dummy implementation if CUDA_SUPPORT is disabled
    rassert(false, 54623523412413);
}
} // namespace cuda
#endif

namespace ocl {
const ocl::ProgramBinaries& getAplusB()
{
    return opencl_binaries_aplusb;
}

const ProgramBinaries& getMergeSort()
{
    return opencl_binaries_fill_buffer_with_zeros;
}
} // namespace ocl

namespace avk2 {
const ProgramBinaries& getAplusB()
{
    return vulkan_binaries_aplusb_comp;
}

const ProgramBinaries& getMergeSort()
{
    return vulkan_binaries_fill_buffer_with_zeros_comp;
}
} // namespace avk2
