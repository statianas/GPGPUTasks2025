#ifdef __CLION_IDE__
#include "libgpu/opencl/cl/clion_defines.cl" // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void prefix_sum_02_prefix_accumulation(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global uint* pow2_sum, // pow2_sum[i] = sum[i*2^pow2; 2*i*2^pow2)
    int n,
    int stride,
    int offset_prev
    )
{
    const int index = get_global_id(0);

    const int offset = offset_prev + (n + (1 << (stride-1)) - 1)/(1 << (stride - 1)) ;

    const int size_loc_prev = (n + (1 << (stride-1)) - 1)/(1 << (stride - 1)) ;

    const int size_loc = (n + (1 << stride) - 1)/(1 << stride) ;

    if (index >= size_loc)
        return;

    //    int index_sum = n - 1 - index * (1 << stride);

    const int lev = offset_prev + size_loc_prev - 1 - index * 2;

    const int prav = lev - 1;

    if ((prav >= offset_prev) && (size_loc - index - 1 - 1 >= 0))
        pow2_sum[prav] += pow2_sum[offset + size_loc - index - 1 - 1];
    pow2_sum[lev] = pow2_sum[offset + size_loc - index - 1];

}
