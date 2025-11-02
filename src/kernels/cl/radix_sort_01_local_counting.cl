#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_01_local_counting(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global const uint* data,
    __global uint* buffer_loc_sum,
    __global uint* buffer_global_sum,
    int offset,
    int n)
{
    const int global_index = get_global_id(0);
    const int group_id = get_group_id(0);
    const int group_size = get_local_size(0);
    const int local_index = get_local_id(0); // global_index = group_size * group_id + local_index

//    if (global_index > n)
//        return;

    __local int sum[4 * GROUP_SIZE];

    for (int byte = 0; byte < 4; byte++) {
        if (global_index < n && ((data[global_index] >> offset) & 3u) == byte) {
            sum[byte*GROUP_SIZE + local_index] = 1;
        } else {
            sum[byte*GROUP_SIZE + local_index] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int byte = 0; byte < 4; byte++) {
        int step = 2;
        while (step / 2 < GROUP_SIZE) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if ((local_index < GROUP_SIZE / step)) {
                int idx = byte*GROUP_SIZE + (local_index + 1) * step - 1;
                sum[idx] = sum[idx] + sum[idx - step / 2];
            }
            step *= 2;
        }
        step /= (2 * 2);

        while (step > 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if ((local_index < GROUP_SIZE / step - 1)) {
                int idx = byte*GROUP_SIZE + (local_index + 1) * step + step / 2 - 1;
                sum[idx] = sum[idx] + sum[idx - step / 2];
            }
            step /= 2;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int byte = 0; byte < 4; byte++) {
       if (global_index < n) buffer_loc_sum[GROUP_SIZE * group_id + n * byte + local_index] = sum[byte*GROUP_SIZE + local_index];
    }

    if (local_index < 4) {
        int byte = local_index;
        buffer_global_sum[group_id * 4 + byte] = sum[byte * GROUP_SIZE + GROUP_SIZE - 1];
    }


}
