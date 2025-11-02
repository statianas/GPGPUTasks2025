#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void radix_sort_04_scatter(
    // это лишь шаблон! смело меняйте аргументы и используемые буфера! можете сделать даже больше кернелов, если это вызовет затруднения - смело спрашивайте в чате
    // НЕ ПОДСТРАИВАЙТЕСЬ ПОД СИСТЕМУ! СВЕРНИТЕ С РЕЛЬС!! БУНТ!!! АНТИХАЙП!11!!1
    __global uint* input_gpu,
    __global uint* prefix_sum_accum_gpu,
    __global uint* buffer_loc_sum,
    __global uint* buffer_output_gpu,
    int offset,
    int group_num,
    int n)
{
    const int global_index = get_global_id(0);
    const int group_id = get_group_id(0);
    const int group_size = get_local_size(0);
    const int local_index = get_local_id(0); // global_index = group_size * group_id + local_index

    if (global_index >= n)
        return;

    int byte = (input_gpu[global_index] >> offset) & 3u;

    int pred_nums = 0;

    if (group_id == 0 && byte == 0) {
        pred_nums = 0;
    }
    else {
        pred_nums = prefix_sum_accum_gpu[byte * group_num + (group_id - 1)];
    }

    int idx = pred_nums + buffer_loc_sum[byte * n + local_index + group_id*GROUP_SIZE] - 1;

//    if (idx > n)
//        return;

    buffer_output_gpu[idx] = input_gpu[global_index];
}