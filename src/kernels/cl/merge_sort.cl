#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // This file helps CLion IDE to know what additional functions exists in OpenCL's extended C99
#endif

#include "helpers/rassert.cl"
#include "../defines.h"

__attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
__kernel void merge_sort(
    __global const uint* input_data,
    __global       uint* output_data,
                   const unsigned int block_size,
                   int n)
{
    const unsigned int i = get_global_id(0);

    const int start_first_list_glob = i / block_size * block_size;
    const int end_first_list = block_size / 2;

    const int index = i - start_first_list_glob;
    const bool first_list = index < end_first_list;

    int left, right, glob_mid;
    const int value = input_data[i];

    if (i >= n) {
        return;
    }

    if (!first_list) {
        left = -1;
        right = end_first_list;
    }
    else {
        left = end_first_list - 1;
        right = block_size;
    }

    while (right - left > 1) {
        int mid = (right + left) / 2;
        glob_mid = start_first_list_glob + mid;

        if (glob_mid < n) {
            int mid_value = input_data[glob_mid];

            if (mid_value < value || (first_list && mid_value == value)) {
                left = mid;
            }
            else right = mid;
        }
        else {
            right = mid;
        }
    }

    output_data[i - block_size/2 +  left + 1] = value;

}

