#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl> // helps IDE with OpenCL builtins
#endif

#include "../defines.h"



__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_02_transpose_coalesced_via_local_memory(
                       __global const float* matrix,            // w x h
                       __global       float* transposed_matrix, // h x w
                                unsigned int w,
                                unsigned int h)
{
    const int SIZE_X = 16;
    const unsigned int index_x = get_global_id(0);
    const unsigned int index_y = get_global_id(1);

    const unsigned int index_group_x = get_group_id(0);
    const unsigned int index_group_y = get_group_id(1);

    const unsigned int index_x_loc = get_local_id(0);
    const unsigned int index_y_loc = get_local_id(1);


    __local float local_data[SIZE_X][SIZE_X + 1];

    if (w > index_x && h > index_y) {
        local_data[index_y_loc][index_x_loc] = matrix[index_y * w + index_x];
    }
    else {
        local_data[index_y_loc][index_x_loc] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if ((index_group_x * SIZE_X + index_y_loc < w) && (index_group_y * SIZE_X + index_x_loc < h))
        transposed_matrix[(index_group_x * SIZE_X + index_y_loc) * h + index_group_y * SIZE_X + index_x_loc] = local_data[index_x_loc][index_y_loc];

}
