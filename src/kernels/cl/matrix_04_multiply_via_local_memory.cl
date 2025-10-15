#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#include "../defines.h"


__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void matrix_04_multiply_via_local_memory(
                       __global const float* a, // rows=h x cols=k
                       __global const float* b, // rows=k x cols=w
                       __global       float* c, // rows=h x cols=w
                                unsigned int w,
                                unsigned int h,
                                unsigned int k)
{

    const int SIZE_X = 16;
    const unsigned int index_x = get_global_id(0);
    const unsigned int index_y = get_global_id(1);

    const unsigned int index_x_local = get_local_id(0);
    const unsigned int index_y_local = get_local_id(1);

    __local float local_a[SIZE_X][SIZE_X];
    __local float local_b[SIZE_X][SIZE_X];

    float sum = 0;

    for (int tile = 0; tile * SIZE_X < k; ++tile) {

        if (index_y < h && (tile * SIZE_X + index_x_local) < k)
            local_a[index_y_local][index_x_local] = a[index_y * k + (tile * SIZE_X + index_x_local)];
        else
            local_a[index_y_local][index_x_local] = 0;

        if ((tile * SIZE_X + index_y_local) < k && index_x < w)
            local_b[index_y_local][index_x_local] = b[(tile * SIZE_X + index_y_local) * w + index_x]; //
        else
            local_b[index_y_local][index_x_local] = 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < SIZE_X; ++k) {
            sum += local_a[index_y_local][k] * local_b[k][index_x_local];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (index_y < h && index_x < w)
        c[index_y * w + index_x] = sum;
}
