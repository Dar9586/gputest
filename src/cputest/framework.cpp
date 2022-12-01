//
// Created by dar9586 on 30/11/22.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int x, y, z;
} dim3;

dim3 blockDim;
dim3 gridDim;
dim3 blockIdx;
dim3 threadIdx;

typedef float *floatptr;

#define min(x, y) (((x) < (y)) * (x) + ((y) <= (x)) * (y))
#define max(x, y) (((x) > (y)) * (x) + ((y) >= (x)) * (y))


void sum_matrix(const floatptr m1, const floatptr m2, floatptr res, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int step_x = m / (gridDim.x * blockDim.x) + ((m % (gridDim.x * blockDim.x)) != 0);
    int step_y = n / (gridDim.y * blockDim.y) + ((n % (gridDim.y * blockDim.y)) != 0);
    int start_i = i * step_x;
    int start_j = j * step_y;
    int end_i = min(((i + 1) * step_x), m);
    int end_j = min(((j + 1) * step_y), n);

    for (int k = start_j; k < end_j; ++k) {
        for (int l = start_i; l < end_i; ++l) {
            int idx = k * m + l;
            res[idx] = m1[idx] + m2[idx];
        }
    }
}



int main(int argc, char *argv[]) {
    gridDim = {.x=1, .y=1, .z=1};
    blockDim = {.x=1, .y=1, .z=1};

    printf("Grid: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
    printf("Block: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);

    for (blockIdx.x = 1; blockIdx.x <= gridDim.x; ++blockIdx.x) {
        for (blockIdx.y = 1; blockIdx.y <= gridDim.y; ++blockIdx.y) {
            for (blockIdx.z = 1; blockIdx.z <= gridDim.z; ++blockIdx.z) {
                for (threadIdx.x = 1; threadIdx.x <= blockDim.x; ++threadIdx.x) {
                    for (threadIdx.y = 1; threadIdx.y <= blockDim.y; ++threadIdx.y) {
                        for (threadIdx.z = 1; threadIdx.z <= blockDim.z; ++threadIdx.z) {

                        }
                    }
                }
            }
        }
    }
}