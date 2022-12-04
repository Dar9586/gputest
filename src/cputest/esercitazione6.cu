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


void fill_vector(floatptr vec, int len) {
    for (int i = 0; i < len; ++i) {
        vec[i] = (float) i;
    }
}

void somma(floatptr temp_res_shared,floatptr res){
    for (int step = 1; step < blockDim.x; step *= 2) {
        int remainder = threadIdx.x % (step * 2);
        if (remainder == 0) {
            temp_res_shared[threadIdx.x] += temp_res_shared[threadIdx.x + step];
        } else {
            break;
        }
    }
    if(threadIdx.x==0)
        res[blockIdx.x]=temp_res_shared[threadIdx.x];
}
void calc_temp(const floatptr v1, const floatptr v2, floatptr res, int vec_len,floatptr temp_res_shared) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    temp_res_shared[threadIdx.x] = v1[i] * v2[i];
}
int main(int argc, char *argv[]) {
    gridDim = {.x=1, .y=1, .z=1};
    blockDim = {.x=1, .y=1, .z=1};

    if (argc != 2) { exit(1); }
    int vec_len = atoi(argv[1]);
    gridDim.x=vec_len/64;
    blockDim.x=64;
    floatptr shared = (floatptr) malloc(blockDim.x * sizeof(float));
    floatptr v1 = (floatptr) malloc(vec_len * sizeof(float));
    floatptr v2 = (floatptr) malloc(vec_len * sizeof(float));
    floatptr temp_res = (floatptr) calloc(gridDim.x, sizeof(float));
    fill_vector(v1, vec_len);
    fill_vector(v2, vec_len);


    for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) {
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) {
            for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) {
                for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y) {
                        for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z) {
                            calc_temp(v1,v2,temp_res,vec_len,shared);
                        }
                    }
                }
                for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y) {
                        for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z) {
                            somma(shared,temp_res);
                        }
                    }
                }
            }
        }
    }
    float total=0;
    for (int i = 0; i < gridDim.x; ++i) {
        printf("%d: %.1f\n",i,temp_res[i]);
        total+=temp_res[i];
    }
    printf("Total: %.1f\n",total);

}