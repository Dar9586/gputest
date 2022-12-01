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


void calc_prod(const floatptr v1,const floatptr v2, floatptr res,bool*written,int vec_len){
    int total_blocks=blockDim.x*gridDim.x;
    int itemPerThread=vec_len/total_blocks;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int startIdx=i*itemPerThread;
    int stopIdx=(i+1)*itemPerThread;
    for (int j = startIdx; j < stopIdx; ++j) {
        if(written[j])fprintf(stderr,"Duplicate %d in thread %d\n",j,i);
        written[j]= true;
    }
    for (int j = startIdx; j < stopIdx; ++j) {
        res[i]+=v1[j]+v2[j];
    }

}

int main(int argc, char *argv[]) {
    gridDim = {.x=1, .y=1, .z=1};
    blockDim = {.x=1, .y=1, .z=1};

    if (argc != 4) { exit(1); }
    int vec_len = atoi(argv[1]);
    gridDim.x=atoi(argv[2]);
    blockDim.x=atoi(argv[3]);
    floatptr v1 = (floatptr) malloc(vec_len * sizeof(float));
    floatptr v2 = (floatptr) malloc(vec_len * sizeof(float));
    bool* written = (bool*) malloc(vec_len * sizeof(bool));
    floatptr temp_res = (floatptr) calloc(gridDim.x*blockDim.x, sizeof(float));
    fill_vector(v1, vec_len);
    fill_vector(v2, vec_len);



    for (blockIdx.x = 0; blockIdx.x < gridDim.x; ++blockIdx.x) {
        for (blockIdx.y = 0; blockIdx.y < gridDim.y; ++blockIdx.y) {
            for (blockIdx.z = 0; blockIdx.z < gridDim.z; ++blockIdx.z) {
                for (threadIdx.x = 0; threadIdx.x < blockDim.x; ++threadIdx.x) {
                    for (threadIdx.y = 0; threadIdx.y < blockDim.y; ++threadIdx.y) {
                        for (threadIdx.z = 0; threadIdx.z < blockDim.z; ++threadIdx.z) {
                            calc_prod(v1,v2,temp_res,written,vec_len);
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < vec_len; ++i) {
        if(!written[i])printf("NO %d\n",i);
    }

}