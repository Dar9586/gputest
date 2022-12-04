#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <math.h>

typedef float *floatptr;

#define min(x, y) (((x) < (y)) * (x) + ((y) <= (x)) * (y))
#define max(x, y) (((x) > (y)) * (x) + ((y) >= (x)) * (y))

void fill_matrix(floatptr m1, floatptr m2, int m, int n) {
    for (int i = 0; i < m * n; ++i) {
        m1[i] = (float) i;
        m2[i] = (float) i;
    }
}

void better_ratio(int m,int n,int*block_x,int*block_y){
    float ratio=(float)m/(float)n;
    float best_delta=9999;
    int approx_i;
    int approx_j;
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            if(i*j!=64)continue;
            float new_approx= (float)i/(float)j;
            float delta=abs(ratio-new_approx);
            if(delta<best_delta){
                best_delta=delta;
                approx_i=i;
                approx_j=j;
            }
        }
    }
    *block_x=approx_i;
    *block_y=approx_j;
}

__global__ void sum_matrix(const floatptr m1, const floatptr m2, floatptr res, int m, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx=j*m+i;
    res[idx] = m1[idx] + m2[idx];
}

//
// Created by dar9586 on 17/11/22.
//


int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Uso: %s <M> <N>\n", argv[0]);
        exit(1);
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int block_x,block_y;
    better_ratio(M,N,&block_x,&block_y);
    floatptr m1, m2, res;
    floatptr m1_dev, m2_dev, res_dev;
    size_t vec_size = (N * M) * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Alloco dati
    m1 = (floatptr) malloc(vec_size);
    m2 = (floatptr) malloc(vec_size);
    res = (floatptr) malloc(vec_size);
    cudaMalloc(&m1_dev, vec_size);
    cudaMalloc(&m2_dev, vec_size);
    cudaMalloc(&res_dev, vec_size);

    fill_matrix(m1, m2, M, N);

    cudaMemcpy(m1_dev, m1, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(m2_dev, m2, vec_size, cudaMemcpyHostToDevice);

    dim3 grid_dim(M/block_x+((M%block_x)!=0), N/block_y+((N%block_y)!=0));
    dim3 block_dim(block_x,block_y);
    printf("Used grid: (%d * %d), block: (%d * %d)\n",  grid_dim.x,grid_dim.y,block_dim.x,block_dim.y);
    cudaEventRecord(start);
    sum_matrix<<<grid_dim, block_dim>>>(m1_dev, m2_dev, res_dev, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    cudaMemcpy(res, res_dev, vec_size, cudaMemcpyDeviceToHost);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("Tempo richiesto: %f ms\n", elapsed);
    for (int i = 0; i < M * N; ++i) {
        if(res[i]!=i*2)printf("NO %d",i);
    }
    if (M <= 10 && N <= 10) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("%f ", res[i * M + j]);
            }
            printf("\n");
        }
    }
    printf("\n");
    // Libero dati
    free(m1);
    free(m2);
    free(res);
    cudaFree(m1_dev);
    cudaFree(m2_dev);
    cudaFree(res_dev);
}

