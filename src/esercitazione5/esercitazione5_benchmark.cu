#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

typedef float *floatptr;

#define min(x, y) (((x) < (y)) * (x) + ((y) <= (x)) * (y))
#define max(x, y) (((x) > (y)) * (x) + ((y) >= (x)) * (y))

void fill_matrix(floatptr m1, floatptr m2, int m, int n) {
    for (int i = 0; i < m * n; ++i) {
        m1[i] = (float) i;
        m2[i] = (float) i;
    }
}

bool check_matrix(const floatptr m1,int m,int n){
    for (int i = 0; i < m * n; ++i) {
        if(m1[i]!=i*2)return false;
    }
    return true;
}

__global__ void sum_matrix(const floatptr m1, const floatptr m2, floatptr res, int m, int n) {
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

    int gridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&gridSize, &blockSize, sum_matrix);
    printf("Suggested grid: %d, block: %d\n", gridSize, blockSize);

    float times[5];
    for (int grid_x = 0; grid_x <= 129; grid_x = grid_x == 0 ? 1 : grid_x * 2) {
        for (int grid_y = 0; grid_y <= 129; grid_y = grid_y == 0 ? 1 : grid_y * 2) {
            if (grid_x == 0)grid_x = 40;
            if (grid_y == 0)grid_y = 40;
            for (int block_x = 1; block_x < 1025; block_x *= 2) {
                for (int block_y = 1; block_y < 1025; block_y *= 2) {
                    int blockDim = block_x * block_y;
                    if (blockDim % 32 != 0)continue;
                    if (blockDim > 1024)continue;
                    dim3 grid_dim(grid_x, grid_y);
                    dim3 block_dim(block_x, block_y);
                    float allTimes = 0;
                    printf("%d|%d|%d|%d|",grid_x, grid_y,block_x, block_y);
                    /*printf("Used grid: (%3d * %3d) = %5d, block: (%4d * %4d) = %4d :   [", grid_x, grid_y,
                           grid_x * grid_y, block_x, block_y,
                           block_x * block_y);*/
                    cudaMemset(res_dev,0,vec_size);
                    for (int attempt = 0; attempt < 5; ++attempt) {
                        cudaEventRecord(start);
                        sum_matrix<<<grid_dim, block_dim>>>(m1_dev, m2_dev, res_dev, M, N);
                        cudaEventRecord(stop);
                        cudaEventSynchronize(stop);
                        cudaEventElapsedTime(&times[attempt], start, stop);
                        allTimes += times[attempt];
                        printf("%.4f|",times[attempt]);
                        //printf("%4.4f ms ,", times[attempt]);
                    }
                    cudaMemcpy(res,res_dev,vec_size,cudaMemcpyDeviceToHost);
                    printf("%4.4f - %d\n", allTimes / 5.0f, check_matrix(res,M,N));
                }
            }

            if (grid_x == 40)grid_x = 0;
            if (grid_y == 40)grid_y = 0;

        }
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

