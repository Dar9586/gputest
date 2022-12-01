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
    if (argc != 7) {
        printf("Uso: %s <M> <N> <Grid X> <Grid y> <Block x> <Block y>\n", argv[0]);
        exit(1);
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int grid_x = atoi(argv[3]);
    int grid_y = atoi(argv[4]);
    int block_x = atoi(argv[5]);
    int block_y = atoi(argv[6]);
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
    printf("Used grid: (%d * %d) = %d, block: (%d * %d) = %d\n", grid_x, grid_y, grid_x * grid_y, block_x, block_y,
           block_x * block_y);
    dim3 grid_dim(grid_x, grid_y);
    dim3 block_dim(block_x, block_y);
    cudaEventRecord(start);
    sum_matrix<<<grid_dim, block_dim>>>(m1_dev, m2_dev, res_dev, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    cudaMemcpy(res, res_dev, vec_size, cudaMemcpyDeviceToHost);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    printf("Tempo richiesto: %f ms\n", elapsed);
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

