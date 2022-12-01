#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

typedef float *floatptr;

__global__ void calc_temp(const floatptr v1, const floatptr v2, floatptr res, int vec_len) {
    extern __shared__ int temp_res[];
    int total_blocks = blockDim.x * gridDim.x;
    int itemPerThread = vec_len / total_blocks;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int startIdx = i * itemPerThread;
    int stopIdx = (i + 1) * itemPerThread;
    // Calcola i risultati parziali
    for (int j = startIdx; j < stopIdx; ++j) {
        temp_res[threadIdx.x] += v1[j] * v2[j];
    }
    __syncthreads();
    // Somma i risultati di tutti i thread del blocco
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (threadIdx.x < step)
            temp_res[threadIdx.x] += temp_res[threadIdx.x + step];
    }
    if (threadIdx.x == 0)
        res[blockIdx.x] = temp_res[0];
}

void fill_vector(floatptr vec, int len) {
    for (int i = 0; i < len; ++i) {
        vec[i] = (float) i;
    }
}


int main(int argc, char *argv[]) {
    floatptr v1_dev, v2_dev, res_dev;

    if (argc != 4) { exit(1); }
    int vec_len = atoi(argv[1]);
    // Definizione dimensione griglia
    dim3 grid_dim(atoi(argv[2]));
    dim3 block_dim(atoi(argv[3]));
    // Creazione eventi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocazione e inizializzazione memoria host
    floatptr v1 = (floatptr) malloc(vec_len * sizeof(float));
    floatptr v2 = (floatptr) malloc(vec_len * sizeof(float));
    floatptr res = (floatptr) malloc(grid_dim.x * sizeof(float));
    fill_vector(v1, vec_len);
    fill_vector(v2, vec_len);

    // Allocazione e inizializzazione memoria device
    cudaMalloc(&v1_dev, vec_len * sizeof(float));
    cudaMalloc(&v2_dev, vec_len * sizeof(float));
    cudaMalloc(&res_dev, grid_dim.x * sizeof(float));
    cudaMemcpy(v1_dev, v1, vec_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v2_dev, v2, vec_len * sizeof(float), cudaMemcpyHostToDevice);

    // Esecuzione kernel
    cudaEventRecord(start);
    calc_temp<<<grid_dim, block_dim, block_dim.x>>>(v1_dev, v2_dev, res_dev, vec_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calcolo risultato finale dai risultati parziali dei blocchi
    cudaMemcpy(res, res_dev, grid_dim.x * sizeof(float), cudaMemcpyDeviceToHost);
    float total = 0;
    for (int i = 0; i < grid_dim.x; ++i) {
        total += res[i];
    }

    // Stampa
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Total: %f in %.4f ms\n", total,elapsed);

    // Free
    cudaFree(v1_dev);
    cudaFree(v2_dev);
    cudaFree(v2_dev);
    free(v1);
    free(v2);
}

