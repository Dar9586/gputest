#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

typedef float *floatptr;

__global__ void calc_temp(const floatptr v1, const floatptr v2, floatptr temp_res, int vec_len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    temp_res[i] += v1[i] * v2[i];
    __syncthreads();
    // Somma i risultati di tutti i thread del blocco
    for (int step = 1; step < blockDim.x; step *= 2) {
        int remainder = threadIdx.x % (step * 2);
        if (remainder == 0) {
            temp_res[i] += temp_res[i + step];
        } else {
            break;
        }
    }
}

void fill_vector(floatptr vec, int len) {
    for (int i = 0; i < len; ++i) {
        vec[i] = (float) i;
    }
}


int main(int argc, char *argv[]) {
    floatptr v1_dev, v2_dev, temp_res_dev;

    if (argc != 2) { exit(1); }
    int vec_len = atoi(argv[1]);
    // Definizione dimensione griglia
    dim3 grid_dim(vec_len / 64 + ((vec_len % 64) != 0));
    dim3 block_dim(64);
    // Creazione eventi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocazione e inizializzazione memoria host
    floatptr v1 = (floatptr) malloc(vec_len * sizeof(float));
    floatptr v2 = (floatptr) malloc(vec_len * sizeof(float));
    floatptr temp_res = (floatptr) malloc(grid_dim.x * block_dim.x * sizeof(float));
    fill_vector(v1, vec_len);
    fill_vector(v2, vec_len);

    // Allocazione e inizializzazione memoria device
    cudaMalloc(&v1_dev, vec_len * sizeof(float));
    cudaMalloc(&v2_dev, vec_len * sizeof(float));
    cudaMalloc(&temp_res_dev, grid_dim.x * block_dim.x * sizeof(float));
    cudaMemset(temp_res_dev, 0, grid_dim.x * block_dim.x);
    cudaMemcpy(v1_dev, v1, vec_len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v2_dev, v2, vec_len * sizeof(float), cudaMemcpyHostToDevice);

    // Esecuzione kernel
    cudaEventRecord(start);
    calc_temp<<<grid_dim, block_dim>>>(v1_dev, v2_dev, temp_res_dev, vec_len);
    // Calcolo risultato finale dai risultati parziali dei blocchi
    cudaMemcpy(temp_res, temp_res_dev, grid_dim.x * block_dim.x * sizeof(float), cudaMemcpyDeviceToHost);
    float total = 0;
    for (int i = 0; i < grid_dim.x; ++i) {
        total += temp_res[i * block_dim.x];
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Stampa
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("\nTotal: %f in %.4f ms\n", total, elapsed);

    // Free
    cudaFree(v1_dev);
    cudaFree(v2_dev);
    cudaFree(v2_dev);
    free(v1);
    free(v2);
}

