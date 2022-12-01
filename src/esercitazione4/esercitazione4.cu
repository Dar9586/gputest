#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void prodotto(const float u[], const float v[], float w[], int N) {
    // Ottengo il numero di thread usati
    int grid_size = gridDim.x * blockDim.x;
    // Controllo quanti elementi deve sommare ogni thread
    int data_size = N / grid_size;
    // Ottengo l'indice del thread
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // Trovo l'indice finale in modo che non vada fuori range
    int to_1 = (index + 1) * data_size;
    int to = (to_1 < N) * to_1 + (N <= to_1) * N; // min tra N e to_1 senza if
    // Eseguo i calcoli
    for (int i = index * data_size; i < to; ++i) {
        w[i] = u[i] * v[i];
    }
}


int main(int argc, char *argv[]) {
    if (argc != 4) { exit(1); }
    int N = atoi(argv[1]);
    int grid_x = atoi(argv[2]);
    int block_x = atoi(argv[3]);
    float *du, *dv, *dw;

    // Alloco memoria
    int vec_size = N * sizeof(float);
    float *u = (float *) malloc(vec_size);
    float *v = (float *) malloc(vec_size);
    float *w = (float *) malloc(vec_size);
    cudaMalloc(&du, vec_size);
    cudaMalloc(&dv, vec_size);
    cudaMalloc(&dw, vec_size);

    // Inizializzo i dati
    for (int i = 0; i < N; i++) {
        u[i] = (float)i;
        v[i] = (float)i;
    }

    dim3 gridDim(grid_x, 1, 1);
    dim3 blockDim(block_x, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Copio i dati
    cudaMemcpy(du, u, vec_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v, vec_size, cudaMemcpyHostToDevice);

    // CHIAMO KERNEL
    cudaEventRecord(start);
    prodotto<<<gridDim, blockDim>>>(du, dv, dw, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Copio i dati
    cudaMemcpy(w, dw, vec_size, cudaMemcpyDeviceToHost);

    // Calcolo tempo richiesto
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    // Eseguo la somma sull'host
    float sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += w[i];
    }

    // Stampe

    printf("Prodotto scalare: %.2f\n", sum);
    printf("Tempo richiesto: %f ms\n", elapsed);

    // Libero memoria
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(u);
    free(v);
    free(w);
    cudaFree(du);
    cudaFree(dv);
    cudaFree(dw);

    return 0;
}