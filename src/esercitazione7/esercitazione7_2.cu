#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef float *floatptr;

#define fsize(x) ((x)*sizeof(float))

void fill_matrix(floatptr vec, int m, int n) {
    for (int i = 0; i < m * n; ++i) {
        vec[i] = (float) i;
    }
}

void fill_vector(floatptr vec, int n) {
    for (int i = 0; i < n; ++i) {
        vec[i] = (float) i;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) { exit(1); }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int mat_size = M * N;
    int vec_size = N;
    cublasHandle_t handle;
    floatptr mat_dev, vec_dev, out_dev;
    float result = 0;     // Risultato finale

    /*
    [3, 10, 20] * [5, 10, 15] = 415
    */

    floatptr mat_host = (floatptr) malloc(fsize(mat_size));      // Alloco h_a e lo inizializzo
    floatptr vec_host = (floatptr) malloc(fsize(vec_size));  // Alloco h_b e lo inizializzo
    floatptr out_host = (floatptr) malloc(fsize(vec_size));  // Alloco h_b e lo inizializzo
    fill_matrix(mat_host, M, N);
    fill_vector(vec_host, N);

    cublasCreate(&handle);               // Creo l'handle per cublas
    cudaMalloc((void **) &mat_dev, fsize(mat_size));       // Alloco d_a
    cudaMalloc((void **) &vec_dev, fsize(vec_size));       // Alloco d_b
    cudaMalloc((void **) &out_dev, fsize(vec_size));       // Alloco d_b
    cublasSetMatrix(M, N, sizeof(float), mat_host, M, mat_dev, M);
    cublasSetVector(N, sizeof(float), vec_host, 1, vec_dev, 1);
    float scalar = 1;
    float beta = 0;
    // Creazione eventi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasSgemv(handle, CUBLAS_OP_N, M, N, &scalar, mat_dev, M, vec_dev, 1, &beta, out_dev, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cublasGetVector(N, sizeof(float), out_dev, 1, out_host, 1);

    if (N <= 10) {
        for (int i = 0; i < N; ++i) {
            printf("%.1f, ", out_host[i]);
        }
        printf("\n");
    }

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Risultato del prodotto %f in %.4f ms\n", result, elapsed);

    cudaFree(vec_dev);
    cudaFree(out_dev);
    cudaFree(mat_host);

    cublasDestroy(handle);  // Distruggo l'handle

    free(vec_host);
    free(out_host);
    free(mat_host);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}