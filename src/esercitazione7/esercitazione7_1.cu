#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

typedef float *floatptr;

void fill_vector(floatptr vec, int len) {
    for (int i = 0; i < len; ++i) {
        vec[i] = (float) i;
    }
}

int main (int argc,char*argv[]){
    if(argc!=2){exit(1);}
    int M = atoi(argv[1]);
    cublasHandle_t handle;
    floatptr h_a;         // Host array a
    floatptr d_a;         // Device array a
    floatptr h_b;         // Host array b
    floatptr d_b;         // Device array b
    float result = 0;     // Risultato finale

    /*
    [3, 10, 20] * [5, 10, 15] = 415
    */

    h_a = (floatptr)malloc (M * sizeof (*h_a));      // Alloco h_a e lo inizializzo
    h_b = (floatptr)malloc (M * sizeof (*h_b));  // Alloco h_b e lo inizializzo
    fill_vector(h_b,M);
    fill_vector(h_a,M);

    cublasCreate(&handle);               // Creo l'handle per cublas
    cudaMalloc ((void**)&d_a, M*sizeof(*h_a));       // Alloco d_a
    cudaMalloc ((void**)&d_b, M*sizeof(*h_b));       // Alloco d_b
    cublasSetVector(M,sizeof(float),h_a,1,d_a,1);    // Setto h_a su d_a
    cublasSetVector(M,sizeof(float),h_b,1,d_b,1);    // Setto h_b su d_b

    // Creazione eventi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    cublasSdot(handle,M,d_a,1,d_b,1,&result);        // Calcolo il prodotto
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("Risultato del prodotto %f in %.4f ms\n",result,elapsed);

    cudaFree (d_a);     // Dealloco d_a
    cudaFree (d_b);     // Dealloco d_b

    cublasDestroy(handle);  // Distruggo l'handle

    free(h_a);      // Dealloco h_a
    free(h_b);      // Dealloco h_b

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return EXIT_SUCCESS;
}