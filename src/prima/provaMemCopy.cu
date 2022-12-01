#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

int main(void) {
    float *a_h, *b_h; // host data
    float *a_d, *b_d; // device data
    int N = 10, nBytes, i;
    nBytes = N * sizeof(float);
    a_h = (float *) malloc(nBytes);
    b_h = (float *) malloc(nBytes);
    cudaMalloc((void **) &a_d, nBytes);
    cudaMalloc((void **) &b_d, nBytes);
    for (i = 0; i < N; i++)
        a_h[i] = 100 + i;
    cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, a_d, nBytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(b_h, b_d, nBytes, cudaMemcpyDeviceToHost);
    for (i = 0; i < N; i++) assert(a_h[i] == b_h[i]);

    for (i = 0; i < N; i++)
        printf("a_h[%d]=%6.2f ", i, a_h[i]);
    printf("\n");
    for (i = 0; i < N; i++)
        printf("b_h[%d]=%6.2f ", i, b_h[i]);
    printf("\n");

    free(a_h);
    free(b_h);
    cudaFree(a_d);
    cudaFree(b_d);
    return 0;
}
