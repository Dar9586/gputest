#include <assert.h>
#include <stdio.h>
#include<cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>

void sommaCPU(float *a, float *b, float *c, int n);

__global__ void sommaGPU(float *a, float *b, float *c, int n);

int main(void) {
    float *a_h, *b_h, *c_h, *c_h2; // host data
    float *a_d, *b_d, *c_d; // device data
    int N, nBytes, i;
    printf("Addio\n");
    cuInit(1);
    printf("Mondo\n");
    int x;cudaGetDeviceCount(&x);
    printf("Devices: %d\n",x);
    dim3 gridDim, blockDim;
    cudaSetDevice(0);
    N=50;
    blockDim.x=10;
    /*printf("***\t SOMMA DI DUE VETTORI \t***\n");
    printf("Inserisci il numero degli elementi dei vettori\n");
    scanf("%d", &N);
    printf("Inserisci il numero di thread per blocco\n");
    scanf("%d", &blockDim.x);*/

//determinazione esatta del numero di blocchi
    gridDim = N / blockDim.x +
              ((N % blockDim.x) == 0 ? 0 : 1);


    nBytes = N * sizeof(float);
    a_h = (float *) malloc(nBytes);
    b_h = (float *) malloc(nBytes);
    c_h = (float *) malloc(nBytes);
    c_h2 = (float *) malloc(nBytes);
    cudaMalloc((void **) &a_d, nBytes);
    cudaMalloc((void **) &b_d, nBytes);
    cudaMalloc((void **) &c_d, nBytes);
    // inizializzo i dati
    /*Inizializza la generazione random dei vettori utilizzando l'ora attuale del sistema*/
    srand((unsigned int) time(0));

    for (i = 0; i < N; i++) {
        a_h[i] = rand() % 5 - 2;
        b_h[i] = rand() % 5 - 2;;
    }

    cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, nBytes, cudaMemcpyHostToDevice);


    //azzeriamo il contenuto del vettore c
    memset(c_h, 0, nBytes);
    cudaMemset(c_d, 0, nBytes);

//invocazione del kernel 
    sommaGPU<<<gridDim, blockDim>>>(a_d, b_d, c_d, N);

    cudaMemcpy(c_h, c_d, nBytes, cudaMemcpyDeviceToHost);

    // calcolo somma seriale su CPU
    sommaCPU(a_h, b_h, c_h2, N);


// verifica che i risultati di CPU e GPU siano uguali
// se non stampa nulla, i due vettori sono uguali 
    for (i = 0; i < N; i++) assert(c_h[i] == c_h2[i]);

    if (N < 20) {
        for (i = 0; i < N; i++)
            printf("a_h[%d]=%6.2f ", i, a_h[i]);
        printf("\n");
        for (i = 0; i < N; i++)
            printf("b_h[%d]=%6.2f ", i, b_h[i]);
        printf("\n");
        for (i = 0; i < N; i++)
            printf("c_h[%d]=%6.2f ", i, c_h[i]);
        printf("\n");
    }
    free(a_h);
    free(b_h);
    free(c_h);
    free(c_h2);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    return 0;
}

//Seriale
void sommaCPU(float *a, float *b, float *c, int n) {
    int i;
    for (i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

//Parallelo
__global__ void sommaGPU
        (float *a, float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
}
