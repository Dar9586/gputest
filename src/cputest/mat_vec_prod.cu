#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>

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
    float result;     // Risultato finale
    clock_t t;
    floatptr mat_host = (floatptr) malloc(fsize(mat_size));  // Alloco h_a e lo inizializzo
    floatptr vec_host = (floatptr) malloc(fsize(N));  // Alloco h_b e lo inizializzo
    floatptr out_host = (floatptr) calloc(M, sizeof(float));  // Alloco h_b e lo inizializzo
    fill_matrix(mat_host, M, N);
    fill_vector(vec_host, N);
    t=clock();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            out_host[i]+=mat_host[i*M+j]*vec_host[j];
        }
    }
    t=clock()-t;
    double time_taken = (((double)t)/CLOCKS_PER_SEC) * 1000; // in seconds
    printf("%d\t%d\t%.4f\n", M,N,time_taken);
    free(vec_host);
    free(out_host);
    free(mat_host);


    return EXIT_SUCCESS;
}