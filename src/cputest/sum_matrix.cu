//
// Created by dar9586 on 07/12/22.
//
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

void fill_matrix(float* m1, float* m2, int m, int n) {
    for (int i = 0; i < m * n; ++i) {
        m1[i] = (float) i;
        m2[i] = (float) i;
    }
}


int main(int argc,char*argv[]){
    if (argc != 3) {
        printf("Uso: %s <M> <N>\n", argv[0]);
        exit(1);
    }
    clock_t t;
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    float*mat_1= (float*)malloc(M*N*sizeof(float));
    float*mat_2= (float*)malloc(M*N*sizeof(float));
    float*mat_out= (float*)malloc(M*N*sizeof(float));

    fill_matrix(mat_1,mat_2,M,N);
    t=clock();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            mat_out[i*M+j]=mat_1[i*M+j]+mat_2[i*M+j];
        }
    }
    t=clock()-t;
    double time_taken = (((double)t)/CLOCKS_PER_SEC) * 1000; // in seconds
    printf("%d\t%d\t%.4lf\n",M,N,time_taken);
}