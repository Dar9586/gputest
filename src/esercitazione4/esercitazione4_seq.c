#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    if (argc != 2) { exit(1); }
    int N = atoi(argv[1]);
    clock_t t;
    // Alloco memoria
    int vec_size = N * sizeof(float);
    float *u = (float *) malloc(vec_size);
    float *v = (float *) malloc(vec_size);

    // Inizializzo i dati
    for (int i = 0; i < N; i++) {
        u[i] = (float)i;
        v[i] = (float)i;
    }
    t=clock();
    float sum=0;
    for (int i = 0; i < N; i++)
    {
        sum+=u[i]*v[i];
    }
    t=clock()-t;
    double time_taken = (((double)t)/CLOCKS_PER_SEC) * 1000; // in seconds
    printf("Somma di %d elementi: %f in %.4lf ms\n",N,sum,time_taken);
}