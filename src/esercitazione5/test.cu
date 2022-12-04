//
// Created by dar9586 on 01/12/22.
//
#include <cstdio>
#include <cstdlib>
#include <math.h>
void better_ratio(int m,int n,int*block_x,int*block_y){
    float ratio=(float)m/(float)n;
    float best_delta=9999;
    int approx_i;
    int approx_j;
    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            if(i*j!=64)continue;
            float new_approx= (float)i/(float)j;
            float delta=abs(ratio-new_approx);
            if(delta<best_delta){
                best_delta=delta;
                approx_i=i;
                approx_j=j;
            }
        }
    }
    *block_x=approx_i;
    *block_y=approx_j;
}
void test(int m,int n){
    int grid_x,grid_y;
    better_ratio(m,n,&grid_x,&grid_y);
    printf("Better ratio for (%d x %d) = (%d x %d)\n",m,n,grid_x,grid_y);
}
int main(){
    test(1024,1024);
    test(1024,2048);
}