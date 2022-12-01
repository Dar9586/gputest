#include <cstdio>
#include <cuda.h>

int main() {
    int version;
    cuDriverGetVersion(&version);
    printf("%d\n",version);
    return 0;
}
