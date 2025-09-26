#include <cstdio>
#include <omp.h>

int main() {
    printf("Max threads: %d\n", omp_get_max_threads()); 
    return 0;
}