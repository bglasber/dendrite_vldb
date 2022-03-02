#include "tsc.h"
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

int main() {
    uint64_t start;
    uint64_t end;
    for( int i = 0; i < 100; i++ ) {
        start = bench_start();
        sleep(1);
        end = bench_end();
        printf( "%ld\n", end-start );
    }
}
