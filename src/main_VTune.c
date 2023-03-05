#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "tsc_x86.h"
#include "nnmf.h"
#include "nnmf_common.h"
#include "init_w.h"
#include "utils.h"

// instructions to run it on Windows:
// replace:
// aligned_alloc(...) with manual_align_malloc(...)
// free(...) with manual_free(...)
// compile with gcc main_VTune.c -O3 -ffast-math -march=native -o asl.exe init_w.h init_w.c utils.h utils.c nnmf_common.h nnmf/opt_1.c nnmf.h nnmf.c -lm

#ifndef M
#   define M 1280
#endif

// threshold value
#ifndef THESHOLD
#   define THRESHOLD 10
#endif

int main(int argc, char **argv) {
    myInt64 start, stop;
    start = start_tsc();
    int m = M;
    int n = m;
    int r = m;

    float_type* V = (float_type*)manual_align_malloc(n * m * sizeof(float_type));
    float_type* W = (float_type*)manual_align_malloc(m * r * sizeof(float_type));
    float_type* H = (float_type*)manual_align_malloc(n * r * sizeof(float_type));

    init_V(V, m, n);
    init_V(W, m, r);
    init_V(H, r, n);

    float_type threshold = THRESHOLD;
    float_type norm;
    long loop_iterations;
    int num_iterations = 5;

    nnmf(V, W, H, m, n, r, &loop_iterations, 0, threshold, num_iterations);

    manual_free(W);
    manual_free(H);
    manual_free(V);
    stop = stop_tsc(start);
    printf("%d, %d, %d, %lf, %lf, %ld\n", m, n, r, threshold, (double)stop, loop_iterations);

    return 0;
}

