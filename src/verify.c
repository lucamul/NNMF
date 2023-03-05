#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

#include "tsc_x86.h"
#include "nnmf.h"
#include "nnmf_common.h"
#include "init_w.h"
#include "utils.h"

#define VERIFICATION_INLINE
#include "nnmf/basic.c"
#include "nnmf/inline.c"
#include "nnmf/without_transposes.c"
#include "nnmf/opt_1.c"
#include "nnmf/opt_2.c"
#include "nnmf/opt_3.c"
#include "nnmf/opt_4.c"
#include "nnmf/opt_5.c"
#include "nnmf/opt_6.c"
#include "nnmf/opt_7.c"

// ratio m/n of the matrix sizes
// M_N_RATIO = 1 for square matrices
// M_N_RATIO > 1 for non-square matrices, e.g. M_N_RATIO = 1.5 would mean that e.g. m = 3, n = 2
#define M_N_RATIO 1

// matrix size m
// set it to a multiple of block size - rn set to 16 - to avoid stragglers
#define M 320

// threshold value for running the nnmf functions
#define THRESHOLD 50

// threshold value for comparison of matrices
// needs to be set to a value slightly larger than 0 if we apply transformation that break the IEEE standard
#define EPS 0.01

// L3 cache size
#define CACHE_SIZE 1<<21

/**
 * @brief copies the elements of X to Y
 *
 * in: X     - m x n matrix X
 * in: m, n  - matrix dimensions
 * in/out: Y - m x n matrix Y
 */
void copy_array(float_type* X, float_type* Y, const int m, const int n) {
    for (int i = 0; i < m * n; ++i) {
        Y[i] = X[i];
    }
}

/**
 * @brief compares the absolute difference elements of X minus elements of Y to eps
 * results in an assertion error if the absolute difference is > eps
 *
 * in: X    - m x n matrix X
 * in: Y    - m x n matrix Y
 * in: m, n - matrix dimensions
 * in: eps  - threshold
 */
void compare_results(float_type* X, float_type* Y, const int m, const int n, const double eps) {
    float_type max_abs_value = 0.0;
    for (int i = 0; i < m * n; ++i) {
        float_type abs_value = fabs(X[i] - Y[i]);
        max_abs_value = fmax(abs_value, max_abs_value);
        if (abs_value > eps) {
            printf("fabs(X[i] - Y[i]): %lf\n", abs_value);
        }
        assert(abs_value <= eps);
    }
    printf("max difference: %lf\n", max_abs_value);
}

/**
 * @brief makes sure that the cache is cold by allocating a vector of the size of the L3 cache
 * such that the other values stored in cache are evicted
 */
void make_cache_cold(float_type* A) {
    for (int i = 0; i < CACHE_SIZE / sizeof(float_type); ++i) {
        A[i] = i;
    }
}

int main(void) {
    srand(11);
    float_type* cold_cache_array = (float_type*)malloc(CACHE_SIZE);

    int m = M;
    int n = m * M_N_RATIO;
    int r = m * M_N_RATIO;
    int threshold = THRESHOLD;
    double eps = EPS;
    myInt64 start, stop;

    long loop_iterations;

    float_type norm_a;
    float_type norm_b;

    float_type* V_a = (float_type*) aligned_alloc(64, n * m * sizeof(float_type));
    float_type* V_b = (float_type*) aligned_alloc(64, n * m * sizeof(float_type));

    float_type* W_a = (float_type*) aligned_alloc(64, m * r * sizeof(float_type));
    float_type* W_b = (float_type*) aligned_alloc(64, m * r * sizeof(float_type));

    float_type* H_a = (float_type*) aligned_alloc(64, n * r * sizeof(float_type));
    float_type* H_b = (float_type*) aligned_alloc(64, n * r * sizeof(float_type));

    init_V(V_a, m, n);
    init_V(V_b, m, n);

    init_WH(V_a, W_a, H_a, m, n, r);

    copy_array(W_a, W_b, m, r);
    copy_array(H_a, H_b, r, n);

    //compare_results(W_a, W_b, m, r, eps);
    //compare_results(H_a, H_b, r, n, eps);

    //print_matrix(W_a, m, r, "W_a");
    //print_matrix(W_b, m, r, "W_b");

    //print_matrix(H_a, r, n, "H_a");
    //print_matrix(H_b, r, n, "H_b");

    loop_iterations = 0;
    make_cache_cold(cold_cache_array);
    start = start_tsc();
    nnmf_opt_3(V_a, W_a, H_a, m, n, r, &loop_iterations, 1, threshold, 2);
    stop = stop_tsc(start);
    printf("nnmf_opt_3:\t%lf cycles,\t%ld iterations\n", (double)stop, loop_iterations);

    loop_iterations = 0;
    make_cache_cold(cold_cache_array);
    start = start_tsc();
    nnmf_opt_4(V_b, W_b, H_b, m, n, r, &loop_iterations, 1, threshold, 2);
    stop = stop_tsc(start);
    printf("nnmf_opt_4:\t%lf cycles,\t%ld iterations\n", (double)stop, loop_iterations);

     //print_matrix(V_a, m, n, "V_a");
     //print_matrix(V_b, m, n, "V_b");
     // print_matrix(W_a, m, r, "W_a");
     // print_matrix(W_b, m, r, "W_b");
     // print_matrix(H_a, r, n, "H_a");
     // print_matrix(H_b, r, n, "H_b");

    compare_results(W_a, W_b, m, r, eps);
    compare_results(H_a, H_b, r, n, eps);

    //verify_results(V_a, W_a, H_a, m, n, r, threshold, &norm_a);
    //verify_results(V_b, W_b, H_b, m, n, r, threshold, &norm_b);

    free(V_a);
    free(V_b);

    free(W_a);
    free(W_b);

    free(H_a);
    free(H_b);

    free(cold_cache_array);

    return 0;
}

