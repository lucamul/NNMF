#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "tsc_x86.h"
#include "nnmf.h"
#include "nnmf_common.h"
#include "init_w.h"
#include "utils.h"

// number of measurements for each configuration
#ifndef NUM_ITERATIONS
#   define NUM_ITERATIONS 3
#endif

// ratio m/n of the matrix sizes
// M_N_RATIO = 1 for square matrices
// M_N_RATIO > 1 for non-square matrices, e.g. M_N_RATIO = 1.5 would mean that e.g. m = 3, n = 2
#define M_N_RATIO 1

// matrix size m: it goes from M_LOW, M_LOW + M_INC, M_LOW + 2 * M_INC, ..., M_HIGH
// you need to M_LOW and M_HIGH such that M_LOW * M_N_RATIO and M_HIGH * M_N_RATIO are integers
#define M_LOW 100
#define M_HIGH 900
#define M_INC 100

// number of different matrices V, e.g. V = 2 means that the measurements are made for 2 different matrices V
#ifndef NUM_V
#   define NUM_V 2
#endif

// threshold value
#ifndef THESHOLD
#   define THRESHOLD 10
#endif

// L3 cache size
#ifndef CACHE_SIZE
#   define CACHE_SIZE 1<<21
#endif

/**
 * @brief warms up the CPU by performing some operations
 */
int warm_up_cpu() {
    int sum;
    for (int i = 0; i < 1000000000; ++i) {
        sum += i;
    }
    return sum;
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

int main(int argc, char **argv) {

    srand(11);
    FILE *foutput;

    // TODO: Calibrate?

    // TODO Attila: outsource the work to the make script
    int m, n, r;
    int num_iterations = 10;
    int file_append = 0, use_arg_mnr = 0;
    if (argc >= 5) {
        use_arg_mnr = 1;
        m = atoi(argv[2]);
        n = atoi(argv[3]);
        r = atoi(argv[4]);
        if (foutput = fopen(argv[1], "r")) {
            fclose(foutput);
            file_append = 1;
        }
        foutput = fopen(argv[1], "a");
    }
    else if (argc >= 2)
        foutput = fopen(argv[1], "w");
    else
        foutput = fopen("../../output/result_basic.csv", "w");
    if (!foutput) {
        exit(1);
    }

    warm_up_cpu();

    int v_id = 0;
    float_type* cold_cache_array = (float_type*) malloc(CACHE_SIZE);

    if (argc >= 6) {
        v_id = atoi(argv[5]);
    }
    if (argc >= 7) {
        num_iterations = atoi(argv[6]);
    }

    printf("test_id,v_id,m,n,r,threshold,norm,cpu_cycles,loop iterations,flops,fw_measured_flops\n");
    if (!file_append) {
        fprintf(foutput, "test_id,v_id,m,n,r,threshold,norm,cpu_cycles,loop iterations,flops,fw_measured_flops\n");
    }
    fflush(stdout);

    if (use_arg_mnr) {
        // use different initializations for V
        for (int num_v = 0; num_v < NUM_V; ++num_v, v_id++) {
            float_type* V = (float_type*) aligned_alloc(64, n * m * sizeof(float_type));

            init_V(V, m, n);

            float_type* W = (float_type*) aligned_alloc(64, m * r * sizeof(float_type));
            float_type* H = (float_type*) aligned_alloc(64, n * r * sizeof(float_type));

            float_type threshold = THRESHOLD;
            int test_id = 0;
            float_type norm;
            myInt64 start, stop;
            long loop_iterations;

            for (int j = 0; j < NUM_ITERATIONS; ++j) {
                make_cache_cold(cold_cache_array);
                loop_iterations = 0;
                init_WH(V, W, H, m, n, r);

                start = start_tsc();

                nnmf(V, W, H, m, n, r, &loop_iterations, 0, threshold, num_iterations);

                stop = stop_tsc(start); 

                //long m_ = (long) m;
                //long n_ = (long) n;
                //long r_ = (long) r;

                // TODO: flop count wrong! replace flop count with automatic flop count from VTUNE
                //long flops = loop_iterations*(10*m_*n_*r_ + 2*r_*m_*r_ + 2*r_*r_*n_ + m_*n_ + 2*r_*n_ + m_*r_) + 2*m_*n_*r_ + m_*n_*r_*n_*Q_RATIO + m_*n_ + m_*r_ + n_;
                long flops = 0;
                //verify_results(V, W, H, m, n, r, threshold, &norm);
                printf("%d, %d, %d, %d, %d, %lf, %lf, %lf, %ld, %ld\n", test_id, v_id, m, n, r, threshold, norm, (double)stop, loop_iterations, flops);
                fprintf(foutput, "%d,%d,%d,%d,%d,%lf,%lf,%lf,%ld,%ld\n", test_id, v_id, m, n, r, threshold, norm, (double)stop, loop_iterations, flops);
                fflush(stdout);
                test_id++;
            }

            free(W);
            free(H);
            free(V);
        }
    }
    else {// increment m & n proportionally
        for (int m = M_LOW; m <= M_HIGH; m += M_INC) {
            int n = m * M_N_RATIO;
            int r = m * M_N_RATIO; // for now just use squared matrices with m=n=r

            // use different initializations for V
            for (int num_v = 0; num_v < NUM_V; ++num_v, v_id++) {
                float_type* V = (float_type*)aligned_alloc(64, n * m * sizeof(float_type));

                init_V(V, m, n);

                float_type* W = (float_type*)aligned_alloc(64, m * r * sizeof(float_type));
                float_type* H = (float_type*)aligned_alloc(64, n * r * sizeof(float_type));

                float_type threshold = THRESHOLD;
                int test_id = 0;
                float_type norm;
                myInt64 start, stop;
                long loop_iterations;

                for (int j = 0; j < NUM_ITERATIONS; ++j) {
                    make_cache_cold(cold_cache_array);
                    loop_iterations = 0;
                    init_WH(V, W, H, m, n, r);

                    start = start_tsc();

                    nnmf(V, W, H, m, n, r, &loop_iterations, 0, threshold, 10);

                    stop = stop_tsc(start); 

                    //long m_ = (long) m;
                    //long n_ = (long) n;
                    //long r_ = (long) r;

                    // TODO: flop count wrong! replace flop count with automatic flop count from VTUNE
                    //long flops = loop_iterations*(10*m_*n_*r_ + 2*r_*m_*r_ + 2*r_*r_*n_ + m_*n_ + 2*r_*n_ + m_*r_) + 2*m_*n_*r_ + m_*n_*r_*n_*Q_RATIO + m_*n_ + m_*r_ + n_;
                    long flops = 0;
                    //verify_results(V, W, H, m, n, r, threshold, &norm);
                    printf("%d, %d, %d, %d, %d, %lf, %lf, %lf, %ld, %ld\n", test_id, v_id, m, n, r, threshold, norm, (double)stop, loop_iterations, flops);
                    fprintf(foutput, "%d,%d,%d,%d,%d,%lf,%lf,%lf,%ld,%ld\n", test_id, v_id, m, n, r, threshold, norm, (double)stop, loop_iterations, flops);
                    fflush(stdout);
                    test_id++;
                }

                free(W);
                free(H);
                free(V);
            }
        }
    }

    free(cold_cache_array);
    fclose(foutput);

    return 0;
}

