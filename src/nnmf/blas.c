#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "cblas.h" 
#include <assert.h>
#include "../init_w.h"

#include <time.h>
#include "../tsc_x86.h"

// TODO Martina: set good values for the following constants

// number of measurements for each configuration
#define NUM_ITERATIONS 3

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
#define NUM_V 2

// threshold value
#define THRESHOLD 10

// L3 cache size
#define CACHE_SIZE 1<<21

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

/**
 * @brief print m x n matrix A
 *
 * in: A    - m x n matrix A
 * in: m, n - matrix dimensions
 * out: C   - m x n matrix C = AB
 */
void print_matrix(float_type* A, int m, int n, char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }
}

/**
 * @brief Generates a random number of type float type
 *
 * in: min, max - range for the random value
 * out:         - a random number of type float_type
 */
float_type rand_from_blas(float_type min, float_type max){
    float_type range = max - min;
    float_type div = RAND_MAX / range;
    return min + (rand() / div);
}

/**
 * @brief Initializes m x n matrix V
 *
 * in: m, n - matrix dimensions
 * out: V   - initialized matrix V (memory already allocated)
 */
void init_V_blas(float_type* V, const int m, const int n) {
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            V[i*n+j] = i+j;
            //V[i*n+j] = rand_from(1,100); // converges much faster with V initialized to i+j
        }
    }
}

/**
 * @brief Matrix Multiplication- blas call
 * 
 * in: A           - m x r matrix A if transpose_A = 0
 *                   r x m matrix A if transpose_A = 1, transpose(A) - m x r matrix
 * in: B           - r x n matrix B if transpose_B = 0
 *                   n x r matrix A if transpose_B = 1, transpose(B) - r x n matrix
 * in: m, n, r     - matrix dimensions
 * in: transpose_A - 0 if A should not be transposed
 *                 - 1 if A should be transposed
 * in: transpose_B - 0 if B should not be transposed
 *                 - 1 if B should be transposed
 * out: C          - m x n matrix C = AB 
 */
void multiply_blas(const float_type* A, const float_type* B, const int m, const int r, const int n, float_type* C, const int transpose_A, const int transpose_B){
    int lda, ldb, ldc;
    if (transpose_A) {
        lda = m;
    }
    else {
        lda = r;
    }
    
    if (transpose_B) {
        ldb = r;
    }
    else {
        ldb = n;
    }

    ldc = n;

    if (!transpose_A && !transpose_B) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, r, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    }
    else if (!transpose_A && transpose_B) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, r, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    }
    else if (transpose_A && !transpose_B) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, m, n, r, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    }
    else {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, m, n, r, 1.0, A, lda, B, ldb, 0.0, C, ldc);
    }

    return;
}

/**
 * @brief Verify if all elements of the the matrices V minus elements of W*H are smaller than threshold
 *
 * in: V       - m x n matrix V
 * in: W       - m x r matrix W
 * in: H       - r x n matrix H
 * in: m, n, r - matrix dimensions
 * out: 1      - verify succeeded
 *      0      - verify didn't succeed
 */
int verify_blas(const float_type* V, float_type* W, float_type* H, const int n, const int m, const int r, const int threshold){
    int ret = 1;
    float_type* WH = malloc(m*n*sizeof(float_type));
    multiply_blas(W,H,m,r,n,WH,0,0);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            float_type x = fabs(WH[i*n+j] - V[i*n+j]);
            if (x >= threshold) {
                //printf("%lf\n", x);
                free(WH);
                return 0;
            }
        }
    }
    free(WH);
    return 1;
}

/**
 * @brief Initializes matrices W and H
 *
 * in: V       - input matrix V
 * in: m, n, r - matrix dimensions
 * out: W, H   - matrices to initialize (W: m x r, H: r x n) (memory already allocated)
 */
void init_WH(const float_type* V, float_type* W, float_type* H, const int m, const int n, const int r) {
    // TODO Martina: find out how to compile the code to use the random_c function
    //init_random_c(W, V, m, n, r, Q_RATIO*n, POOL_SIZE_RATIO*n);
    init_random(W, m, r);
    init_random(H, r, n);
}

/**
 * @brief Non-Negative Matrix Factorization (blas implementation)
 *
 * in: V                - m x n matrix V
 * in: m, n, r          - matrix dimensions
 * in: use_threshold    - 1: stop the iterative update when the error is smaller than threshold
 *                      - 0: run the iterative update for num_iterations
 * in: threshold        - threshold for the error, used if use_threshold = true
 * in: num_iterations   - number of iterations, used if use_threshold = false
 * in/out: W            - m x r matrix W
 * in/out: H            - r x n matrix H
 * out: iteration_count - number of iterations needed until convergence
 */
void nnmf_blas(const float_type* V, float_type* W, float_type* H, const int m, const int n, const int r, long* iteration_count, int use_threshold, const int threshold, const int num_iterations) {
    // TODO Luca
    float_type *WV = malloc(r*n*sizeof(float_type));
    float_type *WW = malloc(r*r*sizeof(float_type));
    float_type *WWH = malloc(r*n*sizeof(float_type));
    float_type *VH = malloc(m*r*sizeof(float_type));
    float_type *WH = malloc(m*n*sizeof(float_type));
    float_type *WHH = malloc(m*r*sizeof(float_type));
    int counter = -1;

    while(use_threshold || counter < num_iterations){
        counter++;
        if (verify_blas(V, W, H, n, m, r, threshold)) {
            if (use_threshold) {
                break;
            }
        }
        int i,j;

        multiply_blas(W,V,r,m,n,WV,1,0);
        multiply_blas(W,W,r,m,r,WW,1,0);
        multiply_blas(WW,H,r,r,n,WWH,0,0);
        for(i = 0; i < r; i++){
            for(j = 0; j < n; j++){
                H[i*n+j] = H[i*n+j]*(WV[i*n+j])/(WWH[i*n+j]);
            }
        }
        multiply_blas(V,H,m,n,r,VH,0,1);
        multiply_blas(W,H,m,r,n,WH,0,0);
        multiply_blas(WH,H,m,n,r,WHH,0,1);
        for(i = 0; i < m; i++){
            for(j = 0; j < r; j++){
                W[i*r+j] = W[i*r+j]*VH[i*r+j]/WHH[i*r+j];
            }
        }
    }
    free(WV);
    free(WW);
    free(WWH);
    free(VH);
    free(WH);
    free(WHH);
    *iteration_count = counter;
    return;
}

void verify_results_blas(const float_type* V, const float_type* W, const float_type* H, const int m, const int n, const int r, const int threshold, float_type* norm) {
     int mr = m * r, rn = r * n;
    *norm = 0.0;
    
    for (int i = 0; i < mr; ++i)
        assert(W[i] >= 0.0);
    for (int i = 0; i < rn; ++i)
        assert(H[i] >= 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float_type sum = 0.;
            for (int k = 0; k < r; ++k) {
                sum += W[i * r + k] * H[k * n + j];
            }
            float_type abs_value = fabs(V[i * n + j] - sum);
            assert(abs_value < threshold);
            *norm += abs_value;
        }
    }
}

int main(int argc, char **argv) {

    srand(11);
    FILE* foutput;

    // TODO: Calibrate?

    // TODO Attila: outsource the work to the make script
    int m, n, r;
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
    float_type* cold_cache_array = (float_type*)malloc(CACHE_SIZE);

    if (argc >= 6) {
        v_id = atoi(argv[5]);
    }

    printf("test_id,v_id,m,n,r,threshold,norm,cpu_cycles,loop iterations,flops,fw_measured_flops\n");
    if (!file_append) {
        fprintf(foutput, "test_id,v_id,m,n,r,threshold,norm,cpu_cycles,loop iterations,flops,fw_measured_flops\n");
    }
    fflush(stdout);

    if (use_arg_mnr) {
        // use different initializations for V
        for (int num_v = 0; num_v < NUM_V; ++num_v, v_id++) {
            float_type* V = (float_type*)malloc(n * m * sizeof(float_type));
            init_V_blas(V, m, n);

            float_type* W = (float_type*)malloc(m * r * sizeof(float_type));
            float_type* H = (float_type*)malloc(n * r * sizeof(float_type));

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


                nnmf_blas(V, W, H, m, n, r, &loop_iterations, 0, threshold, 10);
                
                stop = stop_tsc(start);

                //long m_ = (long) m;
                //long n_ = (long) n;
                //long r_ = (long) r;

                // TODO: flop count wrong! replace flop count with automatic flop count from VTUNE
                //long flops = loop_iterations*(10*m_*n_*r_ + 2*r_*m_*r_ + 2*r_*r_*n_ + m_*n_ + 2*r_*n_ + m_*r_) + 2*m_*n_*r_ + m_*n_*r_*n_*Q_RATIO + m_*n_ + m_*r_ + n_;

                //printf("%d %d %d %ld\n", m, n, r, flops);
                long flops = 0;
                //verify_results_blas(V, W, H, m, n, r, threshold, &norm);
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
    else { // increment m & n proportionally
        for (int m = M_LOW; m <= M_HIGH; m += M_INC) {
            int n = m * M_N_RATIO;
            int r = m * M_N_RATIO; // for now just use squared matrices with m=n=r

            // use different initializations for V
            for (int num_v = 0; num_v < NUM_V; ++num_v, v_id++) {
                float_type* V = (float_type*)malloc(n * m * sizeof(float_type));
                init_V_blas(V, m, n);

                float_type* W = (float_type*)malloc(m * r * sizeof(float_type));
                float_type* H = (float_type*)malloc(n * r * sizeof(float_type));

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

                    nnmf_blas(V, W, H, m, n, r, &loop_iterations, 0, threshold, 10);
                    
                    stop = stop_tsc(start);

                    //long m_ = (long) m;
                    //long n_ = (long) n;
                    //long r_ = (long) r;

                    // TODO: flop count wrong! replace flop count with automatic flop count from VTUNE
                    //long flops = loop_iterations*(10*m_*n_*r_ + 2*r_*m_*r_ + 2*r_*r_*n_ + m_*n_ + 2*r_*n_ + m_*r_) + 2*m_*n_*r_ + m_*n_*r_*n_*Q_RATIO + m_*n_ + m_*r_ + n_;

                    //printf("%d %d %d %ld\n", m, n, r, flops);
                    long flops = 0;
                    //verify_results_blas(V, W, H, m, n, r, threshold, &norm);
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