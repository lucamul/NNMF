#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../nnmf_common.h"
#include "../nnmf.h"

#undef NNMF_FUNCTION
#ifdef VERIFICATION_INLINE
#define NNMF_FUNCTION nnmf_opt_3
#else
#define NNMF_FUNCTION nnmf
#endif

#define BLOCKSIZE 64
#define BLOCK_MATRIX BLOCKSIZE * BLOCKSIZE

// blocking for cache
void NNMF_FUNCTION(const float_type* V, float_type* W, float_type* H, const int m, int n, const int r, long* iteration_count, int use_threshold, const int threshold, const int num_iterations) {
    float_type* VH = malloc(m * r * sizeof(float_type));
    float_type* WH = malloc(m * n * sizeof(float_type));

    int counter = -1;
    float_type sum_WH;
    float_type sum_WV;
    float_type sum_VH;
    float_type sum_WHH;
    float_type sum_WWH;
    float_type w_reuse;
    float_type h_reuse;
    float_type abs_value;
    float_type sum_WHH0, sum_WHH1, sum_WHH2, sum_WHH3;
    int verify;
    int in, ir, jn;
    int i, j, k;
    int ii, jj, kk;
    int iii, jjj, kkk;

    while (use_threshold || counter < num_iterations) {
        counter++;
        verify = 1;

        // Set WH to zero (its faster outside than within the loops)
        for (i = 0; i < m * n; i++) {
            WH[i] = 0.0;
        }

        // Block 1
        for (i = 0; i < m; i+=BLOCKSIZE) {
            for (j = 0; j < n; j+=BLOCKSIZE) {
                for (k = 0; k < r; k+=BLOCKSIZE) {
                    for (ii = i; ii < i+BLOCKSIZE; ii++){
                        for(jj = j; jj < j+BLOCKSIZE; jj++){
                            sum_WH = 0.0;
                            for(kk = k; kk < k+BLOCKSIZE; kk++){
                                sum_WH += W[ii * r + kk] * H[kk * n + jj];
                            }
                            WH[ii*n+jj] += sum_WH;                  
                        }
                    }
                }
                //After k iterations, that block is fully updated
                for (iii = i; iii < i+BLOCKSIZE; iii++)
                {
                    for (jjj = j; jjj < j+BLOCKSIZE; jjj++)
                    {
                        abs_value = fabs(WH[iii*n+jjj] - V[iii * n + jjj]);
                        if (abs_value >= threshold) {
                            verify = 0;
                        } 
                    }                    
                }
            }
        }

        if (verify) {
            if (use_threshold) {
                break;
            }
        }

        // Block 2
        float_type block_sum_WV[BLOCK_MATRIX];
        float_type block_sum_WWH[BLOCK_MATRIX];
        for (i = 0; i < r; i += BLOCKSIZE) {
            for (j = 0; j < n; j += BLOCKSIZE) {
                // Set temp blocks to zero
                for (k = 0; k < BLOCK_MATRIX; k++) {
                    block_sum_WV[k] = 0.0;
                    block_sum_WWH[k] = 0.0;
                }

                // Column by column vector multiplication
                for (k = 0; k < m; k++) {
                    for (ii = 0; ii < BLOCKSIZE; ii++) {
                        for (jj = 0; jj < BLOCKSIZE; jj++) {
                            block_sum_WV[ii * BLOCKSIZE + jj] += W[k * r + i + ii] * V[k * n + j + jj];
                            block_sum_WWH[ii * BLOCKSIZE + jj] += W[k * r + i + ii] * WH[k * n + j + jj];
                        }
                    }
                }

                // Update H
                for (ii = 0; ii < BLOCKSIZE; ii++) {
                    for (jj = 0; jj < BLOCKSIZE; jj++) {
                        H[(i + ii) * n + j + jj] *= block_sum_WV[ii * BLOCKSIZE + jj] / block_sum_WWH[ii * BLOCKSIZE + jj];
                    }
                }
            }
        }

        // Block 3
        // Set WH and VH to zero (its faster outside than within the loops)
        for (i = 0; i < m * n; i++) {
            WH[i] = 0.0;
        }
        for (i = 0; i < m * r; i++) {
            VH[i] = 0.0;
        }
        for (i = 0; i < m; i += BLOCKSIZE) {
            for (k = 0; k < n; k += BLOCKSIZE) {
                for (j = 0; j < r; j += BLOCKSIZE) {
                    for (jj = j; jj < j + BLOCKSIZE; jj++) {
                        for (ii = i; ii < i + BLOCKSIZE; ii++) {
                            for (kk = k; kk < k + BLOCKSIZE; kk++) {
                                VH[ii * r + jj] += V[ii * n + kk] * H[jj * n + kk];
                                WH[ii * n + kk] += W[ii * r + jj] * H[jj * n + kk];
                            }
                        }
                    }
                }
            }
        }

        // Block 4 (Blocking code commented out - loop unrolling is faster)
        // // Reusing block_sum_WWH[BLOCKSIZE * BLOCKSIZE] from part 2
        // for (i = 0; i <= m - BLOCKSIZE; i += BLOCKSIZE) {
        //     for (j = 0; j <= r - BLOCKSIZE; j += BLOCKSIZE) {
        //         // Reset block_sum_WWH
        //         for (int ii = 0; ii < BLOCKSIZE * BLOCKSIZE; ii++) {
        //             block_sum_WWH[ii] = 0.0;
        //         }

        //         // Calculate sums
        //         for (k = 0; k < n; k++) {
        //             for (int ii = 0; ii < BLOCKSIZE; ii++) {
        //                 for (int jj = 0; jj < BLOCKSIZE; jj++) {
        //                     block_sum_WWH[ii * BLOCKSIZE + jj] += WH[(i + ii) * n + k] * H[(j + jj) * n + k];
        //                 }
        //             }
        //         }
        //         for (int ii = 0; ii < BLOCKSIZE; ii++) {
        //             for (int jj = 0; jj < BLOCKSIZE; jj++) {
        //                 W[(i + ii) * r + j + jj] = W[(i + ii) * r + j + jj] * VH[(i + ii) * r + j + jj] / block_sum_WWH[ii * BLOCKSIZE + jj];
        //             }
        //         }
        //     }
        for (i = 0; i < m; i++) {
            in = i*n;
            ir = i*r;
            for (j = 0; j < r; j++) {
                jn = j*n;
                sum_WHH0 = 0.0;
                sum_WHH1 = 0.0;
                sum_WHH2 = 0.0;
                sum_WHH3 = 0.0;
                for (k = 0; k < n-4; k+=4) {
                    sum_WHH0 += WH[in + k] * H[jn + k];
                    sum_WHH1 += WH[in + k+1] * H[jn + k+1];
                    sum_WHH2 += WH[in + k+2] * H[jn + k+2];
                    sum_WHH3 += WH[in + k+3] * H[jn + k+3];
                }
                for (; k < n; k++) {
                    sum_WHH0 += WH[in + k] * H[jn + k];
                }
                W[ir + j] = W[ir + j] * VH[ir + j] / (sum_WHH0 + sum_WHH1 + sum_WHH2 + sum_WHH3);
            }
        }
    }

    free(VH);
    free(WH);

    *iteration_count = counter;
    return;
}