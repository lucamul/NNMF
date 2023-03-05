#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../nnmf_common.h"
#include "../nnmf.h"

#undef NNMF_FUNCTION
#ifdef VERIFICATION_INLINE
#define NNMF_FUNCTION nnmf_opt_4
#else
#define NNMF_FUNCTION nnmf
#endif

#ifndef BLOCKSIZE
#   define BLOCKSIZE 64
#endif

#ifndef NU
#   define NU 16
#endif
#define MU NU
#define KU NU
#define BLOCK_MATRIX BLOCKSIZE * BLOCKSIZE
#define BLOCK_REGISTERS MU * NU

// blocking for registers
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
    int verify;
    float_type abs_value;
    int i, j, k;
    int ii, jj, kk;
    int iii, jjj, kkk;
    int iiii, jjjj;
    int temp_i, temp_j;

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
                    for (ii = i; ii < i+BLOCKSIZE; ii+=MU){
                        for(jj = j; jj < j+BLOCKSIZE; jj+=NU){
                            for(kk = k; kk < k+BLOCKSIZE; kk+=KU){
                                //with each of the kkk iterations, we would be completing one full sweep of WH micro matrix
                                //after kk+KU iterations we would have completed that Micro MMM
                                for (kkk = kk; kkk < kk+KU; kkk++){
                                    for (iii = ii; iii < ii+MU; iii++){
                                        for (jjj = jj; jjj < jj+NU; jjj++){
                                            WH[iii*n+jjj] += W[iii * r + kkk] * H[kkk * n + jjj];
                                        } 
                                    }                                    
                                }
                            }             
                        }
                    }
                }
                //After k iterations, that block is fully updated
                for (iiii = i; iiii < i+BLOCKSIZE; iiii++)
                {
                    for (jjjj = j; jjjj < j+BLOCKSIZE; jjjj++)
                    {
                        abs_value = fabs(WH[iiii*n+jjjj] - V[iiii * n + jjjj]);
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
        float_type block_sum_WV[BLOCK_REGISTERS];
        float_type block_sum_WWH[BLOCK_REGISTERS];
        for (i = 0; i < r; i += BLOCKSIZE) {
            for (j = 0; j < n; j += BLOCKSIZE) {
                // Column by column vector multiplication
                for (ii = i; ii < i+BLOCKSIZE; ii+=MU) {
                    for (jj = j; jj < j+BLOCKSIZE; jj+=NU) {
                        // Set temp blocks to zero
                        for (k = 0; k < BLOCK_REGISTERS; k++) {
                            block_sum_WV[k] = 0.0;
                            block_sum_WWH[k] = 0.0;
                        }

                        for (k = 0; k < m; k++) {
                            temp_i = 0;
                            for (iii = ii; iii < ii+MU; iii++){
                                temp_j = 0;
                                for (jjj = jj; jjj < jj+NU; jjj++){
                                    block_sum_WV[temp_i * MU + temp_j] += W[k * r + iii] * V[k * n + jjj];
                                    block_sum_WWH[temp_i * MU + temp_j] += W[k * r + iii] * WH[k * n + jjj]; 
                                    temp_j++;
                                }
                                temp_i++;                                
                            }
                        }

                        // Update H
                        temp_i=0;
                        for (iii = ii; iii < ii+MU; iii++){
                            temp_j=0;
                            for (jjj = jj; jjj < jj+NU; jjj++){
                                H[iii * n + jjj] *= block_sum_WV[temp_i * MU + temp_j] / block_sum_WWH[temp_i * MU + temp_j];
                                temp_j++;
                            }   
                            temp_i++;                         
                        }
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
                    for (jj = j; jj < j + BLOCKSIZE; jj+=NU) {
                        for (ii = i; ii < i + BLOCKSIZE; ii+=MU) {
                            for (kk = k; kk < k + BLOCKSIZE; kk+=KU) {
                                //split in two to have better ilp over access patterns
                                //but maybe one is better performant ?
                                for (iii = ii; iii < ii+MU; iii++){
                                    for (jjj = jj; jjj < jj+NU; jjj++){
                                        for (kkk = kk; kkk < kk+KU; kkk++){
                                            WH[iii * n + kkk] += W[iii * r + jjj] * H[jjj * n + kkk];
                                        }                                        
                                    }
                                    for (kkk = kk; kkk < kk+KU; kkk++){
                                        for (jjj = jj; jjj < jj+NU; jjj++){
                                            VH[iii * r + jjj] += V[iii * n + kkk] * H[jjj * n + kkk];
                                        }                                        
                                    }
                                }                                                              
                            }
                        }
                    }
                }
            }
        }

        for (i = 0; i <= m - BLOCKSIZE; i += BLOCKSIZE) {
            for (j = 0; j <= r - BLOCKSIZE; j += BLOCKSIZE) {
                for (ii = i; ii < i+BLOCKSIZE; ii+=MU){
                    for (jj = j; jj < j+BLOCKSIZE; jj+=NU){
                        // Reset block_sum_WWH
                        for (k=0; k < BLOCK_REGISTERS; k++) {
                            block_sum_WWH[k] = 0.0;
                        }

                         // Calculate sums
                        for (k = 0; k < n; k++) {
                            temp_i = 0;
                            for (iii = ii; iii < ii+MU; iii++) {
                                temp_j = 0;
                                for (jjj = jj; jjj < jj+NU; jjj++) {
                                    block_sum_WWH[temp_i * MU + temp_j] += WH[iii * n + k] * H[jjj * n + k];
                                    temp_j++;
                                }
                                temp_i++; 
                            }
                        }

                        //Update W
                        temp_i=0;
                        for (iii = ii; iii < ii+MU; iii++) {
                            temp_j=0;
                            for (jjj = jj; jjj < jj+NU; jjj++) {
                                W[iii * r + jjj] *= VH[iii * r + jjj] / block_sum_WWH[temp_i * MU + temp_j];
                                temp_j++;
                            }
                            temp_i++; 
                        }
                    }
                }
            }
        }
    }

    free(VH);
    free(WH);

    *iteration_count = counter;
    return;
}