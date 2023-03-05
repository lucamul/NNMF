#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../nnmf_common.h"
#include "../nnmf.h"

#undef NNMF_FUNCTION
#ifdef VERIFICATION_INLINE
#define NNMF_FUNCTION nnmf_opt_5
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

//!!! doesn't work, don't use this function !!!
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
    
    //float_type accumulator
    float_type accu0, accu1, accu2, accu3, accu4, accu5, accu6, accu7, accu8, accu9, accu10, accu11, accu12, accu13, accu14, accu15;

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
                                for (iii = ii; iii < ii+MU; iii++){
                                        accu0 = 0;
                                        accu1 = 0;
                                        accu2 = 0;
                                        accu3 = 0; 
                                        accu4 = 0;
                                        accu5 = 0;
                                        accu6 = 0;
                                        accu7 = 0;
                                        accu8 = 0;
                                        accu9 = 0;
                                        accu10 = 0; 
                                        accu11 = 0;
                                        accu12 = 0;
                                        accu13 = 0;
                                        accu14 = 0;
                                        accu15 = 0;
                                        for (kkk = kk; kkk < kk+KU; kkk++){
                                            accu0 += W[iii*r+kkk]*H[kkk*n + jj];
                                            accu1 += W[iii*r+kkk]*H[kkk*n+1 + jj];
                                            accu2 += W[iii*r+kkk]*H[kkk*n+2 + jj];
                                            accu3 += W[iii*r+kkk]*H[kkk*n+3 + jj];
                                            accu4 += W[iii*r+kkk]*H[kkk*n+4 + jj];
                                            accu5 += W[iii*r+kkk]*H[kkk*n+5 + jj];
                                            accu6 += W[iii*r+kkk]*H[kkk*n+6 + jj];
                                            accu7 += W[iii*r+kkk]*H[kkk*n+7 + jj];
                                            accu8 += W[iii*r+kkk]*H[kkk*n+8 + jj];
                                            accu9 += W[iii*r+kkk]*H[kkk*n+9 + jj];
                                            accu10 += W[iii*r+kkk]*H[kkk*n+10 + jj]; 
                                            accu11 += W[iii*r+kkk]*H[kkk*n+11 + jj];
                                            accu12 += W[iii*r+kkk]*H[kkk*n+12 + jj];
                                            accu13 += W[iii*r+kkk]*H[kkk*n+13 + jj];
                                            accu14 += W[iii*r+kkk]*H[kkk*n+14 + jj];
                                            accu15 += W[iii*r+kkk]*H[kkk*n+15 + jj];
                                        }
                                        WH[iii*n + jj] += accu0;
                                        WH[iii*n+1 + jj] += accu1;
                                        WH[iii*n+2 + jj] += accu2;
                                        WH[iii*n+3 + jj] += accu3;
                                        WH[iii*n+4 + jj] += accu4;
                                        WH[iii*n+5 + jj] += accu5;
                                        WH[iii*n+6 + jj] += accu6;
                                        WH[iii*n+7 + jj] += accu7;
                                        WH[iii*n+8 + jj] += accu8;
                                        WH[iii*n+9 + jj] += accu9;
                                        WH[iii*n+10 + jj] += accu10;
                                        WH[iii*n+11 + jj] += accu11;
                                        WH[iii*n+12 + jj] += accu12;
                                        WH[iii*n+13 + jj] += accu13;
                                        WH[iii*n+14 + jj] += accu14;
                                        WH[iii*n+15 + jj] += accu15;                                   
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
                                w_reuse = W[k * r + iii];
                                for (jjj = jj; jjj < jj+NU; jjj++){
                                    block_sum_WV[temp_i * MU + temp_j] += w_reuse * V[k * n + jjj];
                                    block_sum_WWH[temp_i * MU + temp_j] += w_reuse * WH[k * n + jjj]; 
                                    temp_j++;
                                }
                                temp_i++;                                
                            }
                        }

                        // Update H
                        temp_i=0;
                        for (iii = ii; iii < ii+MU; iii++){
                            temp_j = 0; // I assume that was missing? Otherwise we have a buffer overflow...
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