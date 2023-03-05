#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include "../nnmf_common.h"
#include "../nnmf.h"

#undef NNMF_FUNCTION
#ifdef VERIFICATION_INLINE
#define NNMF_FUNCTION nnmf_opt_6
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

// vectorization with 128 bits
void NNMF_FUNCTION(const float_type* V, float_type* W, float_type* H, const int m, int n, const int r, long* iteration_count, int use_threshold, const int threshold, const int num_iterations) {
    float_type* VH = (float_type*)aligned_alloc(64, m * r * sizeof(float_type));
    float_type* WH = (float_type*)aligned_alloc(64, m * n * sizeof(float_type));

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
    int mask1 = 241; // binary: 11110001
    int mask2 = 242; // binary: 11110010
    int mask3 = 244; // binary: 11110100
    int mask4 = 248; // binary: 11111000
    __m128 tmp1, tmp2, tmp3, tmp4, wh1, wh2, wh3, wh4, w, v1, wwh1, wwh2, wwh3, wwh4, wv1, wv2, wv3, wv4, v2, v3, v4, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, vh1, vh2, vh3, vh4, v, v1h1, v1h2, v1h3, v1h4, v1h5, v1h6, v1h7, v1h8, v1h9, v1h10, v1h11, v1h12, v1h13, v1h14, v1h15, v1h16;
    __m128 whh11, whh12, whh13, whh14, whh21, whh22, whh23, whh24, whh31, whh32, whh33, whh34, whh41, whh42, whh43, whh44, whh112, whh134, whh212, whh234, whh312, whh334, whh412, whh434, whh1, whh2, whh3, whh4;
    __m128 vh11, vh12, vh13, vh14, vh21, vh22, vh23, vh24, vh31, vh32, vh33, vh34, vh41, vh42, vh43, vh44, vh112, vh134, vh212, vh234, vh312, vh334, vh412, vh434;
    __m128 w1, w2, w3, w4, wvh1, wvh2, wvh3, wvh4, wvhdiv1, wvhdiv2, wvhdiv3, wvhdiv4;
    __m128 zero;

    float_type* h_ptr, * wh_ptr, * wwh_ptr, * wv_ptr, * vh_ptr;
    const float_type* v_ptr;

    while (use_threshold || counter < num_iterations) {
        counter++;
        verify = 1;

        // Set WH to zero (its faster outside than within the loops)
        for (i = 0; i < m * n; i++) {
            WH[i] = 0.0;
        }

        // Block 1
        __m128 abs_mask = _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff));
        __m128 threshold_vector = _mm_set1_ps(threshold);
        __m128 verify_vector = _mm_setzero_ps();
        __m128 abs_value_vector;
        for (i = 0; i < m; i += BLOCKSIZE) {
            for (j = 0; j < n; j += BLOCKSIZE) {
                for (k = 0; k < r; k += BLOCKSIZE) {
                    for (ii = i; ii < i + BLOCKSIZE; ii += MU) {
                        for (jj = j; jj < j + BLOCKSIZE; jj += NU) {
                            for (kk = k; kk < k + BLOCKSIZE; kk += KU) {
                                //with each of the kkk iterations, we would be completing one full sweep of WH micro matrix
                                //after kk+KU iterations we would have completed that Micro MMM
                                for (iii = ii; iii < ii + MU; iii++) {
                                    wh_ptr = &WH[iii * n + jj];
                                    wh1 = _mm_load_ps(wh_ptr);
                                    wh2 = _mm_load_ps(wh_ptr + 4);
                                    wh3 = _mm_load_ps(wh_ptr + 8);
                                    wh4 = _mm_load_ps(wh_ptr + 12);
                                    int iiir = iii * r;
                                    for (kkk = kk; kkk < kk + KU; kkk++) {
                                        h_ptr = &H[kkk * n + jj];
                                        w = _mm_set1_ps(W[iiir + kkk]);
                                        h1 = _mm_load_ps(h_ptr);
                                        h2 = _mm_load_ps(h_ptr + 4);
                                        h3 = _mm_load_ps(h_ptr + 8);
                                        h4 = _mm_load_ps(h_ptr + 12);
                                        wh1 = _mm_fmadd_ps(w, h1, wh1);
                                        wh2 = _mm_fmadd_ps(w, h2, wh2);
                                        wh3 = _mm_fmadd_ps(w, h3, wh3);
                                        wh4 = _mm_fmadd_ps(w, h4, wh4);
                                    }
                                    _mm_store_ps(wh_ptr, wh1);
                                    _mm_store_ps(wh_ptr + 4, wh2);
                                    _mm_store_ps(wh_ptr + 8, wh3);
                                    _mm_store_ps(wh_ptr + 12, wh4);
                                }
                            }
                        }
                    }
                }
                //After k iterations, that block is fully updated
                for (iiii = i; iiii < i + BLOCKSIZE; iiii++)
                {
                    for (jjjj = j; jjjj < j + BLOCKSIZE; jjjj += 4)
                    {
                        wh1 = _mm_load_ps(&WH[iiii * n + jjjj]);
                        v = _mm_load_ps(&V[iiii * n + jjjj]);
                        abs_value_vector = _mm_cmp_ps(_mm_sub_ps(wh1, v), threshold_vector, _CMP_GE_OQ);
                        verify_vector = _mm_or_ps(abs_value_vector, verify_vector);
                    }
                }
            }
        }

        verify = _mm_test_all_zeros(_mm_castps_si128(verify_vector), _mm_set1_epi32(0xffffffff));
        if (verify) {
            if (use_threshold) {
                break;
            }
        }

        // Block 2
        float_type block_sum_WV[BLOCK_REGISTERS] __attribute__((aligned(64)));
        float_type block_sum_WWH[BLOCK_REGISTERS] __attribute__((aligned(64)));
        for (i = 0; i < r; i += BLOCKSIZE) {
            for (j = 0; j < n; j += BLOCKSIZE) {
                // Column by column vector multiplication
                for (ii = i; ii < i + BLOCKSIZE; ii += MU) {
                    for (jj = j; jj < j + BLOCKSIZE; jj += NU) {
                        // Set temp blocks to zero
                        wv_ptr = &block_sum_WV[0];
                        wwh_ptr = &block_sum_WWH[0];
                        zero = _mm_setzero_ps();
                        for (k = 0; k < BLOCK_REGISTERS; k += 4) {
                            // wv1 = _mm_setzero_ps();
                            // _mm_store_ps(wv_ptr + k, wv1);
                            // wwh1 = _mm_setzero_ps();
                            // _mm_store_ps(wwh_ptr + k, wwh1);
                            _mm_store_ps(wv_ptr + k, zero);
                            _mm_store_ps(wwh_ptr + k, zero);
                        }

                        for (k = 0; k < m; k++) {
                            temp_i = 0;
                            for (iii = ii; iii < ii + MU; iii++) {
                                wh_ptr = &WH[k * n + jj];
                                v_ptr = &V[k * n + jj];
                                wwh_ptr = &block_sum_WWH[temp_i * MU];
                                wv_ptr = &block_sum_WV[temp_i * MU];
                                w = _mm_set1_ps(W[k * r + iii]);
                                wh1 = _mm_load_ps(wh_ptr);
                                wh2 = _mm_load_ps(wh_ptr + 4);
                                wh3 = _mm_load_ps(wh_ptr + 8);
                                wh4 = _mm_load_ps(wh_ptr + 12);
                                v1 = _mm_load_ps(v_ptr);
                                v2 = _mm_load_ps(v_ptr + 4);
                                v3 = _mm_load_ps(v_ptr + 8);
                                v4 = _mm_load_ps(v_ptr + 12);
                                wwh1 = _mm_load_ps(wwh_ptr);
                                wwh2 = _mm_load_ps(wwh_ptr + 4);
                                wwh3 = _mm_load_ps(wwh_ptr + 8);
                                wwh4 = _mm_load_ps(wwh_ptr + 12);
                                wv1 = _mm_load_ps(wv_ptr);
                                wv2 = _mm_load_ps(wv_ptr + 4);
                                wv3 = _mm_load_ps(wv_ptr + 8);
                                wv4 = _mm_load_ps(wv_ptr + 12);
                                wv1 = _mm_fmadd_ps(w, v1, wv1);
                                wv2 = _mm_fmadd_ps(w, v2, wv2);
                                wv3 = _mm_fmadd_ps(w, v3, wv3);
                                wv4 = _mm_fmadd_ps(w, v4, wv4);
                                wwh1 = _mm_fmadd_ps(w, wh1, wwh1);
                                wwh2 = _mm_fmadd_ps(w, wh2, wwh2);
                                wwh3 = _mm_fmadd_ps(w, wh3, wwh3);
                                wwh4 = _mm_fmadd_ps(w, wh4, wwh4);
                                _mm_store_ps(wv_ptr, wv1);
                                _mm_store_ps(wv_ptr + 4, wv2);
                                _mm_store_ps(wv_ptr + 8, wv3);
                                _mm_store_ps(wv_ptr + 12, wv4);
                                _mm_store_ps(wwh_ptr, wwh1);
                                _mm_store_ps(wwh_ptr + 4, wwh2);
                                _mm_store_ps(wwh_ptr + 8, wwh3);
                                _mm_store_ps(wwh_ptr + 12, wwh4);
                                temp_i++;
                            }
                        }

                        // Update H
                        temp_i = 0;
                        for (iii = ii; iii < ii + MU; iii++) {
                            h_ptr = &H[iii * n + jj];
                            wwh_ptr = &block_sum_WWH[temp_i * MU];
                            wv_ptr = &block_sum_WV[temp_i * MU];
                            h1 = _mm_load_ps(h_ptr);
                            h2 = _mm_load_ps(h_ptr + 4);
                            h3 = _mm_load_ps(h_ptr + 8);
                            h4 = _mm_load_ps(h_ptr + 12);
                            wwh1 = _mm_load_ps(wwh_ptr);
                            wwh2 = _mm_load_ps(wwh_ptr + 4);
                            wwh3 = _mm_load_ps(wwh_ptr + 8);
                            wwh4 = _mm_load_ps(wwh_ptr + 12);
                            wv1 = _mm_load_ps(wv_ptr);
                            wv2 = _mm_load_ps(wv_ptr + 4);
                            wv3 = _mm_load_ps(wv_ptr + 8);
                            wv4 = _mm_load_ps(wv_ptr + 12);
                            tmp1 = _mm_div_ps(wv1, wwh1);
                            tmp2 = _mm_div_ps(wv2, wwh2);
                            tmp3 = _mm_div_ps(wv3, wwh3);
                            tmp4 = _mm_div_ps(wv4, wwh4);
                            h1 = _mm_mul_ps(h1, tmp1);
                            h2 = _mm_mul_ps(h2, tmp2);
                            h3 = _mm_mul_ps(h3, tmp3);
                            h4 = _mm_mul_ps(h4, tmp4);
                            _mm_store_ps(h_ptr, h1);
                            _mm_store_ps(h_ptr + 4, h2);
                            _mm_store_ps(h_ptr + 8, h3);
                            _mm_store_ps(h_ptr + 12, h4);
                            temp_i++;
                        }
                    }
                }
            }
        }

        // Block 3
        // Set WH and VH to zero (its faster outside than within the loops)
        zero = _mm_setzero_ps();
        for (i = 0; i < m * n; i += 4) {
            _mm_store_ps(&WH[i], zero);
        }
        for (i = 0; i < m * r; i += 4) {
            _mm_store_ps(&VH[i], zero);
        }

        for (i = 0; i < m; i += BLOCKSIZE) {
            for (j = 0; j < n; j += BLOCKSIZE) {
                for (k = 0; k < r; k += BLOCKSIZE) {
                    for (ii = i; ii < i + BLOCKSIZE; ii += MU) {
                        for (jj = j; jj < j + BLOCKSIZE; jj += NU) {
                            for (kk = k; kk < k + BLOCKSIZE; kk += KU) {
                                //with each of the kkk iterations, we would be completing one full sweep of WH micro matrix
                                //after kk+KU iterations we would have completed that Micro MMM
                                for (iii = ii; iii < ii + MU; iii++) {
                                    wh_ptr = &WH[iii * n + jj];
                                    wh1 = _mm_load_ps(wh_ptr);
                                    wh2 = _mm_load_ps(wh_ptr + 4);
                                    wh3 = _mm_load_ps(wh_ptr + 8);
                                    wh4 = _mm_load_ps(wh_ptr + 12);
                                    int iiir = iii * r;
                                    for (kkk = kk; kkk < kk + KU; kkk++) {
                                        h_ptr = &H[kkk * n + jj];
                                        w = _mm_set1_ps(W[iiir + kkk]);
                                        h1 = _mm_load_ps(h_ptr);
                                        h2 = _mm_load_ps(h_ptr + 4);
                                        h3 = _mm_load_ps(h_ptr + 8);
                                        h4 = _mm_load_ps(h_ptr + 12);
                                        wh1 = _mm_fmadd_ps(w, h1, wh1);
                                        wh2 = _mm_fmadd_ps(w, h2, wh2);
                                        wh3 = _mm_fmadd_ps(w, h3, wh3);
                                        wh4 = _mm_fmadd_ps(w, h4, wh4);
                                    }
                                    _mm_store_ps(wh_ptr, wh1);
                                    _mm_store_ps(wh_ptr + 4, wh2);
                                    _mm_store_ps(wh_ptr + 8, wh3);
                                    _mm_store_ps(wh_ptr + 12, wh4);
                                }
                            }
                        }
                    }
                }
            }
        }

        for (i = 0; i < m; i += 4) {
            for (j = 0; j < r; j += 4) {
                vh11 = _mm_setzero_ps();
                vh12 = _mm_setzero_ps();
                vh13 = _mm_setzero_ps();
                vh14 = _mm_setzero_ps();
                vh21 = _mm_setzero_ps();
                vh22 = _mm_setzero_ps();
                vh23 = _mm_setzero_ps();
                vh24 = _mm_setzero_ps();
                vh31 = _mm_setzero_ps();
                vh32 = _mm_setzero_ps();
                vh33 = _mm_setzero_ps();
                vh34 = _mm_setzero_ps();
                vh41 = _mm_setzero_ps();
                vh42 = _mm_setzero_ps();
                vh43 = _mm_setzero_ps();
                vh44 = _mm_setzero_ps();
                for (k = 0; k < n; k += 4) {
                    // Load from V and H
                    v1 = _mm_load_ps(&V[i * n + k]);
                    v2 = _mm_load_ps(&V[(i + 1) * n + k]);
                    v3 = _mm_load_ps(&V[(i + 2) * n + k]);
                    v4 = _mm_load_ps(&V[(i + 3) * n + k]);
                    h1 = _mm_load_ps(&H[j * n + k]);
                    h2 = _mm_load_ps(&H[(j + 1) * n + k]);
                    h3 = _mm_load_ps(&H[(j + 2) * n + k]);
                    h4 = _mm_load_ps(&H[(j + 3) * n + k]);

                    // Add all i and j combinations along k
                    vh11 = _mm_fmadd_ps(v1, h1, vh11);
                    vh12 = _mm_fmadd_ps(v1, h2, vh12);
                    vh13 = _mm_fmadd_ps(v1, h3, vh13);
                    vh14 = _mm_fmadd_ps(v1, h4, vh14);
                    vh21 = _mm_fmadd_ps(v2, h1, vh21);
                    vh22 = _mm_fmadd_ps(v2, h2, vh22);
                    vh23 = _mm_fmadd_ps(v2, h3, vh23);
                    vh24 = _mm_fmadd_ps(v2, h4, vh24);
                    vh31 = _mm_fmadd_ps(v3, h1, vh31);
                    vh32 = _mm_fmadd_ps(v3, h2, vh32);
                    vh33 = _mm_fmadd_ps(v3, h3, vh33);
                    vh34 = _mm_fmadd_ps(v3, h4, vh34);
                    vh41 = _mm_fmadd_ps(v4, h1, vh41);
                    vh42 = _mm_fmadd_ps(v4, h2, vh42);
                    vh43 = _mm_fmadd_ps(v4, h3, vh43);
                    vh44 = _mm_fmadd_ps(v4, h4, vh44);
                }
                // Add horizontally
                vh112 = _mm_hadd_ps(vh11, vh12);
                vh134 = _mm_hadd_ps(vh13, vh14);
                vh212 = _mm_hadd_ps(vh21, vh22);
                vh234 = _mm_hadd_ps(vh23, vh24);
                vh312 = _mm_hadd_ps(vh31, vh32);
                vh334 = _mm_hadd_ps(vh33, vh34);
                vh412 = _mm_hadd_ps(vh41, vh42);
                vh434 = _mm_hadd_ps(vh43, vh44);
                vh1 = _mm_hadd_ps(vh112, vh134);
                vh2 = _mm_hadd_ps(vh212, vh234);
                vh3 = _mm_hadd_ps(vh312, vh334);
                vh4 = _mm_hadd_ps(vh412, vh434);

                // Store result
                _mm_store_ps(&VH[i * r + j], vh1);
                _mm_store_ps(&VH[(i + 1) * r + j], vh2);
                _mm_store_ps(&VH[(i + 2) * r + j], vh3);
                _mm_store_ps(&VH[(i + 3) * r + j], vh4);
            }
        }

        for (i = 0; i < m; i += 4) {
            for (j = 0; j < r; j += 4) {
                whh11 = _mm_setzero_ps();
                whh12 = _mm_setzero_ps();
                whh13 = _mm_setzero_ps();
                whh14 = _mm_setzero_ps();
                whh21 = _mm_setzero_ps();
                whh22 = _mm_setzero_ps();
                whh23 = _mm_setzero_ps();
                whh24 = _mm_setzero_ps();
                whh31 = _mm_setzero_ps();
                whh32 = _mm_setzero_ps();
                whh33 = _mm_setzero_ps();
                whh34 = _mm_setzero_ps();
                whh41 = _mm_setzero_ps();
                whh42 = _mm_setzero_ps();
                whh43 = _mm_setzero_ps();
                whh44 = _mm_setzero_ps();
                for (k = 0; k < n; k += 4) {
                    // Load from WH and H
                    wh1 = _mm_load_ps(&WH[i * n + k]);
                    wh2 = _mm_load_ps(&WH[(i + 1) * n + k]);
                    wh3 = _mm_load_ps(&WH[(i + 2) * n + k]);
                    wh4 = _mm_load_ps(&WH[(i + 3) * n + k]);
                    h1 = _mm_load_ps(&H[j * n + k]);
                    h2 = _mm_load_ps(&H[(j + 1) * n + k]);
                    h3 = _mm_load_ps(&H[(j + 2) * n + k]);
                    h4 = _mm_load_ps(&H[(j + 3) * n + k]);

                    // Add all i and j combinations along k
                    whh11 = _mm_fmadd_ps(wh1, h1, whh11);
                    whh12 = _mm_fmadd_ps(wh1, h2, whh12);
                    whh13 = _mm_fmadd_ps(wh1, h3, whh13);
                    whh14 = _mm_fmadd_ps(wh1, h4, whh14);
                    whh21 = _mm_fmadd_ps(wh2, h1, whh21);
                    whh22 = _mm_fmadd_ps(wh2, h2, whh22);
                    whh23 = _mm_fmadd_ps(wh2, h3, whh23);
                    whh24 = _mm_fmadd_ps(wh2, h4, whh24);
                    whh31 = _mm_fmadd_ps(wh3, h1, whh31);
                    whh32 = _mm_fmadd_ps(wh3, h2, whh32);
                    whh33 = _mm_fmadd_ps(wh3, h3, whh33);
                    whh34 = _mm_fmadd_ps(wh3, h4, whh34);
                    whh41 = _mm_fmadd_ps(wh4, h1, whh41);
                    whh42 = _mm_fmadd_ps(wh4, h2, whh42);
                    whh43 = _mm_fmadd_ps(wh4, h3, whh43);
                    whh44 = _mm_fmadd_ps(wh4, h4, whh44);
                }
                // Add horizontally
                whh112 = _mm_hadd_ps(whh11, whh12);
                whh134 = _mm_hadd_ps(whh13, whh14);
                whh212 = _mm_hadd_ps(whh21, whh22);
                whh234 = _mm_hadd_ps(whh23, whh24);
                whh312 = _mm_hadd_ps(whh31, whh32);
                whh334 = _mm_hadd_ps(whh33, whh34);
                whh412 = _mm_hadd_ps(whh41, whh42);
                whh434 = _mm_hadd_ps(whh43, whh44);
                whh1 = _mm_hadd_ps(whh112, whh134);
                whh2 = _mm_hadd_ps(whh212, whh234);
                whh3 = _mm_hadd_ps(whh312, whh334);
                whh4 = _mm_hadd_ps(whh412, whh434);

                // Load W and VH
                w1 = _mm_load_ps(&W[i * r + j]);
                w2 = _mm_load_ps(&W[(i + 1) * r + j]);
                w3 = _mm_load_ps(&W[(i + 2) * r + j]);
                w4 = _mm_load_ps(&W[(i + 3) * r + j]);
                vh1 = _mm_load_ps(&VH[i * r + j]);
                vh2 = _mm_load_ps(&VH[(i + 1) * r + j]);
                vh3 = _mm_load_ps(&VH[(i + 2) * r + j]);
                vh4 = _mm_load_ps(&VH[(i + 3) * r + j]);

                // Multiply W and VH
                wvh1 = _mm_mul_ps(w1, vh1);
                wvh2 = _mm_mul_ps(w2, vh2);
                wvh3 = _mm_mul_ps(w3, vh3);
                wvh4 = _mm_mul_ps(w4, vh4);

                // Divide by wwh
                wvhdiv1 = _mm_div_ps(wvh1, whh1);
                wvhdiv2 = _mm_div_ps(wvh2, whh2);
                wvhdiv3 = _mm_div_ps(wvh3, whh3);
                wvhdiv4 = _mm_div_ps(wvh4, whh4);

                // Store result
                _mm_store_ps(&W[i * r + j], wvhdiv1);
                _mm_store_ps(&W[(i + 1) * r + j], wvhdiv2);
                _mm_store_ps(&W[(i + 2) * r + j], wvhdiv3);
                _mm_store_ps(&W[(i + 3) * r + j], wvhdiv4);
            }
        }
    }

    free(VH);
    free(WH);

    *iteration_count = counter;
    return;
}