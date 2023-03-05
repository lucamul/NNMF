#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include "../nnmf_common.h"
#include "../nnmf.h"

#undef NNMF_FUNCTION
#ifdef VERIFICATION_INLINE
#define NNMF_FUNCTION nnmf_opt_7
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

// vectorization with 256 bits
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
    __m256 tmp1, tmp2, tmp3, tmp4, wh1, wh2, wh3, wh4, w, v1, wwh1, wwh2, wwh3, wwh4, wv1, wv2, wv3, wv4, v2, v3, v4, h1, h2, h3, h4, v;
    __m256 whh11, whh12, whh13, whh14, whh21, whh22, whh23, whh24, whh31, whh32, whh33, whh34, whh41, whh42, whh43, whh44, whh112, whh134, whh212, whh234, whh312, whh334, whh412, whh434;
    __m256 vh11, vh12, vh13, vh14, vh21, vh22, vh23, vh24, vh31, vh32, vh33, vh34, vh41, vh42, vh43, vh44, vh112, vh134, vh212, vh234, vh312, vh334, vh412, vh434;
    __m128 wh1_128, wh2_128, wh3_128, wh4_128, w_128, h1_128, h2_128, h3_128, h4_128, h5_128, h6_128, h7_128, h8_128, h9_128, h10_128, h11_128, h12_128, h13_128, h14_128, h15_128, h16_128, vh1_128, vh2_128, vh3_128, vh4_128, v_128, v1h1, v1h2, v1h3, v1h4, v1h5, v1h6, v1h7, v1h8, v1h9, v1h10, v1h11, v1h12, v1h13, v1h14, v1h15, v1h16;
    __m128 whh112_128, whh134_128, whh212_128, whh234_128, whh312_128, whh334_128, whh412_128, whh434_128, whh1, whh2, whh3, whh4;
    __m128 vh112_128, vh134_128, vh212_128, vh234_128, vh312_128, vh334_128, vh412_128, vh434_128, vh1, vh2, vh3, vh4;
    __m128 w1, w2, w3, w4, wvh1, wvh2, wvh3, wvh4, wvhdiv1, wvhdiv2, wvhdiv3, wvhdiv4;
    __m256 zero;

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
        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
        __m256 threshold_vector = _mm256_set1_ps(threshold);
        __m256 verify_vector = _mm256_setzero_ps();
        __m256 abs_value_vector;
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
                                    wh1 = _mm256_load_ps(wh_ptr);
                                    wh3 = _mm256_load_ps(wh_ptr + 8);
                                    int iiir = iii * r;
                                    for (kkk = kk; kkk < kk + KU; kkk++) {
                                        h_ptr = &H[kkk * n + jj];
                                        w = _mm256_set1_ps(W[iiir + kkk]);
                                        h1 = _mm256_load_ps(h_ptr);
                                        h3 = _mm256_load_ps(h_ptr + 8);
                                        wh1 = _mm256_fmadd_ps(w, h1, wh1);
                                        wh3 = _mm256_fmadd_ps(w, h3, wh3);
                                    }
                                    _mm256_store_ps(wh_ptr, wh1);
                                    _mm256_store_ps(wh_ptr + 8, wh3);
                                }
                            }
                        }
                    }
                }
                //After k iterations, that block is fully updated
                for (iiii = i; iiii < i + BLOCKSIZE; iiii++)
                {
                    for (jjjj = j; jjjj < j + BLOCKSIZE; jjjj += 8)
                    {
                        wh1 = _mm256_load_ps(&WH[iiii * n + jjjj]);
                        v = _mm256_load_ps(&V[iiii * n + jjjj]);
                        abs_value_vector = _mm256_cmp_ps(_mm256_sub_ps(wh1, v), threshold_vector, _CMP_GE_OQ);
                        verify_vector = _mm256_or_ps(abs_value_vector, verify_vector);
                    }
                }
            }
        }

        verify = _mm256_testz_si256(_mm256_castps_si256(verify_vector), _mm256_set1_epi32(0xffffffff));
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
                        for (k = 0; k < BLOCK_REGISTERS; k += 8) {
                            wv1 = _mm256_setzero_ps();
                            _mm256_store_ps(wv_ptr + k, wv1);
                            wwh1 = _mm256_setzero_ps();
                            _mm256_store_ps(wwh_ptr + k, wwh1);
                        }

                        for (k = 0; k < m; k++) {
                            temp_i = 0;
                            for (iii = ii; iii < ii + MU; iii++) {
                                wh_ptr = &WH[k * n + jj];
                                v_ptr = &V[k * n + jj];
                                wwh_ptr = &block_sum_WWH[temp_i * MU];
                                wv_ptr = &block_sum_WV[temp_i * MU];
                                w = _mm256_set1_ps(W[k * r + iii]);
                                wh1 = _mm256_load_ps(wh_ptr);
                                wh3 = _mm256_load_ps(wh_ptr + 8);
                                v1 = _mm256_load_ps(v_ptr);
                                v3 = _mm256_load_ps(v_ptr + 8);
                                wwh1 = _mm256_load_ps(wwh_ptr);
                                wwh3 = _mm256_load_ps(wwh_ptr + 8);
                                wv1 = _mm256_load_ps(wv_ptr);
                                wv3 = _mm256_load_ps(wv_ptr + 8);
                                wv1 = _mm256_fmadd_ps(w, v1, wv1);
                                wv3 = _mm256_fmadd_ps(w, v3, wv3);
                                wwh1 = _mm256_fmadd_ps(w, wh1, wwh1);
                                wwh3 = _mm256_fmadd_ps(w, wh3, wwh3);
                                _mm256_store_ps(wv_ptr, wv1);
                                _mm256_store_ps(wv_ptr + 8, wv3);
                                _mm256_store_ps(wwh_ptr, wwh1);
                                _mm256_store_ps(wwh_ptr + 8, wwh3);
                                temp_i++;
                            }
                        }

                        // Update H
                        temp_i = 0;
                        for (iii = ii; iii < ii + MU; iii++) {
                            h_ptr = &H[iii * n + jj];
                            wwh_ptr = &block_sum_WWH[temp_i * MU];
                            wv_ptr = &block_sum_WV[temp_i * MU];
                            h1 = _mm256_load_ps(h_ptr);
                            h3 = _mm256_load_ps(h_ptr + 8);
                            wwh1 = _mm256_load_ps(wwh_ptr);
                            wwh3 = _mm256_load_ps(wwh_ptr + 8);
                            wv1 = _mm256_load_ps(wv_ptr);
                            wv3 = _mm256_load_ps(wv_ptr + 8);
                            tmp1 = _mm256_div_ps(wv1, wwh1);
                            tmp3 = _mm256_div_ps(wv3, wwh3);
                            h1 = _mm256_mul_ps(h1, tmp1);
                            h3 = _mm256_mul_ps(h3, tmp3);
                            _mm256_store_ps(h_ptr, h1);
                            _mm256_store_ps(h_ptr + 8, h3);
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
            for (j = 0; j < n; j += BLOCKSIZE) {
                for (k = 0; k < r; k += BLOCKSIZE) {
                    for (ii = i; ii < i + BLOCKSIZE; ii += MU) {
                        for (jj = j; jj < j + BLOCKSIZE; jj += NU) {
                            for (kk = k; kk < k + BLOCKSIZE; kk += KU) {
                                //with each of the kkk iterations, we would be completing one full sweep of WH micro matrix
                                //after kk+KU iterations we would have completed that Micro MMM
                                for (iii = ii; iii < ii + MU; iii++) {
                                    wh_ptr = &WH[iii * n + jj];
                                    wh1 = _mm256_load_ps(wh_ptr);
                                    wh3 = _mm256_load_ps(wh_ptr + 8);
                                    int iiir = iii * r;
                                    for (kkk = kk; kkk < kk + KU; kkk++) {
                                        h_ptr = &H[kkk * n + jj];
                                        w = _mm256_set1_ps(W[iiir + kkk]);
                                        h1 = _mm256_load_ps(h_ptr);
                                        h3 = _mm256_load_ps(h_ptr + 8);
                                        wh1 = _mm256_fmadd_ps(w, h1, wh1);
                                        wh3 = _mm256_fmadd_ps(w, h3, wh3);
                                    }
                                    _mm256_store_ps(wh_ptr, wh1);
                                    _mm256_store_ps(wh_ptr + 8, wh3);
                                }
                            }
                        }
                    }
                }
            }
        }

        for (i = 0; i < m; i += 4) {
            for (j = 0; j < r; j += 4) {
                vh11 = _mm256_setzero_ps();
                vh12 = _mm256_setzero_ps();
                vh13 = _mm256_setzero_ps();
                vh14 = _mm256_setzero_ps();
                vh21 = _mm256_setzero_ps();
                vh22 = _mm256_setzero_ps();
                vh23 = _mm256_setzero_ps();
                vh24 = _mm256_setzero_ps();
                vh31 = _mm256_setzero_ps();
                vh32 = _mm256_setzero_ps();
                vh33 = _mm256_setzero_ps();
                vh34 = _mm256_setzero_ps();
                vh41 = _mm256_setzero_ps();
                vh42 = _mm256_setzero_ps();
                vh43 = _mm256_setzero_ps();
                vh44 = _mm256_setzero_ps();
                for (k = 0; k < n; k += 8) {
                    // Load from V and H
                    v1 = _mm256_load_ps(&V[i * n + k]);
                    v2 = _mm256_load_ps(&V[(i + 1) * n + k]);
                    v3 = _mm256_load_ps(&V[(i + 2) * n + k]);
                    v4 = _mm256_load_ps(&V[(i + 3) * n + k]);
                    h1 = _mm256_load_ps(&H[j * n + k]);
                    h2 = _mm256_load_ps(&H[(j + 1) * n + k]);
                    h3 = _mm256_load_ps(&H[(j + 2) * n + k]);
                    h4 = _mm256_load_ps(&H[(j + 3) * n + k]);

                    // Add all i and j combinations along k
                    vh11 = _mm256_fmadd_ps(v1, h1, vh11);
                    vh12 = _mm256_fmadd_ps(v1, h2, vh12);
                    vh13 = _mm256_fmadd_ps(v1, h3, vh13);
                    vh14 = _mm256_fmadd_ps(v1, h4, vh14);
                    vh21 = _mm256_fmadd_ps(v2, h1, vh21);
                    vh22 = _mm256_fmadd_ps(v2, h2, vh22);
                    vh23 = _mm256_fmadd_ps(v2, h3, vh23);
                    vh24 = _mm256_fmadd_ps(v2, h4, vh24);
                    vh31 = _mm256_fmadd_ps(v3, h1, vh31);
                    vh32 = _mm256_fmadd_ps(v3, h2, vh32);
                    vh33 = _mm256_fmadd_ps(v3, h3, vh33);
                    vh34 = _mm256_fmadd_ps(v3, h4, vh34);
                    vh41 = _mm256_fmadd_ps(v4, h1, vh41);
                    vh42 = _mm256_fmadd_ps(v4, h2, vh42);
                    vh43 = _mm256_fmadd_ps(v4, h3, vh43);
                    vh44 = _mm256_fmadd_ps(v4, h4, vh44);
                }
                // Add horizontally
                vh112 = _mm256_hadd_ps(vh11, vh12);
                vh134 = _mm256_hadd_ps(vh13, vh14);
                vh212 = _mm256_hadd_ps(vh21, vh22);
                vh234 = _mm256_hadd_ps(vh23, vh24);
                vh312 = _mm256_hadd_ps(vh31, vh32);
                vh334 = _mm256_hadd_ps(vh33, vh34);
                vh412 = _mm256_hadd_ps(vh41, vh42);
                vh434 = _mm256_hadd_ps(vh43, vh44);
                vh112_128 = _mm_add_ps(_mm256_castps256_ps128(vh112), _mm256_extractf128_ps(vh112, 1));
                vh134_128 = _mm_add_ps(_mm256_castps256_ps128(vh134), _mm256_extractf128_ps(vh134, 1));
                vh212_128 = _mm_add_ps(_mm256_castps256_ps128(vh212), _mm256_extractf128_ps(vh212, 1));
                vh234_128 = _mm_add_ps(_mm256_castps256_ps128(vh234), _mm256_extractf128_ps(vh234, 1));
                vh312_128 = _mm_add_ps(_mm256_castps256_ps128(vh312), _mm256_extractf128_ps(vh312, 1));
                vh334_128 = _mm_add_ps(_mm256_castps256_ps128(vh334), _mm256_extractf128_ps(vh334, 1));
                vh412_128 = _mm_add_ps(_mm256_castps256_ps128(vh412), _mm256_extractf128_ps(vh412, 1));
                vh434_128 = _mm_add_ps(_mm256_castps256_ps128(vh434), _mm256_extractf128_ps(vh434, 1));
                vh1 = _mm_hadd_ps(vh112_128, vh134_128);
                vh2 = _mm_hadd_ps(vh212_128, vh234_128);
                vh3 = _mm_hadd_ps(vh312_128, vh334_128);
                vh4 = _mm_hadd_ps(vh412_128, vh434_128);

                // Store result
                _mm_store_ps(&VH[i * r + j], vh1);
                _mm_store_ps(&VH[(i + 1) * r + j], vh2);
                _mm_store_ps(&VH[(i + 2) * r + j], vh3);
                _mm_store_ps(&VH[(i + 3) * r + j], vh4);
            }
        }

        // Block 4
        for (i = 0; i < m; i += 4) {
            for (j = 0; j < r; j += 4) {
                whh11 = _mm256_setzero_ps();
                whh12 = _mm256_setzero_ps();
                whh13 = _mm256_setzero_ps();
                whh14 = _mm256_setzero_ps();
                whh21 = _mm256_setzero_ps();
                whh22 = _mm256_setzero_ps();
                whh23 = _mm256_setzero_ps();
                whh24 = _mm256_setzero_ps();
                whh31 = _mm256_setzero_ps();
                whh32 = _mm256_setzero_ps();
                whh33 = _mm256_setzero_ps();
                whh34 = _mm256_setzero_ps();
                whh41 = _mm256_setzero_ps();
                whh42 = _mm256_setzero_ps();
                whh43 = _mm256_setzero_ps();
                whh44 = _mm256_setzero_ps();
                for (k = 0; k < n; k += 8) {
                    // Load from WH and H
                    wh1 = _mm256_load_ps(&WH[i * n + k]);
                    wh2 = _mm256_load_ps(&WH[(i + 1) * n + k]);
                    wh3 = _mm256_load_ps(&WH[(i + 2) * n + k]);
                    wh4 = _mm256_load_ps(&WH[(i + 3) * n + k]);
                    h1 = _mm256_load_ps(&H[j * n + k]);
                    h2 = _mm256_load_ps(&H[(j + 1) * n + k]);
                    h3 = _mm256_load_ps(&H[(j + 2) * n + k]);
                    h4 = _mm256_load_ps(&H[(j + 3) * n + k]);

                    // Add all i and j combinations along k
                    whh11 = _mm256_fmadd_ps(wh1, h1, whh11);
                    whh12 = _mm256_fmadd_ps(wh1, h2, whh12);
                    whh13 = _mm256_fmadd_ps(wh1, h3, whh13);
                    whh14 = _mm256_fmadd_ps(wh1, h4, whh14);
                    whh21 = _mm256_fmadd_ps(wh2, h1, whh21);
                    whh22 = _mm256_fmadd_ps(wh2, h2, whh22);
                    whh23 = _mm256_fmadd_ps(wh2, h3, whh23);
                    whh24 = _mm256_fmadd_ps(wh2, h4, whh24);
                    whh31 = _mm256_fmadd_ps(wh3, h1, whh31);
                    whh32 = _mm256_fmadd_ps(wh3, h2, whh32);
                    whh33 = _mm256_fmadd_ps(wh3, h3, whh33);
                    whh34 = _mm256_fmadd_ps(wh3, h4, whh34);
                    whh41 = _mm256_fmadd_ps(wh4, h1, whh41);
                    whh42 = _mm256_fmadd_ps(wh4, h2, whh42);
                    whh43 = _mm256_fmadd_ps(wh4, h3, whh43);
                    whh44 = _mm256_fmadd_ps(wh4, h4, whh44);
                }
                // Add horizontally
                whh112 = _mm256_hadd_ps(whh11, whh12);
                whh134 = _mm256_hadd_ps(whh13, whh14);
                whh212 = _mm256_hadd_ps(whh21, whh22);
                whh234 = _mm256_hadd_ps(whh23, whh24);
                whh312 = _mm256_hadd_ps(whh31, whh32);
                whh334 = _mm256_hadd_ps(whh33, whh34);
                whh412 = _mm256_hadd_ps(whh41, whh42);
                whh434 = _mm256_hadd_ps(whh43, whh44);
                whh112_128 = _mm_add_ps(_mm256_castps256_ps128(whh112), _mm256_extractf128_ps(whh112, 1));
                whh134_128 = _mm_add_ps(_mm256_castps256_ps128(whh134), _mm256_extractf128_ps(whh134, 1));
                whh212_128 = _mm_add_ps(_mm256_castps256_ps128(whh212), _mm256_extractf128_ps(whh212, 1));
                whh234_128 = _mm_add_ps(_mm256_castps256_ps128(whh234), _mm256_extractf128_ps(whh234, 1));
                whh312_128 = _mm_add_ps(_mm256_castps256_ps128(whh312), _mm256_extractf128_ps(whh312, 1));
                whh334_128 = _mm_add_ps(_mm256_castps256_ps128(whh334), _mm256_extractf128_ps(whh334, 1));
                whh412_128 = _mm_add_ps(_mm256_castps256_ps128(whh412), _mm256_extractf128_ps(whh412, 1));
                whh434_128 = _mm_add_ps(_mm256_castps256_ps128(whh434), _mm256_extractf128_ps(whh434, 1));
                whh1 = _mm_hadd_ps(whh112_128, whh134_128);
                whh2 = _mm_hadd_ps(whh212_128, whh234_128);
                whh3 = _mm_hadd_ps(whh312_128, whh334_128);
                whh4 = _mm_hadd_ps(whh412_128, whh434_128);

                // Load W and VH
                w1 = _mm_load_ps(&W[i * r + j]);
                w2 = _mm_load_ps(&W[(i + 1) * r + j]);
                w3 = _mm_load_ps(&W[(i + 2) * r + j]);
                w4 = _mm_load_ps(&W[(i + 3) * r + j]);
                vh1_128 = _mm_load_ps(&VH[i * r + j]);
                vh2_128 = _mm_load_ps(&VH[(i + 1) * r + j]);
                vh3_128 = _mm_load_ps(&VH[(i + 2) * r + j]);
                vh4_128 = _mm_load_ps(&VH[(i + 3) * r + j]);

                // Multiply W and VH
                wvh1 = _mm_mul_ps(w1, vh1_128);
                wvh2 = _mm_mul_ps(w2, vh2_128);
                wvh3 = _mm_mul_ps(w3, vh3_128);
                wvh4 = _mm_mul_ps(w4, vh4_128);

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