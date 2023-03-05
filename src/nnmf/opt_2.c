#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../nnmf_common.h"
#include "../nnmf.h"

#undef NNMF_FUNCTION
#ifdef VERIFICATION_INLINE
#define NNMF_FUNCTION nnmf_opt_2
#else
#define NNMF_FUNCTION nnmf
#endif

//Unroll loops with 4 accumulators to fill pipeline, decent improvement on opt_1 in my (LUCA) machine, not super large (as expected since memory bound)
//Additionally substituting the i*n or j*n etc with a pre-computed in, jn as with unrolling we re-compute that a lot
void NNMF_FUNCTION(const float_type* V, float_type* W, float_type* H, const int m, int n, const int r, long* iteration_count, int use_threshold, const int threshold, const int num_iterations) {
    float_type* VH = malloc(m * r * sizeof(float_type));
    float_type* WH = malloc(m * n * sizeof(float_type));

    int counter = -1;
    float_type sum_WH0,sum_WH1,sum_WH2,sum_WH3;
    float_type sum_WV0,sum_WV1,sum_WV2,sum_WV3;
    float_type sum_VH0,sum_VH1,sum_VH2,sum_VH3;
    float_type sum_WHH0,sum_WHH1,sum_WHH2,sum_WHH3;
    float_type sum_WWH0,sum_WWH1,sum_WWH2,sum_WWH3;
    float_type w_reuse;
    float_type h_reuse,h_reuse0,h_reuse1,h_reuse2,h_reuse3;
    int verify;
    float_type abs_value;
    int i,j,k;
    int in,ir,jn;
    while (use_threshold || counter < num_iterations) {
        counter++;
        verify = 1;

        for (i = 0; i < m; i++) {
            ir = i*r;
            in = i*n;
            for (j = 0; j < n; j++) {
                sum_WH0 = 0.0;
                sum_WH1 = 0.0;
                sum_WH2 = 0.0;
                sum_WH3 = 0.0;
                for (k = 0; k < r-4; k+=4) {
                    sum_WH0 += W[ir + k] * H[k * n + j];
                    sum_WH1 += W[ir + k+1] * H[(k+1) * n + j];
                    sum_WH2 += W[ir + k+2] * H[(k+2) * n + j];
                    sum_WH3 += W[ir + k+3] * H[(k+3) * n + j];
                }
                for(; k < r; k++){
                    sum_WH0 += W[ir + k] * H[k * n + j];
                }
                WH[in + j] = sum_WH0 + sum_WH1 + sum_WH2 + sum_WH3;

                abs_value = fabs(WH[in + j] - V[in + j]);
                if (abs_value >= threshold) {
                    verify = 0;
                }
            }
        }

        if (verify) {
            if (use_threshold) {
                break;
            }
        }

        for (i = 0; i < r; i++) {
            for (j = 0; j < n; j++) {
                sum_WV0 = 0.0;
                sum_WV1 = 0.0;
                sum_WV2 = 0.0;
                sum_WV3 = 0.0;
                sum_WWH0 = 0.0;
                sum_WWH1 = 0.0;
                sum_WWH2 = 0.0;
                sum_WWH3 = 0.0;
                for(k = 0; k < m-4;k+=4){
                    sum_WV0 += W[k * r + i] * V[k * n + j];
                    sum_WWH0 += W[k * r + i] * WH[k * n + j];
                    sum_WV1 += W[(k+1) * r + i] * V[(k+1) * n + j];
                    sum_WWH1 += W[(k+1) * r + i] * WH[(k+1) * n + j];
                    sum_WV2 += W[(k+2) * r + i] * V[(k+2) * n + j];
                    sum_WWH2 += W[(k+2) * r + i] * WH[(k+2) * n + j];
                    sum_WV3 += W[(k+3) * r + i] * V[(k+3) * n + j];
                    sum_WWH3 += W[(k+3) * r + i] * WH[(k+3) * n + j];
                }
                for (; k < m; k++) {
                    sum_WV0 += W[k * r + i] * V[k * n + j];
                    sum_WWH0 += W[k * r + i] * WH[k * n + j];
                }
                H[i * n + j] = H[i * n + j] * (sum_WV0+sum_WV1+sum_WV2+sum_WV3) / (sum_WWH0+sum_WWH1+sum_WWH2+sum_WWH3);
            }
        }

        for (i = 0; i < m; i++) {
            in = i*n;
            for (j = 0; j < r; j++) {
                sum_VH0 = 0.0;
                sum_VH1 = 0.0;
                sum_VH2 = 0.0;
                sum_VH3 = 0.0;
                w_reuse = W[i * r + j];
                jn = j*n;
                for (k = 0; k < n-4; k+=4) {
                    h_reuse0 = H[jn + k];
                    h_reuse1 = H[jn + k+1];
                    h_reuse2 = H[jn + k+2];
                    h_reuse3 = H[jn + k+3];
                    sum_VH0 += V[in + k] * h_reuse0;
                    sum_VH1 += V[in + k+1] * h_reuse1;
                    sum_VH2 += V[in + k+2] * h_reuse2;
                    sum_VH3 += V[in + k+3] * h_reuse3;
                    if (j == 0) {
                        WH[in + k] = 0.0;
                        WH[in + k+1] = 0.0;
                        WH[in + k+2] = 0.0;
                        WH[in + k+3] = 0.0;
                    }

                    WH[in + k] += w_reuse * h_reuse0;
                    WH[in + k+1] += w_reuse * h_reuse1;
                    WH[in + k+2] += w_reuse * h_reuse2;
                    WH[in + k+3] += w_reuse * h_reuse3;
                }
                for (; k < n; k++) {
                    h_reuse = H[jn + k];
                    sum_VH0 += V[in + k] * h_reuse;

                    if (j == 0) {
                        WH[in + k] = 0;
                    }

                    WH[in + k] += w_reuse * h_reuse;
                }

                VH[i * r + j] = sum_VH0 + sum_VH1 + sum_VH2 + sum_VH3;
            }
        }

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