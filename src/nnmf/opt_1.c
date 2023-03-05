#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "../nnmf_common.h"
#include "../nnmf.h"

#undef NNMF_FUNCTION
#ifdef VERIFICATION_INLINE
#define NNMF_FUNCTION nnmf_opt_1
#else
#define NNMF_FUNCTION nnmf
#endif


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

    while (use_threshold || counter < num_iterations) {
        counter++;
        verify = 1;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                sum_WH = 0.0;
                for (int k = 0; k < r; k++) {
                    sum_WH += W[i * r + k] * H[k * n + j];
                }

                WH[i * n + j] = sum_WH;

                abs_value = fabs(sum_WH - V[i * n + j]);
                //printf("(%lf, %lf)\n",sum_WH, V[i * n + j]);
                //printf("Opt1, Iteration: %d, Abs: %lf\n", counter, abs_value);
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

        for (int i = 0; i < r; i++) {
            for (int j = 0; j < n; j++) {
                sum_WV = 0.0;
                sum_WWH = 0.0;

                for (int k = 0; k < m; k++) {
                    sum_WV += W[k * r + i] * V[k * n + j];
                    sum_WWH += W[k * r + i] * WH[k * n + j];
                }

                H[i * n + j] = H[i * n + j] * sum_WV / sum_WWH;
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < r; j++) {
                sum_VH = 0;
                w_reuse = W[i * r + j];

                for (int k = 0; k < n; k++) {
                    h_reuse = H[j * n + k];
                    sum_VH += V[i * n + k] * h_reuse;

                    if (j == 0) {
                        WH[i * n + k] = 0;
                    }

                    WH[i * n + k] += w_reuse * h_reuse;
                }

                VH[i * r + j] = sum_VH;
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < r; j++) {
                sum_WHH = 0.0;
                for (int k = 0; k < n; k++) {
                    sum_WHH += WH[i * n + k] * H[j * n + k];
                }
                W[i * r + j] = W[i * r + j] * VH[i * r + j] / sum_WHH;
            }
        }
    }

    free(VH);
    free(WH);

    *iteration_count = counter;
    return;
}