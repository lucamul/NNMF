#include <stdlib.h>
#include <math.h>

#include "../nnmf_common.h"
#include "../nnmf.h"

#undef NNMF_FUNCTION
#ifdef VERIFICATION_INLINE
 #define NNMF_FUNCTION nnmf_inline
#else
 #define NNMF_FUNCTION nnmf
#endif

void NNMF_FUNCTION(const float_type* V, float_type* W, float_type* H, const int m, const int n, const int r, long* iteration_count, int use_threshold, const int threshold, const int num_iterations) {
    float_type* Wtranspose = malloc(m*r*sizeof(float_type));
    float_type* WV = malloc(r*n*sizeof(float_type));
    float_type* WW = malloc(r*r*sizeof(float_type));
    float_type* WWH = malloc(r*n*sizeof(float_type));
    float_type* Htranspose = malloc(n*r*sizeof(float_type));
    float_type* VH = malloc(m*r*sizeof(float_type));
    float_type* WH = malloc(m*n*sizeof(float_type));
    float_type* WHH = malloc(m*r*sizeof(float_type));
    int counter = -1;

    while(use_threshold || counter < num_iterations){
        counter++;

        float_type WH[m * n];
        //multiply(W, H, m, r, n, WH);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                WH[i * n + j] = 0;
                for (int k = 0; k < r; k++) {
                    WH[i * n + j] += W[i * r + k] * H[k * n + j];
                }
            }
        }

        int verify = 1;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float_type x = fabs(WH[i * n + j] - V[i * n + j]);
                if (x >= threshold) {
                    verify = 0;
                    break;
                }
            }
            if (!verify) {
                break;
            }
        }

        if (verify) {
            if (use_threshold) {
                break;
            }
        }

        int i,j;

        //transpose(W,m,r,Wtranspose);
        for(int i = 0; i < m; i++){
            for(int j = 0; j < r; j++){
                Wtranspose[j*m+i] = W[i*r+j];
            }
        }

        //multiply(Wtranspose,V,r,m,n,WV);
        for(int i = 0; i < r; i++){
            for(int j = 0; j < n; j++){
                WV[i*n+j] = 0;
                for(int k = 0; k < m; k++){
                    WV[i*n+j] += Wtranspose[i*m+k]*V[k*n+j];
                }
            }
        }

        //multiply(Wtranspose,W,r,m,r,WW);
        for(int i = 0; i < r; i++){
            for(int j = 0; j < r; j++){
                WW[i*r+j] = 0;
                for(int k = 0; k < m; k++){
                    WW[i*r+j] += Wtranspose[i*m+k]*W[k*r+j];
                }
            }
        }

        //multiply(WW,H,r,r,n,WWH);
        for(int i = 0; i < r; i++){
            for(int j = 0; j < n; j++){
                WWH[i*n+j] = 0;
                for(int k = 0; k < r; k++){
                    WWH[i*n+j] += WW[i*r+k]*H[k*n+j];
                }
            }
        }

        for(i = 0; i < r; i++){
            for(j = 0; j < n; j++){
                H[i*n+j] = H[i*n+j]*(WV[i*n+j])/(WWH[i*n+j]);
            }
        }

        //transpose(H,r,n,Htranspose);
        for(int i = 0; i < r; i++){
            for(int j = 0; j < n; j++){
                Htranspose[j*r+i] = H[i*n+j];
            }
        }

        //multiply(V,Htranspose,m,n,r,VH);
        for(int i = 0; i < m; i++){
            for(int j = 0; j < r; j++){
                VH[i*r+j] = 0;
                for(int k = 0; k < n; k++){
                    VH[i*r+j] += V[i*n+k]*Htranspose[k*r+j];
                }
            }
        }

        //multiply(W,H,m,r,n,WH);
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n; j++){
                WH[i*n+j] = 0;
                for(int k = 0; k < r; k++){
                    WH[i*n+j] += W[i*r+k]*H[k*n+j];
                }
            }
        }

        //multiply(WH,Htranspose,m,n,r,WHH);
        for(int i = 0; i < m; i++){
            for(int j = 0; j < r; j++){
                WHH[i*r+j] = 0;
                for(int k = 0; k < n; k++){
                    WHH[i*r+j] += WH[i*n+k]*Htranspose[k*r+j];
                }
            }
        }

        for(i = 0; i < m; i++){
            for(j = 0; j < r; j++){
                W[i*r+j] = W[i*r+j]*VH[i*r+j]/WHH[i*r+j];
            }
        }
    }
    free(Wtranspose);
    free(WV);
    free(WW);
    free(WWH);
    free(Htranspose);
    free(VH);
    free(WH);
    free(WHH);
    *iteration_count = counter;
    return;
}