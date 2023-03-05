#include <stdlib.h>
#include <stdio.h>

#include "../nnmf_common.h"
#include "../nnmf.h"

#undef NNMF_FUNCTION
#ifdef VERIFICATION_INLINE
 #define NNMF_FUNCTION nnmf_basic
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
        if (verify(V, W, H, n, m, r, threshold)) {
            if (use_threshold) {
                break;
            }
        }
        int i,j;
        transpose(W,m,r,Wtranspose);
        multiply(Wtranspose,V,r,m,n,WV);
        multiply(Wtranspose,W,r,m,r,WW);
        multiply(WW,H,r,r,n,WWH);
        for(i = 0; i < r; i++){
            for(j = 0; j < n; j++){
                H[i*n+j] = H[i*n+j]*(WV[i*n+j])/(WWH[i*n+j]);
            }
        }
        transpose(H,r,n,Htranspose);
        multiply(V,Htranspose,m,n,r,VH);
        multiply(W,H,m,r,n,WH);
        multiply(WH,Htranspose,m,n,r,WHH);
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