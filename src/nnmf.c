#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "init_w.h"
#include "nnmf.h"
#include "utils.h"

void init_V(float_type* V, const int m, const int n) {
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            V[i*n+j] = i+j;
            //V[i*n+j] = rand_from(1,100); // converges much faster with V initialized to i+j
        }
    }
}

void multiply(const float_type* A, const float_type* B, const int m, const int r, const int n, float_type* C){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            C[i*n+j] = 0;
            for(int k = 0; k < r; k++){
                C[i*n+j] += A[i*r+k]*B[k*n+j];
            }
        }
    }
    return;
}

void transpose(float_type* A, const int m, const int n, float_type* C){
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            C[j*m+i] = A[i*n+j];
        }
    }
    return;
}

void init_WH(const float_type *V, float_type *W, float_type *H, const int m, const int n, const int r) {
    init_random_c(W, V, m, n, r, Q_RATIO*n, POOL_SIZE_RATIO*n);
    init_random(H, r, n);
}

int verify(const float_type* V, float_type* W, float_type* H, const int n, const int m, const int r, const int threshold){
    float_type WH[m*n];
    multiply(W,H,m,r,n,WH);
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            float_type x = fabs(WH[i*n+j] - V[i*n+j]);
            if (x >= threshold) {
                return 0;
            }
        }
    }
    return 1;
}
