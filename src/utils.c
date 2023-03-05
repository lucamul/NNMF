#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h> 

#include "utils.h"

void print_matrix(float_type* A, int m, int n, char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }
}

float_type rand_from(float_type min, float_type max){
    float_type range = max - min;
    float_type div = RAND_MAX / range;
    return min + (rand() / div);
}

void verify_results(const float_type* V, const float_type* W, const float_type* H, const int m, const int n, const int r, const int threshold, float_type* norm) {
    // TODO: flag NDEBUG active if optimization enabled?
    //return;
    
    int mr = m * r, rn = r * n;
    *norm = 0.0;
    
    for (int i = 0; i < mr; ++i)
        assert(W[i] >= 0.0);
    for (int i = 0; i < rn; ++i)
        assert(H[i] >= 0.0);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float_type sum = 0.;
            for (int k = 0; k < r; ++k) {
                sum += W[i * r + k] * H[k * n + j];
            }
            float_type abs_value = fabs(V[i * n + j] - sum);
            assert(abs_value < threshold);
            *norm += abs_value;
        }
    }
}

void print__mm256(char* name, __m256 var) {
    float_type val[8];
    memcpy(val, &var, sizeof(val));
    printf("%s: %lf %lf %lf %lf %lf %lf %lf %lf\n",
        name, val[0], val[1], val[2], val[3], val[4], val[5], val[6], val[7]);
}

void print__mm128(char* name, __m128 var) {
    float_type val[4];
    memcpy(val, &var, sizeof(val));
    printf("%s: %lf %lf %lf %lf\n",
        name, val[0], val[1], val[2], val[3]);
}

float_type* manual_align_malloc(size_t size_in_bytes) {
    char* ptr = (char*)malloc(size_in_bytes + 64);
    size_t offset = ((size_t)ptr) % 64;
    ptr += 64 - offset;
    ptr[-1] = (char)64 - offset;
    return (float*)ptr; ;
}

void manual_free(void* ptr) {
    char* ptr_c = (char*)ptr;
    size_t offset = ptr_c[-1];
    ptr_c -= offset;
    free(ptr_c);
}

