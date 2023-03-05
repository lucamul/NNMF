#include <immintrin.h>
#include "type_def.h"

/**
 * @brief print m x n matrix A
 *
 * in: A    - m x n matrix A
 * in: m, n - matrix dimensions
 * out: C   - m x n matrix C = AB
 */
void print_matrix(float_type* A, int m, int n, char* name);

/**
 * @brief verifies results
 */
void verify_results(
    const float_type* V, const float_type* W, const float_type* H, 
    const int m, const int n, const int r, const int threshold, float_type* norm);

/**
 * @brief Generates a random number of type float type
 * 
 * in: min, max - range for the random value
 * out:         - a random number of type float_type
 */
float_type rand_from(float_type min, float_type max);

/**
 * @brief print __m256 vector of floats
 *
 * in: name - name of the variable, will be printed on the console
 * in: var  - vector
 */
void print__mm256(char* name, __m256 var);

/**
 * @brief print __m128 vector of floats
 *
 * in: name - name of the variable, will be printed on the console
 * in: var  - vector
 */
void print__mm128(char* name, __m128 var);

/**
 * @brief manually aligns the vector to 64 bytes
 *
 * in: size_in_bytes - size in bytes
 */
float_type* manual_align_malloc(size_t size_in_bytes);

/**
 * @brief manually frees the vector previously aligned to 64 bytes
 *
 * in: size_in_bytes - size in bytes
 */
void manual_free(void* ptr);