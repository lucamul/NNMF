#include "type_def.h"

/**
 * @brief Initializes m x n matrix V
 * 
 * in: m, n - matrix dimensions
 * out: V   - initialized matrix V (memory already allocated)
 */
void init_V(float_type* V, const int m, const int n);

/**
 * @brief Matrix Multiplication- triple loop implementation
 * 
 * in: A       - m x r matrix A
 * in: B       - r x n matrix B
 * in: m, n, r - matrix dimensions 
 * out: C      - m x n matrix C = AB 
 */
void multiply(const float_type* A, const float_type* B, const int m, const int r, const int n, float_type* C);

/**
 * @brief Matrix Transpose
 * 
 * in: A    - m x n matrix A
 * in: m, n - matrix dimensions 
 * out: C   - n x m matrix C 
 */
void transpose(float_type* A, const int m, const int n, float_type* C);

/**
 * @brief Verify if all elements of the the matrices V minus elements of W*H are smaller than threshold
 * 
 * in: V       - m x n matrix V
 * in: W       - m x r matrix W
 * in: H       - r x n matrix H
 * in: m, n, r - matrix dimensions 
 * out: 1      - verify succeeded
 *      0      - verify didn't succeed
 */
int verify(const float_type* V, float_type* W, float_type* H, const int n, const int m, const int r, const int threshold);

/**
 * @brief Initializes matrices W and H
 * 
 * in: V       - input matrix V 
 * in: m, n, r - matrix dimensions
 * out: W, H   - matrices to initialize (W: m x r, H: r x n) (memory already allocated)
 */
void init_WH(const float_type *V, float_type *W, float_type *H, const int m, const int n, const int r);