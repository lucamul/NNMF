#include "type_def.h"

/**
 * @brief Non-Negative Matrix Factorization
 * 
 * in: V                - m x n matrix V
 * in: m, n, r          - matrix dimensions 
 * in: use_threshold    - 1: stop the iterative update when the error is smaller than threshold
 *                      - 0: run the iterative update for num_iterations
 * in: threshold        - threshold for the error, used if use_threshold = true
 * in: num_iterations   - number of iterations, used if use_threshold = false
 * in/out: W            - m x r matrix W
 * in/out: H            - r x n matrix H
 * out: iteration_count - number of iterations needed until convergence 
 */
void nnmf(
    const float_type* V, float_type* W, float_type* H, 
    const int m, const int n, const int r, 
    long* iteration_count, int use_threshold, const int threshold, const int num_iterations
    );