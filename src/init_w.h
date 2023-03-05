#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "type_def.h"

// pick Q_RATIO * n out of POOL_SIZE_RATIO * n columns, where n is the number of columns
// Q_RATIO <= POOL_SIZE_RATIO <= 1
#define Q_RATIO 0.25

// choose POOL_SIZE_RATIO * n columns, where n is the number of columns
// e.g. POOL_SIZE_RATIO = 0.5 -> choose out of 0.5*n columns
#define POOL_SIZE_RATIO 0.5

/**
 * @brief Check validity of q and poolsize relative to the columns in v
 * 
 * in: q
 * in: cols
 * in: poolsize
 * out: 0 | 1
 */
int verify_sample_space(int q, int cols, int poolsize);

/**
 * @brief Compute second norm of V's column col
 * 
 * in: V_matrix
 * in: rows
 * in: cols
 * in: col
 * out: second norm of indicated column index
 */
//
float_type compute_column_norm(const float_type* V_matrix, int rows, int cols, int col);

/**
 * @brief Bubble sort
 * 
 * in: col_norms
 * in: total_columns
 * out: ordered_columns_indices; columns sorted by index (using respective calculated norm as a weight)
 */
void bubble_sort(float_type* col_norms, int total_columns, int* ordered_columns_indices);

/**
 * @brief Computes V's column norm size and then reorders them by norm size (max first)
 * 
 * in: V_matrix
 * in: rows
 * in: cols
 * out: longest_column_of_V; columns sorted by index (using respective calculated norm as a weight)
 */
//
void order_V_columns_by_second_norm(const float_type* V_matrix, int rows, int cols, int* longest_column_of_V);

/**
 * @brief Finds a random column from the poolsize for q times where q is fixed
 * 
 * in: longest_cols
 * in: poolsize
 * in: q
 * out: selected_columns; column indices from V's longest columns
 */
void find_q_cols_from_longest_cols(int* longest_cols, int poolsize, int q, int* selected_columns);

/**
 * @brief Computes mean of q columns
 * 
 * in: V_matrix
 * in: selected_columns
 * in: rows
 * in: cols
 * in: q
 * out: mean_column; mean vector
 */
void compute_mean_of_q_cols(const float_type* V_matrix, int* selected_columns, int rows, int cols, int q, float_type* mean_column);

/**
 * @brief Generates W using random_c algorithm
 * 
 * in: V_matrix
 * in: rows
 * in: cols
 * in: r
 * in: q
 * in: poolsize
 */
void init_random_c(float_type* W, const float_type *V_matrix, int rows, int cols, int r, int q, int poolsize);

/**
 * @brief Generates H using random algorithm
 * 
 * in: rows
 * in: cols
 * out: matrix H
 */
void init_random(float_type* H, int rows, int cols);