#include "init_w.h"
#include <assert.h>

int main(void) {
    double test_matrix[] = {3.3, 2.3, 1.3, 4.3, 4.5, 3.5, 2.5, 5.5, 7.6, 6.6, 5.6, 8.6, 8.43, 9.43, 7.43, 10.43, 9.2, 8.2, 7.2, 10.2};
    int cols = 4;
    int rows = 5;

    //Tests: compute_column_norm
    double result = compute_column_norm(test_matrix, rows, cols, 0);
    double expected_result = 15;
    assert(floor(result) == expected_result);
    printf("Test Passed: compute_column_norm\n");

    //Tests: bubble_sort
    double unordered_col_norms[] = {9.7, 8.6, 5.4, 2.4, 6.5, 3.2, 0};
    double expected_col_norms[] = {6 , 3, 5, 2, 4, 1, 0};
    int cols_local = 7;
    int* ordered_col_norms = malloc(sizeof(int) * cols_local);
    bubble_sort(unordered_col_norms, cols_local, ordered_col_norms);
    for(int i =0; i < cols_local; i++){
       assert(expected_col_norms[i] == ordered_col_norms[i]);
    }
    free(ordered_col_norms);
    printf("Test Passed: bubble_sort\n");

    //Tests: order_V_columns_by_second_norm
    int expected_col_norms_order[] = {2 , 1, 0, 3};
    int* actual_col_norms_order = malloc(sizeof(int) * 4);
    order_V_columns_by_second_norm(test_matrix, rows, cols, actual_col_norms_order);
    for (int i = 0; i < cols; i++)
    {
        assert(expected_col_norms_order[i] == actual_col_norms_order[i]);
    }
    free(actual_col_norms_order);
    printf("Test Passed: order_V_columns_by_second_norm\n");

    //Tests: compute_mean_of_q_cols
    int selected_columns[]= {0,1,2,2};
    double* computed_mean_column = malloc(sizeof(double) * 5);
    compute_mean_of_q_cols(test_matrix, selected_columns, rows, cols, 4, computed_mean_column);
    double expected_computed_column[] = {2.05, 3.25, 6.35, 8.18, 7.95};
    for (int i = 0; i < cols; i++)
    {
        assert(expected_computed_column[i] == computed_mean_column[i]);
    }
    free(computed_mean_column);
    printf("Test Passed: compute_mean_of_q_cols\n");
}