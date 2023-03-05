#include "init_w.h"

/*
//check validity of r
int verify_r(int r, int m, int n){
    return r < m ^ ((m ^ n) & -(m < n));
}
*/

int verify_sample_space(int q, int cols, int poolsize){
    return q <= poolsize <= cols;
}

float_type compute_column_norm(const float_type* V_matrix, int rows, int cols, int col){
    float_type norm = 0.0;
    for(int row_number = 0; row_number < rows; row_number++){
        norm += V_matrix[row_number*cols+col] * V_matrix[row_number*cols+col];
    }
    return sqrt(norm);
}

void bubble_sort(float_type* col_norms, int total_columns, int* ordered_columns_indices){
    for(int i = 0; i < total_columns; i++)
        ordered_columns_indices[i] = i;
    for(int i = 1; i < total_columns; i++){
        float_type y = col_norms[i];
        int z = ordered_columns_indices[i];
        int j = i -1;
        while( j >= 0 && y < col_norms[j]){
            col_norms[j+1] = col_norms[j];
            ordered_columns_indices[j+1] = ordered_columns_indices[j];
            j--;
        }
        col_norms[j+1] = y;
        ordered_columns_indices[j+1] = z;
    }
}

void order_V_columns_by_second_norm(const float_type* V_matrix, int rows, int cols, int* longest_column_of_V){
    float_type col_norms[cols];
    for(int col_number =0; col_number < cols; col_number++){
        col_norms[col_number] = compute_column_norm(V_matrix, rows, cols, col_number);
    }
    bubble_sort(col_norms, cols, longest_column_of_V);
}

void find_q_cols_from_longest_cols(int* longest_cols, int poolsize, int q, int* selected_columns){
    for (int i = 0; i < q; i++) {
        int random_index = rand() % poolsize;
        selected_columns[i] = longest_cols[random_index];
    }
}

void compute_mean_of_q_cols(const float_type* V_matrix, int* selected_columns, int rows, int cols, int q, float_type* mean_column){
    int selected_column;
    for (int i = 0; i < rows; i++) {
        mean_column[i] = 0;
    }
    for(int i = 0; i < q; i++){
        selected_column = selected_columns[i];
        for(int j = 0; j < rows; j++){
            mean_column[j] += V_matrix[j*cols + selected_column];
        }
    }
    for(int i = 0; i < rows; i++){
        mean_column[i] = mean_column[i] / q;
    }
}

void init_random_c(float_type * W, const float_type *V_matrix, int rows, int cols, int r, int q, int poolsize){
    if(verify_sample_space(q, cols, poolsize)){
        int* longest_columns_of_v = malloc(cols*sizeof(int));
        int* selected_q = malloc(q*sizeof(int));
        float_type* mean_column = malloc(rows * sizeof(float_type));

        order_V_columns_by_second_norm(V_matrix, rows, cols, longest_columns_of_v);
        for(int k = 0 ; k < r; k++){
            find_q_cols_from_longest_cols(longest_columns_of_v, poolsize, q, selected_q);
            compute_mean_of_q_cols(V_matrix, selected_q, rows, cols, q, mean_column);
            for(int row_number = 0; row_number < rows; row_number++){
                W[row_number*r + k] = mean_column[row_number];
            }
        }
        free(longest_columns_of_v);
        free(selected_q);
        free(mean_column);
    }else{
        printf("Check your parameters: (r: %d, cols: %d, rows: %d), (q: %d, poolsize: %d)", r, rows, cols, q, poolsize);
    }
}

void init_random(float_type* H, int rows, int cols){
    for (int i = 0; i < rows*cols; i++)
    {
        H[i] = rand();
    }
    
}