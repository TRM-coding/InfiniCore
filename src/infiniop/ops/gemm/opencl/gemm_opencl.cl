
#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifndef T
#define T float
#endif

#ifndef TILE_M
#define TILE_M 16
#endif

#ifndef TILE_N
#define TILE_N 16
#endif

#ifndef TILE_K
#define TILE_K 16
#endif

typedef int Tidx;

// Basic GEMM kernel: C = alpha * A * B + beta * C
kernel void gemm_kernel(
    global T *C,
    Tidx const c_row_stride,
    Tidx const c_col_stride,
    global T const *A, 
    Tidx const a_row_stride,
    Tidx const a_col_stride,
    global T const *B,
    Tidx const b_row_stride,
    Tidx const b_col_stride,
    T const alpha,
    T const beta,
    Tidx const M,
    Tidx const N, 
    Tidx const K,
    Tidx const batch_stride_a,
    Tidx const batch_stride_b,
    Tidx const batch_stride_c) {

    Tidx batch_id = (get_work_dim() >= 3) ? get_group_id(2) : 0;
    Tidx global_row = get_global_id(0);  // M dimension
    Tidx global_col = get_global_id(1);  // N dimension
    
    if (global_row >= M || global_col >= N) return;
    
    // Offset pointers for batched operation - handle single batch case
    global T const *A_batch = A + (batch_stride_a > 0 ? batch_id * batch_stride_a : 0);
    global T const *B_batch = B + (batch_stride_b > 0 ? batch_id * batch_stride_b : 0);  
    global T *C_batch = C + (batch_stride_c > 0 ? batch_id * batch_stride_c : 0);
    
    T acc = 0;
    
    // Compute dot product for C[global_row][global_col]
    for (Tidx k = 0; k < K; ++k) {
        Tidx a_idx = global_row * a_row_stride + k * a_col_stride;
        Tidx b_idx = k * b_row_stride + global_col * b_col_stride;
        T a_val = A_batch[a_idx];
        T b_val = B_batch[b_idx];
        acc += a_val * b_val;
    }
    
    // Apply alpha and beta scaling
    Tidx c_idx = global_row * c_row_stride + global_col * c_col_stride;
    T c_val = C_batch[c_idx];
    C_batch[c_idx] = alpha * acc + beta * c_val;
}

// Optimized tiled GEMM kernel for better performance
kernel void gemm_tiled_kernel(
    global T *C,
    Tidx const c_row_stride,
    Tidx const c_col_stride,
    global T const *A,
    Tidx const a_row_stride,
    Tidx const a_col_stride, 
    global T const *B,
    Tidx const b_row_stride,
    Tidx const b_col_stride,
    T const alpha,
    T const beta,
    Tidx const M,
    Tidx const N,
    Tidx const K,
    Tidx const batch_stride_a,
    Tidx const batch_stride_b,
    Tidx const batch_stride_c) {
    
    local T tile_a[TILE_M][TILE_K];
    local T tile_b[TILE_K][TILE_N];
    
    Tidx batch_id = (get_work_dim() >= 3) ? get_group_id(2) : 0;
    Tidx local_row = get_local_id(0);
    Tidx local_col = get_local_id(1);
    Tidx group_row = get_group_id(0);
    Tidx group_col = get_group_id(1);
    
    Tidx global_row = group_row * TILE_M + local_row;
    Tidx global_col = group_col * TILE_N + local_col;
    
    // Offset pointers for batched operation - handle single batch case
    global T const *A_batch = A + (batch_stride_a > 0 ? batch_id * batch_stride_a : 0);
    global T const *B_batch = B + (batch_stride_b > 0 ? batch_id * batch_stride_b : 0);
    global T *C_batch = C + (batch_stride_c > 0 ? batch_id * batch_stride_c : 0);
    
    T acc = 0;
    
    // Loop over tiles
    for (Tidx tile_k = 0; tile_k < K; tile_k += TILE_K) {
        // Load tile of A into local memory
        if (global_row < M && (tile_k + local_col) < K) {
            Tidx a_idx = global_row * a_row_stride + (tile_k + local_col) * a_col_stride;
            tile_a[local_row][local_col] = A_batch[a_idx];
        } else {
            tile_a[local_row][local_col] = 0;
        }
        
        // Load tile of B into local memory
        if ((tile_k + local_row) < K && global_col < N) {
            Tidx b_idx = (tile_k + local_row) * b_row_stride + global_col * b_col_stride;
            tile_b[local_row][local_col] = B_batch[b_idx];
        } else {
            tile_b[local_row][local_col] = 0;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial result for this tile
        for (Tidx k = 0; k < TILE_K; ++k) {
            acc += tile_a[local_row][k] * tile_b[k][local_col];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Write result back to global memory
    if (global_row < M && global_col < N) {
        Tidx c_idx = global_row * c_row_stride + global_col * c_col_stride;
        T c_val = C_batch[c_idx];
        C_batch[c_idx] = alpha * acc + beta * c_val;
    }
}