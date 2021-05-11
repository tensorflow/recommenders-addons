/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef NV_MULTISPLIT_H_
#define NV_MULTISPLIT_H_
#include <cassert>
#include "cub/device/device_radix_sort.cuh"
#include "cub/device/device_run_length_encode.cuh"
#include "nv_util.h"


namespace nv {

template<typename T>
__global__ void fill_idx(T* idx, size_t len) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        idx[i] = i;
    }
}


/*template<typename KeyType>
__global__ void fill_mod(const KeyType* keys, size_t len, int modulo, int* result) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        result[i] = keys[i] % modulo;
    }
}*/

/* This template kernel now calling user-defined policy device function to distribute pairs according to the key*/
template<typename KeyType, typename KeyGPUMapPolicy_>
__global__ void fill_partition(const KeyType* keys, size_t len, int num_gpu, int* result, KeyGPUMapPolicy_& policy){
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len){
        result[i] =  policy.map_policy(num_gpu, keys[i]);
    }
}


inline int log2i(unsigned x) {
    assert(x != 0);
    unsigned res = 0;
    while (x >>= 1) {
        ++res;
    }
    return res;
}



template<typename Allocator>
void calc_partition_sizes(const int* d_data, size_t len, size_t* d_sizes, int max_num_part, cudaStream_t stream, Allocator& allocator) {
    int* d_uniq;
    int* d_num_part;
    allocator.malloc((void**) &d_uniq, max_num_part * sizeof(*d_uniq));
    allocator.malloc((void**) &d_num_part, sizeof(*d_num_part));

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(NULL, temp_storage_bytes,
                                                  d_data,
                                                  d_uniq, d_sizes, d_num_part,
                                                  len,
                                                  stream));

    allocator.malloc(&d_temp_storage, temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes,
                                                  d_data,
                                                  d_uniq, d_sizes, d_num_part,
                                                  len,
                                                  stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    allocator.free(d_temp_storage);

    int h_num_part;
    CUDA_CHECK(cudaMemcpy(&h_num_part, d_num_part, sizeof(h_num_part), cudaMemcpyDeviceToHost));
    assert(h_num_part <= max_num_part);

    if (h_num_part < max_num_part) {
        int* h_uniq = new int[h_num_part];
        size_t* h_sizes = new size_t[h_num_part];
        CUDA_CHECK(cudaMemcpy(h_uniq, d_uniq, h_num_part * sizeof(*h_uniq), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_sizes, d_sizes, h_num_part * sizeof(*h_sizes), cudaMemcpyDeviceToHost));

        size_t* h_sizes_new = new size_t[max_num_part];
        memset(h_sizes_new, 0, max_num_part * sizeof(*h_sizes_new));

        for (int i = 0; i < h_num_part; ++i) {
            assert(h_uniq[i] < max_num_part);
            h_sizes_new[h_uniq[i]] = h_sizes[i];
        }
        CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes_new, max_num_part * sizeof(*d_sizes), cudaMemcpyHostToDevice));

        delete [] h_uniq;
        delete [] h_sizes;
        delete [] h_sizes_new;
    }

    allocator.free(d_uniq);
    allocator.free(d_num_part);
}




template<typename KeyType, typename Allocator, typename KeyGPUMapPolicy_>
void multisplit(const KeyType* d_keys,
                size_t* d_idx,
                size_t len,
                size_t* d_part_sizes,
                int num_part,
                cudaStream_t stream,
                Allocator& allocator,
                KeyGPUMapPolicy_& policy) {
    int* d_mod_in;
    int* d_mod_out;
    size_t* d_idx_in;
    allocator.malloc((void**) &d_mod_in, len * sizeof(*d_mod_in));
    allocator.malloc((void**) &d_mod_out, len * sizeof(*d_mod_out));
    allocator.malloc((void**) &d_idx_in, len * sizeof(*d_idx_in));


    const int block_size = 256;
    const int grid_size = (len - 1) / block_size + 1;
    fill_idx<<<grid_size, block_size, 0, stream>>>(d_idx_in, len);
    //fill_mod<<<grid_size, block_size, 0, stream>>>(d_keys, len, num_part, d_mod_in);
    fill_partition<<<grid_size, block_size, 0, stream>>>(d_keys, len, num_part, d_mod_in, policy);


    void* d_temp_storage = NULL;
    size_t temp_storage_bytes;
    const int num_bits = 1 + log2i(num_part);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes,
                                               d_mod_in, d_mod_out,
                                               d_idx_in, d_idx,
                                               len,
                                               0,
                                               num_bits,
                                               stream));

    allocator.malloc(&d_temp_storage, temp_storage_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                               d_mod_in, d_mod_out,
                                               d_idx_in, d_idx,
                                               len,
                                               0,
                                               num_bits,
                                               stream));

    calc_partition_sizes(d_mod_out, len, d_part_sizes, num_part, stream, allocator);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    allocator.free(d_temp_storage);

    allocator.free(d_mod_in);
    allocator.free(d_mod_out);
    allocator.free(d_idx_in);
}


}



#endif

