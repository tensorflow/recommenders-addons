/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */


#ifndef NV_HASHTABLE_H_
#define NV_HASHTABLE_H_
#include "thrust/pair.h"
#include "cudf/concurrent_unordered_map.cuh"
#include "nv_util.h"
#include <mutex>


namespace nv {

template<typename value_type>
struct ReplaceOp {
  constexpr static value_type IDENTITY{0};

  __host__ __device__ value_type operator()(value_type new_value, value_type old_value)
  {
      return new_value;
  }
};


template<typename Table>
__global__ void insert_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const typename Table::mapped_type* const vals,
                              size_t len) {

    thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        kv.first = keys[i];
        kv.second = vals[i];
        auto it = table->insert(kv);
        assert(it != table->end() && "error: insert fails: table is full");
    }
}

template<typename Table>
__global__ void insert_kernel_mask(Table* table,
                              const typename Table::key_type* const keys,
                              const typename Table::mapped_type* const vals,
                              const bool* const status,
                              size_t len) {

    thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        if(!status[i]){
            kv.first = keys[i];
            kv.second = vals[i];
            auto it = table->insert(kv);
            assert(it != table->end() && "error: insert fails: table is full");
        }
    }
}

template<typename Table>
__global__ void upsert_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const typename Table::mapped_type* const vals,
                              size_t len) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        auto it = table->find(keys[i]);
        if(it.Iterator != table->end()){
            ((it.Iterator).getter())->second = vals[i];
        } else {
            thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;
            kv.first = keys[i];
            kv.second = vals[i];
            auto it = table->insert(kv);
            assert(it != table->end() && "error: insert fails: table is full");
        }
    }
}


template <typename Table>
__global__ void accum_kernel(
    Table* table, const typename Table::key_type* const keys,
    const typename Table::mapped_type* const vals_or_deltas, const bool* exists,
    size_t len) {

  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    //thrust::pair<typename Table::key_type, typename Table::mapped_type> kv;
    //kv.first = keys[i];
    //kv.second = vals_or_deltas[i];
    auto it = table->accum(keys[i], vals_or_deltas[i], exists[i]);
  }
}


template<typename Table>
__global__ void set_kernel(Table* table,
                            const typename Table::key_type* const keys,
                            const typename Table::mapped_type* const vals,
                            size_t len){

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        auto it = table->find(keys[i]);
        assert(it.Iterator != table->end() && "error: can't find key");
        ((it.Iterator).getter())->second = vals[i];
    }
}
                            
template<typename Table>
__global__ void set_kernel_mask(Table* table,
                            const typename Table::key_type* const keys,
                            const typename Table::mapped_type* const vals,
                            const bool *status,
                            size_t len){

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        if (status[i]) {
            auto it = table->find(keys[i]);
            assert(it.Iterator != table->end() && "set_kernel_mask: can't find key, but should exist.");
            ((it.Iterator).getter())->second = vals[i];
        }
    }
}

template<typename Table, typename GradType, typename Optimizer>
__global__ void update_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              const GradType* const gradients,
                              size_t len,
                              Optimizer& op){

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        auto it = table->find(keys[i]);
        assert(it.Iterator != table->end() && "error: can't find key");
        op.update(((it.Iterator).getter())->second, gradients[i]);
    }
}


template<typename Table>
__global__ void search_kernel(Table* table,
                             const typename Table::key_type * const keys,
                             typename Table::mapped_type * const vals,
                             bool* const status,
                             size_t len,
                             typename Table::mapped_type* const def_val,
                             bool full_size_default) {

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        auto it = table->find(keys[i]);
        if(it.Iterator != table->end()){
            vals[i] = ((it.Iterator).getter())->second;
            status[i] = true;
        }
        else{
            vals[i] = full_size_default ? def_val[i] : def_val[0];
            status[i] = false;
        }
    }
}
                             
template<typename Table>
__global__ void get_status_kernel(Table* table,
                             const typename Table::key_type * const keys,
                             bool* const status,
                             size_t len) {

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        auto it = table->find(keys[i]);
        //assert(it != table->end() && "error: can't find key");
        if(it.Iterator != table->end()){
            status[i] = true;
        }
        else{
            status[i] = false;
        }
    }
}


template<typename Table, typename counter_type>
__global__ void get_insert_kernel(Table* table,
                                  const typename Table::key_type * const keys,
                                  typename Table::mapped_type * const vals,
                                  size_t len,
                                  counter_type * d_counter) {

    ReplaceOp<typename Table::mapped_type> op;
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        auto it = table->get_insert(keys[i], op, d_counter);
        vals[i] = it->second;
    }
}

template<typename Table, typename KeyType>
__global__ void size_kernel(const Table* table,
                            const size_t hash_capacity,
                            size_t * table_size,
                            KeyType unused_key) {
    /* Per block accumulator */
    __shared__ size_t block_acc;

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Initialize */
    if(threadIdx.x == 0){
        block_acc = 0;
    }
    __syncthreads();

    /* Whether the bucket mapping to the current thread is empty? do nothing : Atomically add to counter */
    if(i < hash_capacity){
        typename Table::value_type val = load_pair_vectorized(table->data() + i);
        bool valid = table->get_valid(i);
        if((val.first != unused_key) && valid){
            atomicAdd(&block_acc, 1);
        }
    }
    __syncthreads();

    /* Atomically reduce block counter to global conuter */
    if(threadIdx.x == 0){
        atomicAdd(table_size, block_acc);
    }
}


template<typename KeyType, typename ValType, typename Table>
__global__ void dump_kernel(KeyType* d_key, 
                            ValType* d_val, 
                            const Table* table, 
                            const size_t offset, 
                            const size_t search_length, 
                            size_t * d_dump_counter,
                            KeyType unused_key){
    
    // inter-block gathered key, value and counter. Global conuter for storing shared memory into global memory. 
    //__shared__ KeyType block_result_key[BLOCK_SIZE_];
    //__shared__ ValType block_result_val[BLOCK_SIZE_];
    extern __shared__ unsigned char s[];
    KeyType * smem = (KeyType *)s;
    KeyType * block_result_key = smem;
    ValType * block_result_val = (ValType *) &(smem[blockDim.x]);
    __shared__ size_t block_acc;
    __shared__ size_t global_acc;

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    /* Initialize */
    if(threadIdx.x == 0){
        block_acc = 0;
    }
    __syncthreads();

    // Each thread gather the key and value from bucket assigned to them and store them into shared mem.
    if(i < search_length){
        typename Table::value_type val = load_pair_vectorized(table->data() + offset + i);
        bool valid = table->get_valid(offset + i);
        if((val.first != unused_key) && valid){
            size_t local_index = atomicAdd(&block_acc, 1);
            block_result_key[local_index] = val.first;
            block_result_val[local_index] = val.second;
        }
    }
    __syncthreads();

    //Each block request a unique place in global memory buffer, this is the place where shared memory store back to.
    if(threadIdx.x == 0){
        global_acc = atomicAdd(d_dump_counter, block_acc);
    }
    __syncthreads();

    //Each thread store one bucket's data back to global memory, d_dump_counter is how many buckets in total dumped.
    if(threadIdx.x < block_acc){
        d_key[global_acc + threadIdx.x] = block_result_key[threadIdx.x];
        d_val[global_acc + threadIdx.x] = block_result_val[threadIdx.x];
    }
}

template<typename Table>
__global__ void delete_kernel(Table* table,
                              const typename Table::key_type* const keys,
                              size_t len) {

    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        auto it = table->find(keys[i]);
        assert(it.Iterator != table->end() && "error: can't find key");
        table->set_valid(it.current_index, false);
    }
}

template<typename KeyType, typename ValType, typename BaseValType, KeyType empty_key, size_t DIM, typename counter_type = unsigned long long int>
class HashTable {
public:
    HashTable(size_t capacity, counter_type count = 0) {
        //assert(capacity <= std::numeric_limits<ValType>::max() && "error: Table is too large for the value type");
        cudaDeviceProp deviceProp;
        table_ = new Table(capacity, std::numeric_limits<ValType>::max());
        update_counter_ = 0;
        get_counter_ = 0;
        // Allocate device-side counter and copy user input to it
        CUDA_CHECK(cudaMallocManaged((void **)&d_counter_, sizeof(*d_counter_)));
        CUDA_CHECK(cudaMemcpy(d_counter_, &count, sizeof(*d_counter_), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp ,0));
        shared_mem_size = deviceProp.sharedMemPerBlock;
    }
    ~HashTable() {
        delete table_;
        // De-allocate device-side counter
        CUDA_CHECK(cudaFree(d_counter_));
    }
    HashTable(const HashTable&) = delete;
    HashTable& operator=(const HashTable&) = delete;


    void insert(const KeyType* d_keys, const BaseValType* d_vals, size_t len, cudaStream_t stream) {
        if (len == 0) {
            return;
        }
        const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
        insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, (const ValType*)d_vals, len);
    }

    void upsert(const KeyType* d_keys, const BaseValType* d_vals, size_t len, cudaStream_t stream) {
        if (len == 0) {
            return;
        }
        const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
        upsert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, (const ValType*)d_vals, len);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void get(const KeyType* d_keys, BaseValType* d_vals, bool* d_status, size_t len, BaseValType *d_def_val, cudaStream_t stream, bool full_size_default) const {
        if (len == 0) {
            return;
        }
        CUDA_CHECK(cudaMemset((void *)d_vals, 0, sizeof(ValType) * len));
        const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
        search_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, (ValType*)d_vals, d_status, len, (ValType*)d_def_val, full_size_default);
    }
    
    void get_status(const KeyType* d_keys, bool* d_status, size_t len, cudaStream_t stream) const {
        if (len == 0) {
            return;
        }
        const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
        get_status_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_status, len);
    }

    void set(const KeyType* d_keys, const BaseValType* d_vals, size_t len, cudaStream_t stream) {
        if (len == 0) {
            return;
        }
        const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
        set_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, (const ValType*)d_vals, len);
    }

    void accum(const KeyType* d_keys, const BaseValType* d_vals_or_deltas, const bool* d_exists, size_t len,
               cudaStream_t stream) {
        if (len == 0) {
            return;
        }
        const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
        accum_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys,
                                                            (const ValType*)d_vals_or_deltas, d_exists, len);
    }

    size_t get_size(cudaStream_t stream) const{
        /* size variable on Host and device, total capacity of the hashtable */
        size_t table_size;
        size_t * d_table_size;
        const size_t hash_capacity = table_-> size();

        /* grid_size and allocating/initializing variable on dev, lauching kernel*/
        const int grid_size = (hash_capacity - 1) / BLOCK_SIZE_ + 1;
        CUDA_CHECK(cudaMallocManaged((void **)&d_table_size, sizeof(size_t)));
        CUDA_CHECK(cudaMemset ( d_table_size, 0, sizeof(size_t)));
        size_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, hash_capacity, d_table_size, empty_key);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaMemcpy(&table_size, d_table_size, sizeof(size_t),  cudaMemcpyDeviceToHost));

        /* Copy result back and do clean up*/
        CUDA_CHECK(cudaFree(d_table_size));
        return table_size;
    }

    void dump(KeyType* d_key, BaseValType* d_val, const size_t offset, const size_t search_length, size_t * d_dump_counter, cudaStream_t stream) const{
        //Before we call the kernel, set the global counter to 0
        CUDA_CHECK(cudaMemset (d_dump_counter, 0, sizeof(size_t)));
        // grid size according to the searching length.
        size_t block_size = shared_mem_size * 0.5 / (sizeof(KeyType) + sizeof(ValType)) ;
        block_size = block_size <= 1024 ? block_size : 1024;
        assert(block_size > 0 && "nvhash: block_size <= 0, the KV size may be too large!");
        size_t shared_size = sizeof(* d_key) * block_size + sizeof(ValType) * block_size;
        const int grid_size = (search_length - 1) / (block_size) + 1;
        
        dump_kernel<<<grid_size, block_size, shared_size, stream>>>(d_key, (ValType* )d_val, table_, offset, search_length, d_dump_counter, empty_key);
    }

    size_t get_capacity() const{
        return (table_-> size());
    }

    counter_type get_value_head() const{
        counter_type counter;
        CUDA_CHECK(cudaMemcpy(&counter, d_counter_, sizeof(*d_counter_), cudaMemcpyDeviceToHost));
        return counter;
    }

    void set_value_head(counter_type counter_value){
        CUDA_CHECK(cudaMemcpy(d_counter_, &counter_value, sizeof(*d_counter_), cudaMemcpyHostToDevice));
    }

    counter_type add_value_head(counter_type counter_add){
        counter_type counter;
        CUDA_CHECK(cudaMemcpy(&counter, d_counter_, sizeof(*d_counter_), cudaMemcpyDeviceToHost));
        counter += counter_add;
        CUDA_CHECK(cudaMemcpy(d_counter_, &counter, sizeof(*d_counter_), cudaMemcpyHostToDevice));
        return counter;
    }

    template<typename GradType, typename Optimizer>
    void update(const KeyType* d_keys, const GradType* d_gradients, size_t len, cudaStream_t stream, Optimizer& op){
        if (len == 0) {
            return;
        }
        const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
        update_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, d_gradients, len, op);
    }

    void get_insert(const KeyType* d_keys, void* d_vals, size_t len, cudaStream_t stream){
        if (len == 0) {
            return;
        }
        const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
        get_insert_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, (ValType*)d_vals, len, d_counter_);
    }

    // Before any get API is called, call this to check and update counter
    bool get_lock(){
        counter_mtx_.lock();
        bool ret_val;
        if(update_counter_ > 0){
            ret_val = false; // There are update APIs running, can't do get.
        }
        else{
            get_counter_++;
            ret_val = true; // There is no update API running, can do get, increase counter
        }
        counter_mtx_.unlock();
        return ret_val;
    }

    // Before any update API is called, call this to check and update counter
    bool update_lock(){
        counter_mtx_.lock();
        bool ret_val;
        if(get_counter_ > 0){
            ret_val = false; // There are get APIs running, can't do update
        }
        else{
            update_counter_++;
            ret_val = true; // There is no get API running, can do update, increase counter
        }
        counter_mtx_.unlock();
        return ret_val;
    }

    // After each get API finish on this GPU's hashtable, decrease the counter
    void get_release(){
        counter_mtx_.lock();
        get_counter_--; // one get API finish, dec counter
        counter_mtx_.unlock();
    }

    void update_release(){
        counter_mtx_.lock();
        update_counter_--; // one update API finish, dec counter
        counter_mtx_.unlock();
    }

    void clear(cudaStream_t stream){
        table_-> clear_async(stream);
    }

    void remove(const KeyType* d_keys, size_t len, cudaStream_t stream){
        if (len == 0) {
            return;
        }
        const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
        delete_kernel<<<grid_size, BLOCK_SIZE_, 0, stream>>>(table_, d_keys, len);
    }


private:
    static const int BLOCK_SIZE_ = 256;
    using Table = concurrent_unordered_map<KeyType, ValType, empty_key, DIM>;

    Table* table_;

    // GPU-level lock and counters for get and update APIs
    std::mutex counter_mtx_; // Lock that protect the counters
    volatile size_t update_counter_; // How many update APIs are currently called on this GPU' hashtable 
    volatile size_t get_counter_; // How many get APIs are currently called on this GPU's hashtable 

    // Counter for value index
    counter_type * d_counter_; 
    size_t shared_mem_size;
};


}
#endif
