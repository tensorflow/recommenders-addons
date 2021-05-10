/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#ifndef NV_MULTIGPU_H_
#define NV_MULTIGPU_H_
#include <vector>
#include "nv_allocator.h"
#include "nv_hashtable.cuh"
#include "nv_multisplit.cuh"
#include "nv_util.h"
#include <mutex>
#include <chrono>
#include <thread>


namespace nv {

template<typename T1, typename T2>
__global__ void gather(const size_t* d_idx,
                       const T1* d_x_in,
                       const T2* d_y_in,
                       T1* d_x_out,
                       T2* d_y_out,
                       size_t len) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        const size_t idx = d_idx[i];
        d_x_out[i] = d_x_in[idx];
        d_y_out[i] = d_y_in[idx];
    }
}


template<typename T>
__global__ void gather(const size_t* d_idx,
                       const T* d_x_in,
                       T* d_x_out,
                       size_t len) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        const size_t idx = d_idx[i];
        d_x_out[i] = d_x_in[idx];
    }
}


template<typename T>
__global__ void scatter(const size_t* d_idx,
                        const T* d_x_in,
                        T* d_x_out,
                        size_t offset,
                        size_t len) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        const size_t t = offset + i;
        const size_t idx = d_idx[t];
        d_x_out[idx] = d_x_in[t];
    }
}




template<typename KeyType, typename ValType>
struct Resource {
    size_t* d_idx;
    size_t* h_part_sizes;
    size_t* h_offsets;

    KeyType* d_keys_parted;
    ValType* d_vals_parted;
    bool*    d_status_parted;
    KeyType** d_remote_keys;
    ValType** d_remote_vals;
    bool**    d_remote_status;

    //cudaEvent_t* events;
    //cudaStream_t* local_streams;
    //cudaStream_t* remote_streams;
};


struct Stream_event_resource {
    // CUDA events
    cudaEvent_t* local_events; // Events associated with local streams
    cudaEvent_t* remote_events; // Event associated with remote streams
    // CUDA streams
    cudaStream_t* local_streams; // Local streams for calling GPU to communicate with other GPUs that contains the hashtable
    cudaStream_t* remote_streams; // Remote streams for other GPUs to perform their task
    cudaStream_t* caller_stream; // Local stream for calling GPU to perform its own task
};


template<typename KeyType, typename ValType, typename Allocator>
void create_local_resource(Resource<KeyType, ValType>& resource,
                           size_t len,
                           int num_part,
                           Allocator& allocator) {

    allocator.malloc((void**) &resource.d_idx, len * sizeof(*resource.d_idx));

    resource.h_part_sizes = new size_t[num_part];
    resource.h_offsets = new size_t[num_part + 1];

    allocator.malloc((void**) &resource.d_keys_parted, len * sizeof(*resource.d_keys_parted));
    allocator.malloc((void**) &resource.d_vals_parted, len * sizeof(*resource.d_vals_parted));
    allocator.malloc((void**) &resource.d_status_parted, len * sizeof(*resource.d_status_parted));


    //resource.events = new cudaEvent_t[num_part];
    //resource.local_streams = new cudaStream_t[num_part];
    /*for (int i = 0; i < num_part; ++i) {
        CUDA_CHECK(cudaStreamCreate(&resource.local_streams[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&resource.events[i], cudaEventDisableTiming));
    }*/
}

template<typename KeyType, typename ValType, typename Allocator>
void create_remote_resource(Resource<KeyType, ValType>& resource,
                            const std::vector<int>& gpu_id,
                            Allocator& allocator) {

    const int num_part = gpu_id.size();
    resource.d_remote_keys = new KeyType*[num_part];
    resource.d_remote_vals = new ValType*[num_part];
    resource.d_remote_status = new bool*[num_part];
    //resource.remote_streams = new cudaStream_t[num_part];

    CudaDeviceRestorer dev_restorer;
    for (int i = 0; i < num_part; ++i) {
        CUDA_CHECK(cudaSetDevice(gpu_id[i]));
        //CUDA_CHECK(cudaStreamCreate(&resource.remote_streams[i]));
        allocator.malloc((void**) &resource.d_remote_keys[i],
                         resource.h_part_sizes[i] * sizeof(**resource.d_remote_keys));
        allocator.malloc((void**) &resource.d_remote_vals[i],
                         resource.h_part_sizes[i] * sizeof(**resource.d_remote_vals));
        allocator.malloc((void**) &resource.d_remote_status[i],
                         resource.h_part_sizes[i] * sizeof(**resource.d_remote_status));
    }
}


template<typename KeyType, typename ValType, typename Allocator>
void destroy_resource(Resource<KeyType, ValType>& resource,
                      const std::vector<int>& gpu_id,
                      Allocator& allocator) {

    const int num_part = gpu_id.size();
    {
        CudaDeviceRestorer dev_restorer;
        for (int i = 0; i < num_part; ++i) {
            CUDA_CHECK(cudaSetDevice(gpu_id[i]));

            allocator.free(resource.d_remote_keys[i]);
            allocator.free(resource.d_remote_vals[i]);
            allocator.free(resource.d_remote_status[i]);
            //CUDA_CHECK(cudaStreamDestroy(resource.remote_streams[i]));
        }
    }
    delete [] resource.d_remote_keys;
    delete [] resource.d_remote_vals;
    delete [] resource.d_remote_status; 
    //delete [] resource.remote_streams;


    /*for (int i = 0; i < num_part; ++i) {
        //CUDA_CHECK(cudaEventDestroy(resource.events[i]));
        //CUDA_CHECK(cudaStreamDestroy(resource.local_streams[i]));
    }*/
    //delete [] resource.events;
    //delete [] resource.local_streams;

    allocator.free(resource.d_idx);
    allocator.free(resource.d_keys_parted);
    allocator.free(resource.d_vals_parted);
    allocator.free(resource.d_status_parted);
    delete [] resource.h_part_sizes;
    delete [] resource.h_offsets;
}



template<typename KeyType, typename ValType, typename Allocator, typename KeyGPUMapPolicy_>
void create_resource_and_do_partition(const KeyType* d_keys,
                                      Resource<KeyType, ValType>& resource,
                                      size_t len,
                                      const std::vector<int>& gpu_id,
                                      cudaStream_t stream,
                                      Allocator& allocator,
                                      KeyGPUMapPolicy_& policy) {
    const int num_gpu = gpu_id.size();
    create_local_resource(resource,
                          len,
                          num_gpu,
                          allocator);


    size_t* d_part_sizes;
    allocator.malloc((void**) &d_part_sizes, num_gpu * sizeof(*d_part_sizes));
    multisplit(d_keys,
               resource.d_idx,
               len,
               d_part_sizes,
               num_gpu,
               stream,
               allocator,
               policy);
    CUDA_CHECK(cudaStreamSynchronize(stream));


    size_t* h_part_sizes = resource.h_part_sizes;
    size_t* h_offsets = resource.h_offsets;
    CUDA_CHECK(cudaMemcpy(h_part_sizes,
                          d_part_sizes,
                          num_gpu * sizeof(*h_part_sizes),
                          cudaMemcpyDeviceToHost));
    allocator.free(d_part_sizes);

    memcpy(h_offsets + 1, h_part_sizes, num_gpu * sizeof(*h_offsets));
    h_offsets[0] = 0;
    for (int i = 1; i < num_gpu + 1; ++i) {
        h_offsets[i] += h_offsets[i-1];
    }


    create_remote_resource(resource,
                           gpu_id,
                           allocator);
}





template<typename T>
void send(const T* d_local,
          T** d_remote,
          const size_t* offsets,
          const size_t* part_sizes,
          cudaStream_t* local_streams,
          const std::vector<int>& gpu_id) {
    int dev_local = get_dev(d_local);
    assert(dev_local >= 0);

    const int num_gpu = gpu_id.size();
    for (int i = 0; i < num_gpu; ++i) {
        if (part_sizes[i] == 0) {
            continue;
        }

        CUDA_CHECK(cudaMemcpyPeerAsync(d_remote[i],
                                       gpu_id[i],
                                       &d_local[offsets[i]],
                                       dev_local,
                                       part_sizes[i] * sizeof(T),
                                       local_streams[i]
                       ));
    }
}


template<typename T>
void receive(T* d_local,
             const T* const * d_remote,
             const size_t* offsets,
             const size_t* part_sizes,
             cudaStream_t* remote_streams,
             const std::vector<int>& gpu_id) {
    CudaDeviceRestorer dev_restorer;
    int dev_local = get_dev(d_local);
    assert(dev_local >= 0);

    const int num_gpu = gpu_id.size();
    for (int i = 0; i < num_gpu; ++i) {
        if (part_sizes[i] == 0) {
            continue;
        }

        CUDA_CHECK(cudaSetDevice(gpu_id[i]));
        CUDA_CHECK(cudaMemcpyPeerAsync(&d_local[offsets[i]],
                                       dev_local,
                                       d_remote[i],
                                       gpu_id[i],
                                       part_sizes[i] * sizeof(T),
                                       remote_streams[i]
                       ));
    }
}



void remote_wait_local_streams(cudaEvent_t* events,
                               cudaStream_t* local_streams,
                               cudaStream_t* remote_streams,
                               int num_gpu) {
    for (int i = 0; i < num_gpu; ++i) {
        CUDA_CHECK(cudaEventRecord(events[i], local_streams[i]));
        CUDA_CHECK(cudaStreamWaitEvent(remote_streams[i], events[i], 0));
    }
}

void sync_remotes(cudaStream_t* remote_streams, const std::vector<int>& gpu_id) {
    // CudaDeviceRestorer dev_restorer;

    const int num_gpu = gpu_id.size();
    for (int i = 0; i < num_gpu; ++i) {
        // CUDA_CHECK(cudaSetDevice(gpu_id[i]));
        CUDA_CHECK(cudaStreamSynchronize(remote_streams[i]));
    }
}



template<typename KeyType,
         typename ValType,
         typename KeyGPUMapPolicy_,
         KeyType empty_key = std::numeric_limits<KeyType>::max()>
class MultiGpuHashTable {
public:
    MultiGpuHashTable(size_t capacity, const int* gpu_id, int gpu_id_len);
    ~MultiGpuHashTable();
    MultiGpuHashTable(const MultiGpuHashTable&) = delete;
    MultiGpuHashTable& operator=(const MultiGpuHashTable&) = delete;


    void insert(const KeyType* d_keys, const ValType* d_vals, size_t len, Stream_event_resource& s_e_resource) {
        insert_or_set_helper_(d_keys, d_vals, len, &Table_::insert, s_e_resource);
    }

    void set(const KeyType* d_keys, const ValType* d_vals, size_t len, Stream_event_resource& s_e_resource) {
        insert_or_set_helper_(d_keys, d_vals, len, &Table_::set, s_e_resource);
    }

    void get(const KeyType* d_keys, ValType* d_vals, bool* d_status, size_t len, Stream_event_resource& s_e_resource) const;

    void insert_from_cpu(int gpu_id, const KeyType* h_key, 
        const ValType* h_val, size_t len, const size_t buffer_len,
        const int buffer_count=2);

    size_t get_size(int gpu_id) const;

    void dump_to_cpu(int gpu_id, KeyType* h_key, 
                    ValType* h_val, size_t len, const size_t buffer_len, 
                    const int buffer_count=2) const;

    void dump_to_gpu(int gpu_id, KeyType* d_key,
                    ValType* d_val, size_t len) const;

    //void accum(const KeyType* d_keys, const ValType* d_vals, size_t len, Stream_event_resource& s_e_resource);

    Stream_event_resource stream_event_resource_create(int gpu_id) const;

    void stream_event_resource_destroy(Stream_event_resource& resource) const;

    template<typename GradType, typename Optimizer>
    void update(const KeyType* d_keys, const GradType* d_gradient, size_t len, Optimizer& op, Stream_event_resource& s_e_resource);

    void clear(Stream_event_resource& s_e_resource);

    void remove(const KeyType* d_keys, size_t len, Stream_event_resource& s_e_resource);


private:
    static const int BLOCK_SIZE_ = 256;

    using Table_ = nv::HashTable<KeyType, ValType, empty_key>;
    using TableFunction_ = void (Table_::*)(const KeyType* d_keys, const ValType* d_vals, size_t len, cudaStream_t stream);

    void insert_or_set_helper_(const KeyType* d_keys, const ValType* d_vals, size_t len, TableFunction_ func, Stream_event_resource& s_e_resource);


    const int num_gpu_;
    std::vector<int> gpu_id_;
    std::vector<Table_*> tables_;
    mutable nv::CubAllocator allocator_;
    KeyGPUMapPolicy_ KeyGPUMapPolicy;
    //The Global Hashtable lock for update APIs
    std::mutex update_mtx_;
};


template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
Stream_event_resource MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::stream_event_resource_create(int gpu_id) const{

    /* Check for any invalid input */
    assert(gpu_id >= 0);

    /* We do not check whether the GPU is within the GPU list, user may want to call get/set/insert/accum on non-hashtable GPU */
    /* User need to be caution to make sure the GPU ID is the GPU ID he want */

    /* Save Current device */
    CudaDeviceRestorer dev_restorer;

    /* Set up GPU */
    CUDA_CHECK(cudaSetDevice(gpu_id));

    /* The resource that will be returned */
    Stream_event_resource resource;

    /* How many streams and event we need to create */
    assert(gpu_id_.size() == num_gpu_);
    const int num_gpu = gpu_id_.size();

    /* Create local streams */
    resource.local_streams = new cudaStream_t[num_gpu];
    for (int i = 0; i < num_gpu; ++i) {
        CUDA_CHECK(cudaStreamCreate(&resource.local_streams[i]));
    }

    /* Create remote streams */
    resource.remote_streams = new cudaStream_t[num_gpu];
    for (int i = 0; i < num_gpu; ++i) {
        CUDA_CHECK(cudaSetDevice(gpu_id_[i]));
        CUDA_CHECK(cudaStreamCreate(&resource.remote_streams[i]));
    }

    /* Reset current device */
    CUDA_CHECK(cudaSetDevice(gpu_id));

    /* Create caller stream */
    resource.caller_stream = new cudaStream_t[1];
    CUDA_CHECK(cudaStreamCreate(&resource.caller_stream[0]));

    /* Create local event */
    resource.local_events = new cudaEvent_t[num_gpu];
    for (int i = 0; i < num_gpu; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&resource.local_events[i], cudaEventDisableTiming));
    }

    /* Create remote event */
    resource.remote_events = new cudaEvent_t[num_gpu];
    for (int i = 0; i < num_gpu; ++i) {
        CUDA_CHECK(cudaSetDevice(gpu_id_[i]));
        CUDA_CHECK(cudaEventCreateWithFlags(&resource.remote_events[i], cudaEventDisableTiming));
    }

    return resource; 
}

template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
void MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::stream_event_resource_destroy(Stream_event_resource& resource) const{

    /* How many streams and event we need to Destroy */
    const int num_gpu = gpu_id_.size();

    /* Destroy local streams */
    for (int i = 0; i < num_gpu; ++i) {
        CUDA_CHECK(cudaStreamDestroy(resource.local_streams[i]));
    }
    delete [] resource.local_streams;

    /* Destroy remote streams */
    for (int i = 0; i < num_gpu; ++i) {
        CUDA_CHECK(cudaStreamDestroy(resource.remote_streams[i]));
    }
    delete [] resource.remote_streams;

    /* Destroy caller stream */
    CUDA_CHECK(cudaStreamDestroy(resource.caller_stream[0]));
    delete [] resource.caller_stream;

    /* Destroy local event */
    for (int i = 0; i < num_gpu; ++i) {
        CUDA_CHECK(cudaEventDestroy(resource.local_events[i]));
    }
    delete [] resource.local_events;

    /* Destroy remote event */
    for (int i = 0; i < num_gpu; ++i) {
        CUDA_CHECK(cudaEventDestroy(resource.remote_events[i]));
    }
    delete [] resource.remote_events;
}


template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
void MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::insert_from_cpu(int gpu_id, const KeyType* h_key, 
                                                                    const ValType* h_val, size_t len, const size_t buffer_len,
                                                                    const int buffer_count){
    /* Check for any invalid input*/
    if(len <= 0){
        return;
    }
    assert(buffer_count >= 1);
    assert(gpu_id >= 0);
    assert(buffer_len >= 1);

    /* Calculate which table to use(i.e. The index of "gpu_id" in gpu_id_ vector) */
    std::vector <int>::iterator iElement = std::find(gpu_id_.begin(),gpu_id_.end(),gpu_id);
    assert(iElement != gpu_id_.end());
    int table_index = std::distance(gpu_id_.begin(),iElement);
    

    /* Save Current device */
    CudaDeviceRestorer dev_restorer;

    /* Set up GPU and allocate resources*/
    CUDA_CHECK(cudaSetDevice(gpu_id));
    cudaStream_t streams[buffer_count];
    for(int i=0 ; i < buffer_count; i++){
        CUDA_CHECK(cudaStreamCreate(&(streams[i])));
    }
    KeyType* d_temp_key[buffer_count];
    ValType* d_temp_val[buffer_count];
    for(int i=0 ; i < buffer_count ; i++){
        CUDA_CHECK(cudaMalloc( (void**) &(d_temp_key[i]), sizeof(*h_key) * buffer_len));
        CUDA_CHECK(cudaMalloc( (void**) &(d_temp_val[i]), sizeof(*h_val) * buffer_len)); 
    }

    /* Counters recording how much we have done*/
    size_t len_counter=0;
    int pipeline_counter=0;
    
    /* Assign tasks to different pipeline, until all <K,V> are inserted */
    while(len_counter < len){
        int current_stream = pipeline_counter % buffer_count;

        int copy_len = (len_counter+buffer_len > len ? len-len_counter : buffer_len);

        CUDA_CHECK(cudaMemcpyAsync(d_temp_key[current_stream], h_key + len_counter, sizeof(*h_key) * copy_len ,cudaMemcpyHostToDevice , streams[current_stream]));
        CUDA_CHECK(cudaMemcpyAsync(d_temp_val[current_stream], h_val + len_counter, sizeof(*h_val) * copy_len ,cudaMemcpyHostToDevice , streams[current_stream]));
        tables_[table_index]->insert(d_temp_key[current_stream], d_temp_val[current_stream], copy_len, streams[current_stream]);

        pipeline_counter++;
        len_counter+=copy_len;
    }

    /* Waiting on all tasks to finish*/
    for(int i=0 ; i < buffer_count; i++){
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    /* Finished tasks and clean up*/
    for(int i=0 ; i < buffer_count; i++){
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    for(int i=0 ; i < buffer_count ; i++){
        CUDA_CHECK(cudaFree(d_temp_key[i]));
        CUDA_CHECK(cudaFree(d_temp_val[i])); 
    }
}

template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
size_t MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::get_size(int gpu_id) const{
    /* Check for any invalid input */
    assert(gpu_id >= 0);

    /* Calculate which table to use(i.e. The index of "gpu_id" in gpu_id_ vector) */
    std::vector <int>::const_iterator iElement = std::find(gpu_id_.begin(),gpu_id_.end(),gpu_id);
    assert(iElement != gpu_id_.end());
    int table_index = std::distance(gpu_id_.begin(),iElement);

    /* Save Current device */
    CudaDeviceRestorer dev_restorer;

    /* Set up GPU and allocate resources*/
    CUDA_CHECK(cudaSetDevice(gpu_id));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* Caculate the actual size of the hash table on this GPU */
    size_t hash_table_size;
    hash_table_size = tables_[table_index]-> get_size(stream);

    /* Finished tasks and clean up */
    CUDA_CHECK(cudaStreamDestroy(stream));

    return hash_table_size;

}

template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
void MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::dump_to_cpu(int gpu_id, KeyType* h_key, 
                                                                ValType* h_val, size_t len, const size_t buffer_len, 
                                                                const int buffer_count) const{
    /* Check for any invalid input*/
    if(len <= 0){
        return;
    }
    assert(buffer_count >= 1);
    assert(gpu_id >= 0);
    assert(buffer_len >= 1);

    /* Calculate which table to use(i.e. The index of "gpu_id" in gpu_id_ vector) */
    std::vector <int>::const_iterator iElement = std::find(gpu_id_.begin(),gpu_id_.end(),gpu_id);
    assert(iElement != gpu_id_.end());
    int table_index = std::distance(gpu_id_.begin(),iElement);

    /* Save Current device */
    CudaDeviceRestorer dev_restorer;

    /* Set up GPU and allocate resources*/
    CUDA_CHECK(cudaSetDevice(gpu_id));
    cudaStream_t streams[buffer_count];
    cudaEvent_t Events[buffer_count];
    size_t * d_dump_counter[buffer_count];
    size_t * h_dump_counter;
    size_t h_ptr = 0;
    for(int i=0 ; i < buffer_count; i++){
        CUDA_CHECK(cudaStreamCreate(&(streams[i])));
    }
    for(int i=0 ; i < buffer_count; i++){
        CUDA_CHECK(cudaEventCreate(&(Events[i])));
    }
    
    h_dump_counter = (size_t *) malloc(sizeof(size_t) * buffer_count);

    KeyType* d_temp_key[buffer_count];
    ValType* d_temp_val[buffer_count];
    for(int i=0 ; i < buffer_count ; i++){
        CUDA_CHECK(cudaMalloc( (void**) &(d_temp_key[i]), sizeof(*h_key) * buffer_len));
        CUDA_CHECK(cudaMalloc( (void**) &(d_temp_val[i]), sizeof(*h_val) * buffer_len)); 
        CUDA_CHECK(cudaMalloc( (void**) &(d_dump_counter[i]), sizeof(size_t)));
    }

    /* Counters recording how much we have done*/
    const size_t table_capacity = tables_[table_index] -> get_capacity(); // The Actual capacity of hashtable on gpu_id.
    size_t len_counter = 0; // How much of the hash table we have processed
    

    /* Assign tasks to different pipeline, until all of the hashtable is processed */
    while(len_counter < table_capacity){
        size_t valid_stream = 0;
        int pipeline_counter = 0;

        while(len_counter < table_capacity && pipeline_counter < buffer_count){
            int current_stream = pipeline_counter % buffer_count;

            int search_length = (len_counter + buffer_len > table_capacity ? table_capacity-len_counter : buffer_len);

            tables_[table_index]->dump(d_temp_key[current_stream], d_temp_val[current_stream], 
                                    len_counter, search_length, d_dump_counter[current_stream], streams[current_stream]);

            CUDA_CHECK(cudaMemcpyAsync(&(h_dump_counter[current_stream]), d_dump_counter[current_stream], 
                                        sizeof(size_t),  cudaMemcpyDeviceToHost, streams[current_stream]));

            CUDA_CHECK(cudaEventRecord( Events[current_stream], streams[current_stream]));

            len_counter += search_length;

            pipeline_counter++ ;
            valid_stream++ ;
        }

        pipeline_counter = 0;

        while(valid_stream > 0){
            int current_stream = pipeline_counter % buffer_count;

            CUDA_CHECK(cudaEventSynchronize(Events[current_stream]));

            CUDA_CHECK(cudaMemcpyAsync(h_key + h_ptr, d_temp_key[current_stream], sizeof(*h_key) * h_dump_counter[current_stream], 
                                        cudaMemcpyDeviceToHost, streams[current_stream]));
            CUDA_CHECK(cudaMemcpyAsync(h_val + h_ptr, d_temp_val[current_stream], sizeof(*h_val) * h_dump_counter[current_stream], 
                                        cudaMemcpyDeviceToHost, streams[current_stream]));
            h_ptr += h_dump_counter[current_stream];

            valid_stream-- ;
            pipeline_counter++ ;
        }

    }

    // Double check with get_size output
    assert(h_ptr == len);

    /* Waiting on all tasks to finish*/
    for(int i=0 ; i < buffer_count; i++){
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    /* Finished tasks and clean up*/
    for(int i=0 ; i < buffer_count; i++){
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }
    for(int i=0 ; i < buffer_count; i++){
        CUDA_CHECK(cudaEventDestroy(Events[i]));
    }
    for(int i=0 ; i < buffer_count ; i++){
        CUDA_CHECK(cudaFree(d_temp_key[i]));
        CUDA_CHECK(cudaFree(d_temp_val[i])); 
        CUDA_CHECK(cudaFree(d_dump_counter[i]));
    }
    
    free((void*) h_dump_counter);

}

template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
void MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::dump_to_gpu(int gpu_id, 
                                                                                    KeyType* d_key, 
                                                                                    ValType* d_val, 
                                                                                    size_t len) const{
    /* Check for any invalid input*/
    if(len <= 0){
        return;
    }
    assert(gpu_id >= 0);
    
    /* Calculate which table to use(i.e. The index of "gpu_id" in gpu_id_ vector) */
    std::vector <int>::const_iterator iElement = std::find(gpu_id_.begin(),gpu_id_.end(),gpu_id);
    assert(iElement != gpu_id_.end());
    int table_index = std::distance(gpu_id_.begin(),iElement);

    /* Make Sure d_key and d_val buffer are on the same device */
    assert(get_dev(d_key) == get_dev(d_val));

    /* Make Sure GPU buffer provided by the user is on the same device as requested GPU or hashtable */
    assert(get_dev(d_key) == gpu_id);

    /* Save Current device */
    CudaDeviceRestorer dev_restorer;

    /* Set to GPU and allocate resource */
    CUDA_CHECK(cudaSetDevice(gpu_id));
    cudaStream_t stream;
    size_t * d_dump_counter;
    size_t   h_dump_counter;

    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMalloc( (void**) &(d_dump_counter), sizeof(size_t)));

    // The Actual capacity of hashtable on gpu_id, NOT the size.
    const size_t table_capacity = tables_[table_index] -> get_capacity(); 

    /* Dump the hashtable on the required GPU to the buffer provided */
    tables_[table_index]->dump(d_key, d_val, 0, table_capacity, d_dump_counter, stream);

    CUDA_CHECK(cudaMemcpyAsync(&h_dump_counter, d_dump_counter, 
            sizeof(size_t),  cudaMemcpyDeviceToHost, stream));

    /* Waiting on all tasks to finish*/
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Double check with get_size output
    assert(h_dump_counter == len);

    /* Finished tasks and clean up*/
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_dump_counter));

}

template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::MultiGpuHashTable(size_t capacity, const int* gpu_id, int gpu_id_len)
    : num_gpu_(gpu_id_len), gpu_id_(gpu_id, gpu_id + gpu_id_len), allocator_(gpu_id, gpu_id_len) {
    assert(gpu_id_len > 0);
    for (auto gpu : gpu_id_) {
        assert(gpu >= 0);
    }
    assert(num_gpu_ == gpu_id_.size());


    CudaDeviceRestorer dev_restorer;
    for (int i = 0; i < gpu_id_len; ++i) {
        CUDA_CHECK(cudaSetDevice(gpu_id[i]));
        auto table = new Table_(capacity / gpu_id_len + 1);
        tables_.push_back(table);
    }

    for (int cur = 0; cur < gpu_id_len; ++cur) {
        CUDA_CHECK(cudaSetDevice(gpu_id[cur]));
        for (int peer = 0; peer < gpu_id_len; ++peer) {
            if (cur == peer) {
                continue;
            }
            int can_access;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, gpu_id[cur], gpu_id[peer]));
            if (can_access) {
                cudaError_t ret = cudaDeviceEnablePeerAccess(gpu_id[peer], 0);
                if (ret == cudaErrorPeerAccessAlreadyEnabled
                    && cudaPeekAtLastError() == cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError();
                }
                else {
                    CUDA_CHECK(ret);
                }
            }
        }
    }

}


template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::~MultiGpuHashTable() {
    for (auto& table : tables_) {
        delete table;
        table = nullptr;
    }
}


template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
void MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::insert_or_set_helper_(const KeyType* d_keys,
                                                                const ValType* d_vals,
                                                                size_t len,
                                                                TableFunction_ func,
                                                                Stream_event_resource& s_e_resource) {
    if (len == 0) {
        return;
    }

    CudaDeviceRestorer dev_restorer;
    assert(get_dev(d_keys) == get_dev(d_vals));
    switch_to_dev(d_keys);

    //cudaStream_t caller_stream;
    //CUDA_CHECK(cudaStreamCreate(&caller_stream));

    Resource<KeyType, ValType> resource;
    create_resource_and_do_partition(d_keys,
                                     resource,
                                     len,
                                     gpu_id_,
                                     s_e_resource.caller_stream[0],
                                     allocator_,
                                     KeyGPUMapPolicy);

    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    gather<<<grid_size, BLOCK_SIZE_, 0, s_e_resource.caller_stream[0]>>>(resource.d_idx,
                                                         d_keys,
                                                         d_vals,
                                                         resource.d_keys_parted,
                                                         resource.d_vals_parted,
                                                         len);
    CUDA_CHECK(cudaStreamSynchronize(s_e_resource.caller_stream[0]));


    send(resource.d_keys_parted,
         resource.d_remote_keys,
         resource.h_offsets,
         resource.h_part_sizes,
         s_e_resource.local_streams,
         gpu_id_);
    send(resource.d_vals_parted,
         resource.d_remote_vals,
         resource.h_offsets,
         resource.h_part_sizes,
         s_e_resource.local_streams,
         gpu_id_);
    remote_wait_local_streams(s_e_resource.local_events,
                              s_e_resource.local_streams,
                              s_e_resource.remote_streams,
                              num_gpu_);

    {
        CudaDeviceRestorer dev_restorer;
        for (int i = 0; i < num_gpu_; ++i) {
            CUDA_CHECK(cudaSetDevice(gpu_id_[i]));
            (tables_[i]->*func)(resource.d_remote_keys[i],
                                resource.d_remote_vals[i],
                                resource.h_part_sizes[i],
                                s_e_resource.remote_streams[i]);
        }
    }
    sync_remotes(s_e_resource.remote_streams, gpu_id_);

    destroy_resource(resource, gpu_id_, allocator_);
    //CUDA_CHECK(cudaStreamDestroy(caller_stream));
}


template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
void MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::get(const KeyType* d_keys, ValType* d_vals, bool* d_status,
                                                                           size_t len, Stream_event_resource& s_e_resource) const {
    if (len == 0) {
        return;
    }

    CudaDeviceRestorer dev_restorer;
    assert(get_dev(d_keys) == get_dev(d_vals));
    assert(get_dev(d_keys) == get_dev(d_status));
    switch_to_dev(d_keys);

    //cudaStream_t caller_stream;
    //CUDA_CHECK(cudaStreamCreate(&caller_stream));

    Resource<KeyType, ValType> resource;
    create_resource_and_do_partition(d_keys,
                                     resource,
                                     len,
                                     gpu_id_,
                                     s_e_resource.caller_stream[0],
                                     allocator_,
                                     KeyGPUMapPolicy);

    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    gather<<<grid_size, BLOCK_SIZE_, 0, s_e_resource.caller_stream[0]>>>(resource.d_idx,
                                                         d_keys,
                                                         resource.d_keys_parted,
                                                         len);
    CUDA_CHECK(cudaStreamSynchronize(s_e_resource.caller_stream[0]));


    send(resource.d_keys_parted,
         resource.d_remote_keys,
         resource.h_offsets,
         resource.h_part_sizes,
         s_e_resource.local_streams,
         gpu_id_);
    remote_wait_local_streams(s_e_resource.local_events,
                              s_e_resource.local_streams,
                              s_e_resource.remote_streams,
                              num_gpu_);


    {
        CudaDeviceRestorer dev_restorer;
        for (int i = 0; i < num_gpu_; ++i) {
            while(!(tables_[i]->get_lock())){
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }

            CUDA_CHECK(cudaSetDevice(gpu_id_[i]));
            tables_[i]->get(resource.d_remote_keys[i],
                            resource.d_remote_vals[i],
                            resource.d_remote_status[i],
                            resource.h_part_sizes[i],
                            s_e_resource.remote_streams[i]);
        }
    }

    //sync_remotes(s_e_resource.remote_streams, gpu_id_);
    for (int i = 0; i < num_gpu_; ++i) {
        // CUDA_CHECK(cudaSetDevice(gpu_id[i]));
        CUDA_CHECK(cudaStreamSynchronize(s_e_resource.remote_streams[i]));
        tables_[i]->get_release();
    }

    receive(resource.d_vals_parted,
            resource.d_remote_vals,
            resource.h_offsets,
            resource.h_part_sizes,
            s_e_resource.remote_streams,
            gpu_id_);

    // Also receive status
    receive(resource.d_status_parted,
            resource.d_remote_status,
            resource.h_offsets,
            resource.h_part_sizes,
            s_e_resource.remote_streams,
            gpu_id_);

    //cudaEvent_t* remote_events = new cudaEvent_t[num_gpu_];
    {
        CudaDeviceRestorer dev_restorer;
        for (int i = 0; i < num_gpu_; ++i) {
            CUDA_CHECK(cudaSetDevice(gpu_id_[i]));
            //CUDA_CHECK(cudaEventCreateWithFlags(&remote_events[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(s_e_resource.remote_events[i], s_e_resource.remote_streams[i]));
        }
    }

    for (int i = 0; i < num_gpu_; ++i) {
        CUDA_CHECK(cudaStreamWaitEvent(s_e_resource.local_streams[i], s_e_resource.remote_events[i], 0));
        if(resource.h_part_sizes[i] <= 0){ // Important! size_t is uint_64, can't be 0 !
            continue;
        }

        int tmp_grid_size = (resource.h_part_sizes[i] - 1) / BLOCK_SIZE_ + 1;
        scatter<<<tmp_grid_size, BLOCK_SIZE_, 0, s_e_resource.local_streams[i]>>>(
            resource.d_idx,
            resource.d_vals_parted,
            d_vals,
            resource.h_offsets[i],
            resource.h_part_sizes[i]);

        //Also scatter the status
        scatter<<<tmp_grid_size, BLOCK_SIZE_, 0, s_e_resource.local_streams[i]>>>(
            resource.d_idx,
            resource.d_status_parted,
            d_status,
            resource.h_offsets[i],
            resource.h_part_sizes[i]);
    }

    for (int i = 0; i < num_gpu_; ++i) {
        CUDA_CHECK(cudaStreamSynchronize(s_e_resource.local_streams[i]));
    }


    /*for (int i = 0; i < num_gpu_; ++i) {
        CUDA_CHECK(cudaEventDestroy(remote_events[i]));
    }*/
    //delete [] remote_events;
    //CUDA_CHECK(cudaStreamDestroy(caller_stream));

    destroy_resource(resource, gpu_id_, allocator_);
}

/*template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
void MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::accum(const KeyType* d_keys, const ValType* d_vals, 
                                                                             size_t len, Stream_event_resource& s_e_resource){
    if (len == 0) {
        return;
    }

    CudaDeviceRestorer dev_restorer;
    assert(get_dev(d_keys) == get_dev(d_vals));
    switch_to_dev(d_keys);

    //cudaStream_t caller_stream;
    //CUDA_CHECK(cudaStreamCreate(&caller_stream));

    Resource<KeyType, ValType> resource;
    create_resource_and_do_partition(d_keys,
                                     resource,
                                     len,
                                     gpu_id_,
                                     s_e_resource.caller_stream[0],
                                     allocator_,
                                     KeyGPUMapPolicy);

    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    gather<<<grid_size, BLOCK_SIZE_, 0, s_e_resource.caller_stream[0]>>>(resource.d_idx,
                                                         d_keys,
                                                         d_vals,
                                                         resource.d_keys_parted,
                                                         resource.d_vals_parted,
                                                         len);
    CUDA_CHECK(cudaStreamSynchronize(s_e_resource.caller_stream[0]));


    send(resource.d_keys_parted,
         resource.d_remote_keys,
         resource.h_offsets,
         resource.h_part_sizes,
         s_e_resource.local_streams,
         gpu_id_);
    send(resource.d_vals_parted,
         resource.d_remote_vals,
         resource.h_offsets,
         resource.h_part_sizes,
         s_e_resource.local_streams,
         gpu_id_);
    remote_wait_local_streams(s_e_resource.local_events,
                              s_e_resource.local_streams,
                              s_e_resource.remote_streams,
                              num_gpu_);

    {
        CudaDeviceRestorer dev_restorer;
        for (int i = 0; i < num_gpu_; ++i) {
            CUDA_CHECK(cudaSetDevice(gpu_id_[i]));
            tables_[i]->accum(resource.d_remote_keys[i],
                              resource.d_remote_vals[i],
                              resource.h_part_sizes[i],
                              s_e_resource.remote_streams[i]);
        }
    }

    sync_remotes(s_e_resource.remote_streams, gpu_id_);

    destroy_resource(resource, gpu_id_, allocator_);
    //CUDA_CHECK(cudaStreamDestroy(caller_stream));
}*/

template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
template<typename GradType, typename Optimizer>
void MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::update(const KeyType* d_keys, const GradType* d_gradient, 
                                                                              size_t len, Optimizer& op, 
                                                                              Stream_event_resource& s_e_resource){
    if (len == 0) {
        return;
    }

    CudaDeviceRestorer dev_restorer;
    assert(get_dev(d_keys) == get_dev(d_gradient));
    switch_to_dev(d_keys);

    //Here, we need to lock the global update lock!
    update_mtx_.lock();

    //cudaStream_t caller_stream;
    //CUDA_CHECK(cudaStreamCreate(&caller_stream));

    Resource<KeyType, GradType> resource;
    create_resource_and_do_partition(d_keys,
                                     resource,
                                     len,
                                     gpu_id_,
                                     s_e_resource.caller_stream[0],
                                     allocator_,
                                     KeyGPUMapPolicy);

    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    gather<<<grid_size, BLOCK_SIZE_, 0, s_e_resource.caller_stream[0]>>>(resource.d_idx,
                                                         d_keys,
                                                         d_gradient,
                                                         resource.d_keys_parted,
                                                         resource.d_vals_parted,
                                                         len);
    CUDA_CHECK(cudaStreamSynchronize(s_e_resource.caller_stream[0]));


    send(resource.d_keys_parted,
         resource.d_remote_keys,
         resource.h_offsets,
         resource.h_part_sizes,
         s_e_resource.local_streams,
         gpu_id_);
    send(resource.d_vals_parted,
         resource.d_remote_vals,
         resource.h_offsets,
         resource.h_part_sizes,
         s_e_resource.local_streams,
         gpu_id_);
    remote_wait_local_streams(s_e_resource.local_events,
                              s_e_resource.local_streams,
                              s_e_resource.remote_streams,
                              num_gpu_);

    {
        CudaDeviceRestorer dev_restorer;
        for (int i = 0; i < num_gpu_; ++i) {
            while(!(tables_[i]->update_lock())){
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }

            CUDA_CHECK(cudaSetDevice(gpu_id_[i]));
            tables_[i]->update(resource.d_remote_keys[i],
                               resource.d_remote_vals[i],
                               resource.h_part_sizes[i],
                               s_e_resource.remote_streams[i],
                               op);
        }
    }

    //sync_remotes(s_e_resource.remote_streams, gpu_id_);
    for (int i = 0; i < num_gpu_; ++i) {
        // CUDA_CHECK(cudaSetDevice(gpu_id[i]));
        CUDA_CHECK(cudaStreamSynchronize(s_e_resource.remote_streams[i]));
        tables_[i]->update_release();
    }

    // Here, we need to release global update lock!
    update_mtx_.unlock();

    destroy_resource(resource, gpu_id_, allocator_);
    //CUDA_CHECK(cudaStreamDestroy(caller_stream));
}

template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
void MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::clear(Stream_event_resource& s_e_resource){
    
    // Recover the device setting 
    CudaDeviceRestorer dev_restorer;

    // Clear all hashtables
    for (int i = 0; i < num_gpu_; ++i) {

        CUDA_CHECK(cudaSetDevice(gpu_id_[i]));

        tables_[i]->clear(s_e_resource.remote_streams[i]);

    }

    // Wait for all clear kernel to finish
    for (int i = 0; i < num_gpu_; ++i) {

        CUDA_CHECK(cudaStreamSynchronize(s_e_resource.remote_streams[i]));

    }
}




template<typename KeyType, typename ValType, typename KeyGPUMapPolicy_, KeyType empty_key>
void MultiGpuHashTable<KeyType, ValType, KeyGPUMapPolicy_, empty_key>::remove(const KeyType* d_keys, size_t len, Stream_event_resource& s_e_resource){

    if (len == 0) {
        return;
    }

    CudaDeviceRestorer dev_restorer;
    switch_to_dev(d_keys);

    Resource<KeyType, ValType> resource;
    create_resource_and_do_partition(d_keys,
                                     resource,
                                     len,
                                     gpu_id_,
                                     s_e_resource.caller_stream[0],
                                     allocator_,
                                     KeyGPUMapPolicy);

    const int grid_size = (len - 1) / BLOCK_SIZE_ + 1;
    gather<<<grid_size, BLOCK_SIZE_, 0, s_e_resource.caller_stream[0]>>>(resource.d_idx,
                                                         d_keys,
                                                         resource.d_keys_parted,
                                                         len);
    CUDA_CHECK(cudaStreamSynchronize(s_e_resource.caller_stream[0]));


    send(resource.d_keys_parted,
         resource.d_remote_keys,
         resource.h_offsets,
         resource.h_part_sizes,
         s_e_resource.local_streams,
         gpu_id_);
    remote_wait_local_streams(s_e_resource.local_events,
                              s_e_resource.local_streams,
                              s_e_resource.remote_streams,
                              num_gpu_);

    // Remove keys on each GPU
    {
        CudaDeviceRestorer dev_restorer;
        for (int i = 0; i < num_gpu_; ++i) {

            CUDA_CHECK(cudaSetDevice(gpu_id_[i]));
            tables_[i]->remove(resource.d_remote_keys[i],
                               resource.h_part_sizes[i],
                               s_e_resource.remote_streams[i]);
        }
    }

    // Wait for all GPU delete kernels to finish
    sync_remotes(s_e_resource.remote_streams, gpu_id_);

    // Destroy resources
    destroy_resource(resource, gpu_id_, allocator_);


}


}
#endif

