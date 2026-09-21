#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>

// 1 million elements
const int DATA_SIZE = 1000000;

// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " (" << cudaGetErrorString(err) << ")\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// CPU method without branching
void cpuNoBranch(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

// CPU method with branching
void cpuBranch(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; ++i) {
        if (i % 2 == 0) {
            c[i] = a[i] + b[i] * 2.0f;
        } else {
            c[i] = a[i] - b[i];
        }
    }
}

// GPU kernel without branching using grid-stride loop
__global__ void gpuNoBranch(const float* a, const float* b, float* c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        c[i] = a[i] + b[i];
    }
}

// GPU kernel with branching causing warp divergence
__global__ void gpuBranch(const float* a, const float* b, float* c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < size; i += stride) {
        if (i % 2 == 0) {
            c[i] = a[i] + b[i] * 2.0f;
        } else {
            c[i] = a[i] - b[i];
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <total_threads> <threads_per_block>\n";
        return EXIT_FAILURE;
    }

    int total_threads = std::atoi(argv[1]);
    int threads_per_block = std::atoi(argv[2]);

    if (total_threads <= 0 || threads_per_block <= 0) {
        std::cerr << "Error: Threads and block size must be greater than 0.\n";
        return EXIT_FAILURE;
    }

    // Calculate block count
    int blocks_per_grid = (total_threads + threads_per_block - 1) / threads_per_block;

    std::cout << "--- Configuration ---\n";
    std::cout << "Total Threads: " << total_threads << std::endl;
    std::cout << "Threads/Block: " << threads_per_block << std::endl;
    std::cout << "Blocks/Grid:   " << blocks_per_grid << std::endl;
    std::cout << "Data Size:     " << DATA_SIZE << " elements\n\n";

    size_t bytes = DATA_SIZE * sizeof(float);
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;

    // Allocate and initialize host memory
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    for (int i = 0; i < DATA_SIZE; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Allocate and copy to device memory
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // CPU without branching
    auto start = std::chrono::high_resolution_clock::now();
    cpuNoBranch(h_a, h_b, h_c, DATA_SIZE);
    auto end = std::chrono::high_resolution_clock::now();
    auto cpu_nb_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "CPU (No Branch) Time: \t" << cpu_nb_time << " us\n";

    // CPU with branching
    start = std::chrono::high_resolution_clock::now();
    cpuBranch(h_a, h_b, h_c, DATA_SIZE);
    end = std::chrono::high_resolution_clock::now();
    auto cpu_b_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "CPU (Branch) Time:    \t" << cpu_b_time << " us\n";

    // GPU without branching
    start = std::chrono::high_resolution_clock::now();
    gpuNoBranch<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, DATA_SIZE);
    // Synchronize before stopping timer to measure true execution time
    CUDA_CHECK(cudaDeviceSynchronize()); 
    end = std::chrono::high_resolution_clock::now();
    auto gpu_nb_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "GPU (No Branch) Time: \t" << gpu_nb_time << " us\n";

    // GPU with branching
    start = std::chrono::high_resolution_clock::now();
    gpuBranch<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_c, DATA_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    auto gpu_b_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "GPU (Branch) Time:    \t" << gpu_b_time << " us\n";

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return EXIT_SUCCESS;
}