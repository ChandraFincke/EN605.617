#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// Final project context intent is for overlapping H2D/D2H transfers with execution for a CV pipeline.

const int NUM_STREAMS = 4;
const int DATASET_SIZE = 16777216; // 16M elements

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void featureExtractionKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        for (int i = 0; i < 25; i++) {
            val = sinf(val) * cosf(val) + expf(-val) * 0.01f;
        }
        output[idx] = val;
    }
}

__global__ void scalingKernel(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        output[idx] = (val - 0.5f) * 2.0f + sqrtf(fabsf(val));
    }
}

void runPipeline(int totalElements, int threadsPerBlock) {
    int paddedSize = (totalElements / NUM_STREAMS) * NUM_STREAMS; 
    size_t bytes = paddedSize * sizeof(float);
    size_t streamSize = paddedSize / NUM_STREAMS;
    size_t streamBytes = streamSize * sizeof(float);

    // Pinned host memory (Required for async operations)
    float *hostInput, *hostOutputSync, *hostOutputAsync;
    CUDA_CHECK(cudaMallocHost((void**)&hostInput, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&hostOutputSync, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&hostOutputAsync, bytes));

    // Device memory allocations
    float *deviceInput, *deviceMid, *deviceOutput;
    CUDA_CHECK(cudaMalloc((void**)&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc((void**)&deviceMid, bytes));
    CUDA_CHECK(cudaMalloc((void**)&deviceOutput, bytes));

    for (int i = 0; i < paddedSize; i++) {
        hostInput[i] = (float)(rand() % 100) / 100.0f;
    }

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }

    // 1. Synchronous baseline execution
    CUDA_CHECK(cudaEventRecord(startEvent));
    CUDA_CHECK(cudaMemcpy(deviceInput, hostInput, bytes, cudaMemcpyHostToDevice));
    
    int totalBlocks = (paddedSize + threadsPerBlock - 1) / threadsPerBlock;
    featureExtractionKernel<<<totalBlocks, threadsPerBlock>>>(deviceInput, deviceMid, paddedSize);
    scalingKernel<<<totalBlocks, threadsPerBlock>>>(deviceMid, deviceOutput, paddedSize);
    
    CUDA_CHECK(cudaMemcpy(hostOutputSync, deviceOutput, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    
    float syncTimeMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&syncTimeMs, startEvent, stopEvent));

    // 2. Asynchronous pipeline (Streams + Events)
    CUDA_CHECK(cudaEventRecord(startEvent));

    for (int i = 0; i < NUM_STREAMS; ++i) {
        int offset = i * streamSize;
        int blocks = (streamSize + threadsPerBlock - 1) / threadsPerBlock;

        CUDA_CHECK(cudaMemcpyAsync(&deviceInput[offset], &hostInput[offset], streamBytes, cudaMemcpyHostToDevice, streams[i]));
        featureExtractionKernel<<<blocks, threadsPerBlock, 0, streams[i]>>>(&deviceInput[offset], &deviceMid[offset], streamSize);
        scalingKernel<<<blocks, threadsPerBlock, 0, streams[i]>>>(&deviceMid[offset], &deviceOutput[offset], streamSize);
        CUDA_CHECK(cudaMemcpyAsync(&hostOutputAsync[offset], &deviceOutput[offset], streamBytes, cudaMemcpyDeviceToHost, streams[i]));
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent)); 

    float asyncTimeMs = 0;
    CUDA_CHECK(cudaEventElapsedTime(&asyncTimeMs, startEvent, stopEvent));

    // Display results
    printf("\n==========================================\n");
    printf("Threads Per Block  : %d\n", threadsPerBlock);
    printf("Blocks Per Grid    : %d\n", totalBlocks);
    printf("Synchronous Time   : %f ms\n", syncTimeMs);
    printf("Asynchronous Time  : %f ms\n", asyncTimeMs);
    printf("Performance Speedup: %.2fx\n", syncTimeMs / asyncTimeMs);
    printf("==========================================\n");
    fflush(stdout); 

    // Cleanup resources
    for (int i = 0; i < NUM_STREAMS; i++) cudaStreamDestroy(streams[i]);
    CUDA_CHECK(cudaEventDestroy(startEvent)); 
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFreeHost(hostInput)); 
    CUDA_CHECK(cudaFreeHost(hostOutputSync)); 
    CUDA_CHECK(cudaFreeHost(hostOutputAsync));
    CUDA_CHECK(cudaFree(deviceInput)); 
    CUDA_CHECK(cudaFree(deviceMid)); 
    CUDA_CHECK(cudaFree(deviceOutput));
}

int main(int argc, char** argv) {
    int threadsPerBlock = 128; // Default
    
    // Parse command line arguments for the test harness
    for (int i = 1; i < argc; i += 2) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threadsPerBlock = atoi(argv[i+1]);
        }
    }
    
    runPipeline(DATASET_SIZE, threadsPerBlock);
    return 0;
}