#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define RADIUS 2
#define FILTER_SIZE 5

__constant__ float c_f[FILTER_SIZE]; // Constant mem

__global__ void naiveConv(float* in, float* out, float* f, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Register
    
    if (i < n) {
        float sum = 0.0f; // Register
        for (int j = -RADIUS; j <= RADIUS; j++) {
            int offset = i + j;
            if (offset >= 0 && offset < n) {
                sum += in[offset] * f[j + RADIUS]; // Global reads
            }
        }
        out[i] = sum; // Global write
    }
}

__global__ void optConv(float* in, float* out, int n) {
    int tx = threadIdx.x; // Register
    int i = blockIdx.x * blockDim.x + tx;
    extern __shared__ float s_in[]; // Shared mem

    s_in[tx + RADIUS] = (i < n) ? in[i] : 0.0f;

    if (tx < RADIUS) {
        s_in[tx] = (i >= RADIUS) ? in[i - RADIUS] : 0.0f;
        int rIdx = i + blockDim.x;
        s_in[tx + blockDim.x + RADIUS] = (rIdx < n) ? in[rIdx] : 0.0f;
    }
    __syncthreads(); 

    if (i < n) {
        float sum = 0.0f;
        for (int j = -RADIUS; j <= RADIUS; j++) {
            sum += s_in[tx + RADIUS + j] * c_f[j + RADIUS]; // Shared/Const
        }
        out[i] = sum;
    }
}

void parseArgs(int argc, char** argv, int* b, int* t) {
    *b = 16; 
    *t = 256; 

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 < argc) {
            if (strcmp(argv[i], "-b") == 0) *b = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "-t") == 0) *t = atoi(argv[i + 1]);
        }
    }
    if (*t < 64) *t = 64; 
}

float timeNaive(float* d_in, float* d_out, float* d_f, int n, int b, int t) {
    cudaEvent_t start, stop; float ms = 0;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    naiveConv<<<b, t>>>(d_in, d_out, d_f, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

float timeOpt(float* d_in, float* d_out, int n, int b, int t) {
    cudaEvent_t start, stop; float ms = 0;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    int sMem = (t + 2 * RADIUS) * sizeof(float);

    cudaEventRecord(start);
    optConv<<<b, t, sMem>>>(d_in, d_out, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return ms;
}

void run(int b, int t) {
    int n = b * t;
    size_t bytes = n * sizeof(float);
    float h_f[FILTER_SIZE] = {0.1f, 0.2f, 0.4f, 0.2f, 0.1f};

    float *h_in = (float*)malloc(bytes); // Host mem
    for (int i = 0; i < n; i++) h_in[i] = (float)(rand() % 100);

    float *d_in, *d_out_n, *d_out_o, *d_f;
    cudaMalloc(&d_in, bytes); cudaMalloc(&d_out_n, bytes);
    cudaMalloc(&d_out_o, bytes); cudaMalloc(&d_f, FILTER_SIZE * sizeof(float));

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, h_f, FILTER_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_f, h_f, FILTER_SIZE * sizeof(float));

    float tN = timeNaive(d_in, d_out_n, d_f, n, b, t);
    float tO = timeOpt(d_in, d_out_o, n, b, t);

    printf("T/B: %d | Blks: %d\nNaive: %f ms\nOpt: %f ms\n", t, b, tN, tO);

    free(h_in); 
    cudaFree(d_in); cudaFree(d_out_n); cudaFree(d_out_o); cudaFree(d_f);
}

int main(int argc, char** argv) {
    int b = 0, t = 0;
    parseArgs(argc, argv, &b, &t);
    run(b, t);
    return 0;
}