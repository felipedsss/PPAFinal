#include <cuda_runtime.h>
#include <stdio.h>

__global__ void rsi_kernel(float* input, float* output, int len, int window) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < len - window) {
        float gain = 0.0f, loss = 0.0f;
        for (int i = 0; i < window; i++) {
            float diff = input[idx + i + 1] - input[idx + i];
            if (diff > 0) gain += diff;
            else loss -= diff;  // loss é negativo, por isso inverte
        }

        float avg_gain = gain / window;
        float avg_loss = loss / window;

        if (avg_loss == 0)
            output[idx + window] = 100.0f; // RSI máximo
        else {
            float rs = avg_gain / avg_loss;
            output[idx + window] = 100.0f - (100.0f / (1.0f + rs));
        }
    }
}

extern "C" void run_cuda_rsi(float* input, float* output, int len, int window) {
    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, len * sizeof(float));
    cudaMalloc((void**)&d_output, len * sizeof(float));

    cudaMemcpy(d_input, input, len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;

    rsi_kernel<<<blocks, threads>>>(d_input, d_output, len, window);

    cudaMemcpy(output, d_output, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}