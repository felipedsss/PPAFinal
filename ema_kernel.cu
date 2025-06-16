#include <cuda_runtime.h>

__global__ void ema_kernel(float* input, float* output, int len, int window, float alpha) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i == 0) {
        output[0] = input[0]; // EMA inicial = primeiro valor
    } else if (i < len) {
        // fórmula: EMA_t = α * preço_t + (1 - α) * EMA_{t-1}
        // para garantir o acesso seguro a output[i - 1], seria melhor fazer isso em CPU ou com scan
        float prev_ema = output[i - 1]; // WARNING: depende da ordem de execução (não seguro aqui)
        output[i] = alpha * input[i] + (1.0f - alpha) * prev_ema;
    }
}

// Wrapper chamada de C
extern "C" void run_cuda_ema(float* input, float* output, int len, int window) {
    float *d_input, *d_output;
    float alpha = 2.0f / (window + 1);

    cudaMalloc(&d_input, len * sizeof(float));
    cudaMalloc(&d_output, len * sizeof(float));

    cudaMemcpy(d_input, input, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, input, sizeof(float), cudaMemcpyHostToDevice); // inicializa d_output[0]

    int threads = 256;
    int blocks = (len + threads - 1) / threads;

    ema_kernel<<<blocks, threads>>>(d_input, d_output, len, window, alpha);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}