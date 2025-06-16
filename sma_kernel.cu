extern "C" void run_cuda_sma(float* input, float* output, int len, int window);

__global__ void sma_kernel(float* input, float* output, int len, int window) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= window && i < len) {
        float sum = 0.0f;
        for (int j = 0; j < window; j++)
            sum += input[i - j];
        output[i] = sum / window;
    }
}

void run_cuda_sma(float* input, float* output, int len, int window) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, len * sizeof(float));
    cudaMalloc(&d_output, len * sizeof(float));

    cudaMemcpy(d_input, input, len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    sma_kernel<<<blocks, threads>>>(d_input, d_output, len, window);

    cudaMemcpy(output, d_output, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}