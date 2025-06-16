__global__ void stochastic_kernel(const float *prices, float *stoch_k, float *stoch_d, int len, int window) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= window && i < len) {
        float min_val = prices[i - window];
        float max_val = prices[i - window];

        for (int j = i - window + 1; j <= i; j++) {
            if (prices[j] < min_val) min_val = prices[j];
            if (prices[j] > max_val) max_val = prices[j];
        }

        if (max_val != min_val)
            stoch_k[i] = ((prices[i] - min_val) / (max_val - min_val)) * 100.0f;
        else
            stoch_k[i] = 0.0f;

        // %D = média móvel simples de 3 períodos de %K
        if (i >= window + 2) {
            stoch_d[i] = (stoch_k[i] + stoch_k[i - 1] + stoch_k[i - 2]) / 3.0f;
        } else {
            stoch_d[i] = 0.0f;
        }
    }
}

extern "C" void run_cuda_stochastic(const float *prices, float *stoch_k, float *stoch_d, int len, int window) {
    float *d_prices, *d_k, *d_d;
    cudaMalloc(&d_prices, len * sizeof(float));
    cudaMalloc(&d_k, len * sizeof(float));
    cudaMalloc(&d_d, len * sizeof(float));

    cudaMemcpy(d_prices, prices, len * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (len + blockSize - 1) / blockSize;
    stochastic_kernel<<<numBlocks, blockSize>>>(d_prices, d_k, d_d, len, window);

    cudaMemcpy(stoch_k, d_k, len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(stoch_d, d_d, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_prices);
    cudaFree(d_k);
    cudaFree(d_d);
}