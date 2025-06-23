__kernel void indicadores2d(__global const float *series,
                            __global const int *sizes,
                            const int max_size,
                            const int result_cols,
                            __global float *resultados) {

    int file_id = get_global_id(0);  // cada linha é um arquivo
    int idx = get_global_id(1);      // cada coluna é um ponto da série

    int size = sizes[file_id];
    if (idx >= size) return;

    int offset = file_id * max_size;
    int res_offset = (file_id * max_size + idx) * result_cols;

    float close = series[offset + idx];

    // Inicializa os resultados
    resultados[res_offset + 0] = close;

    // SMA
    if (idx == 0) {
        resultados[res_offset + 1] = close;
    } else {
        float soma = 0.0f;
        int n = (idx < 14) ? (idx + 1) : 14;
        for (int k = 0; k < n; k++) {
            soma += series[offset + idx - k];
        }
        resultados[res_offset + 1] = soma / n;
    }

    // EMA (inicializado igual ao SMA, mas ajustado posteriormente na CPU)
    resultados[res_offset + 2] = resultados[res_offset + 1];

    // RSI
    if (idx < 14) {
        resultados[res_offset + 3] = NAN;
    } else {
        float gain = 0.0f, loss = 0.0f;
        for (int i = idx - 13; i <= idx; i++) {
            float diff = series[offset + i] - series[offset + i - 1];
            if (diff > 0) gain += diff;
            else loss -= diff;
        }
        float rs = (loss == 0.0f) ? 0.0f : gain / loss;
        resultados[res_offset + 3] = (loss == 0.0f) ? 100.0f : (100.0f - (100.0f / (1.0f + rs)));
    }

    // Stochastic %K
    if (idx < 14) {
        resultados[res_offset + 4] = NAN;
    } else {
        float low = series[offset + idx - 13];
        float high = series[offset + idx - 13];
        for (int i = idx - 13; i <= idx; i++) {
            float val = series[offset + i];
            if (val < low) low = val;
            if (val > high) high = val;
        }
        resultados[res_offset + 4] = (high - low == 0.0f) ? NAN : 100.0f * (close - low) / (high - low);
    }
}
