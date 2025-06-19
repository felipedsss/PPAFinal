__kernel void calcular_indicadores(
    __global const float *series,     // [num_series * MAX_SIZE]
    __global const int *sizes,        // [num_series]
    __global float *resultados,       // [num_series * MAX_SIZE * 5]
    const int num_series,
    const int max_size,
    const int period
) {
    int global_id = get_global_id(0);
    int i = global_id / max_size;  // série (empresa)
    int j = global_id % max_size;  // índice na série

    if (i >= num_series || j >= sizes[i])
        return;

    int offset = i * max_size;
    int res_offset = offset * 5;

    float k = 2.0f / (period + 1.0f);
    float k_antes = 1.0f - k;

    // Variáveis auxiliares por série (apenas uma thread j=0 faz a inicialização)
    if (j == 0) {
        float close = series[offset];
        resultados[res_offset + j * 5 + 0] = close;
        resultados[res_offset + j * 5 + 1] = close;
        resultados[res_offset + j * 5 + 2] = close;
        resultados[res_offset + j * 5 + 3] = 50.0f;
        resultados[res_offset + j * 5 + 4] = 50.0f;
        return;
    }

    float sum_sma = 0.0f;
    float ema = series[offset];
    float sum_gain = 0.0f;
    float sum_loss = 0.0f;
    float close = series[offset + j];
    float low = close, high = close;

    if (j < period) {
        // SMA e EMA
        for (int k = 0; k <= j; k++) {
            sum_sma += series[offset + k];
        }
        resultados[res_offset + j * 5 + 1] = sum_sma / (j + 1);

        for (int k = 1; k <= j; k++) {
            float atual = series[offset + k];
            ema = atual * k + ema * k_antes;
        }
        resultados[res_offset + j * 5 + 2] = ema;

        // RSI
        for (int k = 1; k <= j; k++) {
            float diff = series[offset + k] - series[offset + k - 1];
            if (diff > 0) sum_gain += diff;
            else sum_loss -= diff;
        }

        float rs = (sum_loss == 0.0f) ? 0.0f : (sum_gain / sum_loss);
        resultados[res_offset + j * 5 + 3] = (sum_loss == 0.0f) ? 100.0f : 100.0f - (100.0f / (1.0f + rs));

        // Stoch %K
        for (int k = 0; k <= j; k++) {
            float val = series[offset + k];
            if (val < low) low = val;
            if (val > high) high = val;
        }

        resultados[res_offset + j * 5 + 4] = (high == low) ? 50.0f : (100.0f * (close - low) / (high - low));

    } else {
        // Rolling SMA
        for (int k = j - period + 1; k <= j; k++) {
            sum_sma += series[offset + k];
        }
        resultados[res_offset + j * 5 + 1] = sum_sma / period;

        // Rolling EMA
        ema = series[offset];
        for (int k = 1; k <= j; k++) {
            ema = series[offset + k] * k + ema * k_antes;
        }
        resultados[res_offset + j * 5 + 2] = ema;

        // RSI
        sum_gain = 0.0f;
        sum_loss = 0.0f;
        for (int k = j - period + 1; k <= j; k++) {
            float diff = series[offset + k] - series[offset + k - 1];
            if (diff > 0) sum_gain += diff;
            else sum_loss -= diff;
        }

        float rs = (sum_loss == 0.0f) ? 0.0f : (sum_gain / sum_loss);
        resultados[res_offset + j * 5 + 3] = (sum_loss == 0.0f) ? 100.0f : 100.0f - (100.0f / (1.0f + rs));

        // Stoch %K
        low = high = series[offset + j - period + 1];
        for (int k = j - period + 1; k <= j; k++) {
            float val = series[offset + k];
            if (val < low) low = val;
            if (val > high) high = val;
        }

        resultados[res_offset + j * 5 + 4] = (high == low) ? 50.0f : (100.0f * (close - low) / (high - low));
    }

    resultados[res_offset + j * 5 + 0] = close; // Sempre salvar o Close
}
