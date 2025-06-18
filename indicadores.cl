__kernel void calc_sma(__global const float* data, __global float* sma, int size, int period) {
    int i = get_global_id(0);
    if (i >= size) return;

    if (i < period - 1) {
        sma[i] = NAN;
        return;
    }

    float sum = 0.0f;
    for (int j = i - period + 1; j <= i; j++) {
        sum += data[j];
    }
    sma[i] = sum / period;
}

__kernel void calc_ema(__global const float* data, __global float* ema, int size, int period) {
    int i = get_global_id(0);
    if (i >= size) return;

    float k = 2.0f / (period + 1);
    if (i == 0) {
        ema[0] = data[0];
    } else {
        // Como EMA depende do valor anterior, calcularemos sequencialmente no host após
        // ou usando abordagem simplificada:
        ema[i] = data[i] * k + ema[i - 1] * (1 - k);
    }
}

__kernel void calc_rsi(__global const float* data, __global float* rsi, int size, int period) {
    int i = get_global_id(0);
    if (i >= size) return;

    if (i < period) {
        rsi[i] = NAN;
        return;
    }

    float gain = 0.0f, loss = 0.0f;
    for (int j = i - period + 1; j <= i; j++) {
        float diff = data[j] - data[j - 1];
        if (diff > 0)
            gain += diff;
        else
            loss -= diff;
    }
    if (loss == 0.0f) {
        rsi[i] = 100.0f;
        return;
    }
    float rs = gain / loss;
    rsi[i] = 100.0f - (100.0f / (1.0f + rs));
}

__kernel void calc_stochastic_k(__global const float* data, __global float* stochk, int size, int period) {
    int i = get_global_id(0);
    if (i >= size) return;

    if (i < period - 1) {
        stochk[i] = NAN;
        return;
    }

    float high = data[i];
    float low = data[i];
    for (int j = i - period + 1; j <= i; j++) {
        if (data[j] > high) high = data[j];
        if (data[j] < low)  low = data[j];
    }
    if (high == low) {
        stochk[i] = NAN;
        return;
    }
    stochk[i] = 100.0f * (data[i] - low) / (high - low);
}
