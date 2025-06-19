#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#include <sys/time.h>

#define MAX_SIZE 1300
#define MAX_FILES 1000
#define FILENAME_LEN 256
#define RESULT_COLS 5
#define PERIOD 14


struct timeval start, end;

char filenames[MAX_FILES][FILENAME_LEN];
int num_files = 0;

__global__ void indicadores_kernel(float *series, int *sizes, float *resultados, int max_size, int result_cols, int period, int num_files) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_files) return;

    int size = sizes[i];
    if (size <= 0) return;

    float k = 2.0f / (period + 1);
    float k_antes = 1.0f - k;
    float *serie = &series[i * max_size];
    float *res = &resultados[i * max_size * result_cols];

    float ema = serie[0];
    float sum_sma = serie[0];
    float sum_gain = 0.0f, sum_loss = 0.0f;
    float low = serie[0], high = serie[0];

    res[0 * result_cols + 0] = serie[0];
    res[0 * result_cols + 1] = serie[0];
    res[0 * result_cols + 2] = serie[0];
    res[0 * result_cols + 3] = 50.0f;
    res[0 * result_cols + 4] = 50.0f;

    for (int j = 1; j < period && j < size; j++) {
        float close = serie[j];
        res[j * result_cols + 0] = close;
        sum_sma += close;
        res[j * result_cols + 1] = sum_sma / (j + 1);
        ema = close * k + ema * k_antes;
        res[j * result_cols + 2] = ema;

        float gain = 0.0f, loss = 0.0f;
        for (int p = 1; p <= j; p++) {
            float diff = serie[p] - serie[p - 1];
            if (diff > 0) gain += diff;
            else loss -= diff;
        }
        float rs = (loss == 0.0f) ? 0.0f : gain / loss;
        res[j * result_cols + 3] = (loss == 0.0f) ? 100.0f : (100.0f - (100.0f / (1.0f + rs)));

        float local_low = close, local_high = close;
        for (int k = 0; k <= j; k++) {
            float val = serie[k];
            if (val < local_low) local_low = val;
            if (val > local_high) local_high = val;
        }
        res[j * result_cols + 4] = (local_high == local_low) ? 50.0f : (100.0f * (close - local_low) / (local_high - local_low));
    }

    sum_gain /= period;
    sum_loss /= period;

    for (int j = period; j < size; j++) {
        float close = serie[j];
        res[j * result_cols + 0] = close;
        sum_sma += close - serie[j - period];
        res[j * result_cols + 1] = sum_sma / period;
        ema = close * k + ema * k_antes;
        res[j * result_cols + 2] = ema;

        float diff = close - serie[j - 1];
        float gain = (diff > 0) ? diff : 0.0f;
        float loss = (diff < 0) ? -diff : 0.0f;
        sum_gain = (sum_gain * (period - 1) + gain) / period;
        sum_loss = (sum_loss * (period - 1) + loss) / period;

        float rs = (sum_loss == 0.0f) ? 0.0f : (sum_gain / sum_loss);
        res[j * result_cols + 3] = (sum_loss == 0.0f) ? 100.0f : (100.0f - (100.0f / (1.0f + rs)));

        float val_out = serie[j - period];
        if (close >= high) high = close;
        else if (val_out == high) {
            high = serie[j - period + 1];
            for (int m = j - period + 2; m <= j; m++)
                if (serie[m] > high) high = serie[m];
        }
        if (close <= low) low = close;
        else if (val_out == low) {
            low = serie[j - period + 1];
            for (int m = j - period + 2; m <= j; m++)
                if (serie[m] < low) low = serie[m];
        }
        res[j * result_cols + 4] = (high == low) ? NAN : (100.0f * (close - low) / (high - low));
    }
}

int load_csv(const char *filename, float *data, int max_size) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;
    char line[512];
    int count = 0;
    fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp) && count < max_size) {
        char *token;
        int col = 0;
        token = strtok(line, ",");
        while (token != NULL) {
            if (col == 4) {
                data[count++] = strtof(token, NULL);
                break;
            }
            token = strtok(NULL, ",");
            col++;
        }
    }
    fclose(fp);
    return count;
}

void listar_csvs(const char *dirpath) {
    DIR *dir = opendir(dirpath);
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (strstr(entry->d_name, ".csv")) {
            snprintf(filenames[num_files], FILENAME_LEN, "%s/%s", dirpath, entry->d_name);
            num_files++;
        }
    }
    closedir(dir);
}

void salvar_csv(const char *path, float **resultados, int size) {
    FILE *out = fopen(path, "w");
    fprintf(out, "Index,Close,SMA,EMA,RSI,StochK\n");
    for (int i = 0; i < size; i++) {
        fprintf(out, "%d,%.2f,%.2f,%.2f,%.2f,%.2f\n",
            i, resultados[i][0], resultados[i][1], resultados[i][2], resultados[i][3], resultados[i][4]);
    }
    fclose(out);
}

int main() {
    const char *dir_empresas = "empresas";
    const char *dir_saida = "saida";

    listar_csvs(dir_empresas);

    float **series = (float**)malloc(num_files * sizeof(float*));
    float ***resultados = (float***)malloc(num_files * sizeof(float**));
    int *sizes = (int*)malloc(num_files * sizeof(int));

    for (int i = 0; i < num_files; i++) {
        series[i] = (float*)malloc(MAX_SIZE * sizeof(float));
        resultados[i] = (float**)malloc(MAX_SIZE * sizeof(float*));
        for (int j = 0; j < MAX_SIZE; j++) {
            resultados[i][j] = (float*)malloc(RESULT_COLS * sizeof(float));
        }
    }

    float *h_series_flat = (float*)malloc(num_files * MAX_SIZE * sizeof(float));
    float *h_result_flat = (float*)malloc(num_files * MAX_SIZE * RESULT_COLS * sizeof(float));
    int *h_sizes = (int*)malloc(num_files * sizeof(int));

    for (int i = 0; i < num_files; i++) {
        sizes[i] = load_csv(filenames[i], series[i], MAX_SIZE);
        h_sizes[i] = sizes[i];
        for (int j = 0; j < MAX_SIZE; j++) {
            h_series_flat[i * MAX_SIZE + j] = series[i][j];
        }
    }

    float *d_series, *d_result;
    int *d_sizes;
    cudaMalloc(&d_series, num_files * MAX_SIZE * sizeof(float));
    cudaMalloc(&d_result, num_files * MAX_SIZE * RESULT_COLS * sizeof(float));
    cudaMalloc(&d_sizes, num_files * sizeof(int));

    cudaMemcpy(d_series, h_series_flat, num_files * MAX_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizes, h_sizes, num_files * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (num_files + threads - 1) / threads;
    indicadores_kernel<<<blocks, threads>>>(d_series, d_sizes, d_result, MAX_SIZE, RESULT_COLS, PERIOD, num_files);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result_flat, d_result, num_files * MAX_SIZE * RESULT_COLS * sizeof(float), cudaMemcpyDeviceToHost);
    
    gettimeofday(&start, NULL);
    for (int i = 0; i < num_files; i++) {
        for (int j = 0; j < MAX_SIZE; j++) {
            for (int k = 0; k < RESULT_COLS; k++) {
                resultados[i][j][k] = h_result_flat[(i * MAX_SIZE + j) * RESULT_COLS + k];
            }
        }
    }
    gettimeofday(&end, NULL);
    double tempo = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
    printf("Tempo cálculo indicadores: %.6f s\n", tempo);


    for (int i = 0; i < num_files; i++) {
        if (sizes[i] > 0) {
            const char *nome_arquivo = strrchr(filenames[i], '/');
            nome_arquivo = nome_arquivo ? nome_arquivo + 1 : filenames[i];
            char nome_empresa[64];
            strncpy(nome_empresa, nome_arquivo, strchr(nome_arquivo, '.') - nome_arquivo);
            nome_empresa[strchr(nome_arquivo, '.') - nome_arquivo] = '\0';

            char output_path[256];
            snprintf(output_path, sizeof(output_path), "%s/cuda-%s.csv", dir_saida, nome_empresa);
            salvar_csv(output_path, resultados[i], sizes[i]);
        }
    }

    for (int i = 0; i < num_files; i++) {
        free(series[i]);
        for (int j = 0; j < MAX_SIZE; j++) free(resultados[i][j]);
        free(resultados[i]);
    }
    free(series); free(resultados); free(sizes);
    free(h_series_flat); free(h_result_flat); free(h_sizes);
    cudaFree(d_series); cudaFree(d_result); cudaFree(d_sizes);

    return 0;
}