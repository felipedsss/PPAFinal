#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#include <sys/time.h>

#define MAX_SIZE 1300
#define MAX_FILES 5000
#define FILENAME_LEN 256
#define RESULT_COLS 5
#define PERIOD 14
struct timeval start, end;

char filenames[MAX_FILES][FILENAME_LEN];
int num_files = 0;

__global__ void indicadores_kernel_2D(float *series, int *sizes, float *resultados, int max_size, int result_cols, int period) {
    int file_idx = blockIdx.x;
    int j = threadIdx.x;

    float *serie = &series[file_idx * max_size];
    float *res = &resultados[file_idx * max_size * result_cols];
    int size = sizes[file_idx];
    if (j >= size) return;

    float k = 2.0f / (period + 1);
    float k_antes = 1.0f - k;

    if (j == 0) {
        float ema = serie[0];
        res[0 * result_cols + 0] = serie[0];
        res[0 * result_cols + 1] = serie[0];
        res[0 * result_cols + 2] = ema;
        res[0 * result_cols + 3] = 50.0f;
        res[0 * result_cols + 4] = 50.0f;
    } else if (j < period) {
        float sum_sma = 0.0f, ema = serie[0], gain = 0.0f, loss = 0.0f;
        for (int i = 0; i <= j; i++) {
            sum_sma += serie[i];
            if (i > 0) {
                float diff = serie[i] - serie[i - 1];
                if (diff > 0) gain += diff;
                else loss -= diff;
            }
        }
        float close = serie[j];
        float rs = (loss == 0.0f) ? 0.0f : gain / loss;

        float local_low = close, local_high = close;
        for (int i = 0; i <= j; i++) {
            if (serie[i] < local_low) local_low = serie[i];
            if (serie[i] > local_high) local_high = serie[i];
        }

        res[j * result_cols + 0] = close;
        res[j * result_cols + 1] = sum_sma / (j + 1);
        for (int i = 1; i <= j; i++) ema = serie[i] * k + ema * k_antes;
        res[j * result_cols + 2] = ema;
        res[j * result_cols + 3] = (loss == 0.0f) ? 100.0f : (100.0f - (100.0f / (1.0f + rs)));
        res[j * result_cols + 4] = (local_high == local_low) ? 50.0f : (100.0f * (close - local_low) / (local_high - local_low));

    } else {
        float sum_sma = 0.0f;
        for (int i = j - period + 1; i <= j; i++) sum_sma += serie[i];
        float ema = serie[j - 1];
        for (int i = j - period + 1; i <= j; i++) ema = serie[i] * k + ema * k_antes;
        float diff = serie[j] - serie[j - 1];
        float gain = (diff > 0) ? diff : 0.0f;
        float loss = (diff < 0) ? -diff : 0.0f;

        float sum_gain = 0.0f, sum_loss = 0.0f;
        for (int i = j - period + 1; i <= j; i++) {
            float d = serie[i] - serie[i - 1];
            if (d > 0) sum_gain += d;
            else sum_loss -= d;
        }

        float rs = (sum_loss == 0.0f) ? 0.0f : sum_gain / sum_loss;

        float high = serie[j - period + 1], low = serie[j - period + 1];
        for (int i = j - period + 1; i <= j; i++) {
            if (serie[i] > high) high = serie[i];
            if (serie[i] < low) low = serie[i];
        }

        res[j * result_cols + 0] = serie[j];
        res[j * result_cols + 1] = sum_sma / period;
        res[j * result_cols + 2] = ema;
        res[j * result_cols + 3] = (sum_loss == 0.0f) ? 100.0f : (100.0f - (100.0f / (1.0f + rs)));
        res[j * result_cols + 4] = (high == low) ? 50.0f : (100.0f * (serie[j] - low) / (high - low));
    }
}

// Restante igual (listar_csvs, load_csv, salvar_csv)... Se quiser também, posso colar aqui.
int load_csv(const char *filename, float *data, int max_size) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;
    char line[512]; int count = 0;
    fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp) && count < max_size) {
        char *token = strtok(line, ","); int column = 0;
        while (token != NULL) {
            if (column == 4) {
                data[count++] = strtof(token, NULL);
                break;
            }
            token = strtok(NULL, ","); column++;
        }
    }
    fclose(fp); return count;
}

void listar_csvs(const char *dirpath) {
    DIR *dir = opendir(dirpath); struct dirent *entry;
    if (!dir) { perror("Erro ao abrir diretório"); exit(1); }
    while ((entry = readdir(dir))) {
        if (strstr(entry->d_name, ".csv")) {
            snprintf(filenames[num_files], FILENAME_LEN, "%s/%s", dirpath, entry->d_name);
            num_files++;
        }
    }
    closedir(dir);
}

void salvar_csv(const char *output_path, float **resultados, int tamanho) {
    FILE *out = fopen(output_path, "w");
    if (!out) return;
    fprintf(out, "Index,Close,SMA,EMA,RSI,StochK\n");
    for (int i = 0; i < tamanho; i++) {
        char sma_buf[16] = "", rsi_buf[16] = "", stoch_buf[16] = "";
        if (!isnan(resultados[i][1])) sprintf(sma_buf, "%.2f", resultados[i][1]);
        if (!isnan(resultados[i][3])) sprintf(rsi_buf, "%.2f", resultados[i][3]);
        if (!isnan(resultados[i][4])) sprintf(stoch_buf, "%.2f", resultados[i][4]);
        fprintf(out, "%d,%.2f,%s,%.2f,%s,%s\n", i, resultados[i][0], sma_buf, resultados[i][2], rsi_buf, stoch_buf);
    }
    fclose(out);
}

int main() {
    listar_csvs("empresas");

    float **series = (float **)malloc(num_files * sizeof(float*));
    float ***resultados = (float ***)malloc(num_files * sizeof(float**));
    int *sizes = (int *)malloc(num_files * sizeof(int));

    for (int i = 0; i < num_files; i++) {
        series[i] = (float *)malloc(MAX_SIZE * sizeof(float));
        resultados[i] = (float **)malloc(MAX_SIZE * sizeof(float*));
        for (int j = 0; j < MAX_SIZE; j++) {
            resultados[i][j] = (float *)malloc(RESULT_COLS * sizeof(float));
        }
    }

    float *h_series_flat = (float *)malloc(num_files * MAX_SIZE * sizeof(float));
    float *h_result_flat = (float *)malloc(num_files * MAX_SIZE * RESULT_COLS * sizeof(float));
    int *h_sizes = (int *)malloc(num_files * sizeof(int));

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

    dim3 grid(num_files);
    dim3 block(MAX_SIZE);
    gettimeofday(&start, NULL);
    indicadores_kernel_2D<<<grid, block>>>(d_series, d_sizes, d_result, MAX_SIZE, RESULT_COLS, PERIOD);
    gettimeofday(&end, NULL);

    cudaDeviceSynchronize();
    double tempo = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
    printf("Tempo cálculo indicadores: %.6f s\n", tempo);
    cudaMemcpy(h_result_flat, d_result, num_files * MAX_SIZE * RESULT_COLS * sizeof(float), cudaMemcpyDeviceToHost);


    for (int i = 0; i < num_files; i++) {
        for (int j = 0; j < MAX_SIZE; j++) {
            for (int k = 0; k < RESULT_COLS; k++) {
                resultados[i][j][k] = h_result_flat[(i * MAX_SIZE + j) * RESULT_COLS + k];
            }
        }
    }

    /*
    for (int i = 0; i < num_files; i++) {
        if (sizes[i] > 0) {
            const char *nome = strrchr(filenames[i], '/');
            nome = nome ? nome + 1 : filenames[i];
            char nome_empresa[64];
            strncpy(nome_empresa, nome, strchr(nome, '.') - nome);
            nome_empresa[strchr(nome, '.') - nome] = '\0';

            char output_path[256];
            snprintf(output_path, sizeof(output_path), "saida/cuda-%s.csv", nome_empresa);
            salvar_csv(output_path, resultados[i], sizes[i]);
        }
    }
    */
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

