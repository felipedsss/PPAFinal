// Adaptado para processar todas as empresas em lote (kernel 2D)
// Indicadores SMA, EMA, RSI e Stochastic (com CUDA kernel 2D)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#define MAX_SIZE 1300
#define MAX_FILES 1000
#define FILENAME_LEN 256
#define RESULT_COLS 5

char filenames[MAX_FILES][FILENAME_LEN];
int num_files = 0;


// --- Acesso linear para 2D ---
#define IDX(s,i,len) ((s)*(len) + (i))

// --- Kernel SMA com stride ---
__global__ void sma_kernel_stride(const float *input, float *output,
                                  int len, int window, int total_series)
{
    int series_idx = blockIdx.x;
    int tid        = threadIdx.x;
    int stride     = blockDim.x;

    if (series_idx >= total_series) return;

    // Cada thread processa índices i = tid, tid+stride, tid+2*stride, ...
    for (int i = tid; i < len; i += stride) {
        if (i >= window - 1) {
            float sum = 0.0f;
            for (int k = 0; k < window; ++k)
                sum += input[ IDX(series_idx, i - k, len) ];
            output[ IDX(series_idx, i, len) ] = sum / window;
        } else {
            output[ IDX(series_idx, i, len) ] = NAN;
        }
    }
}

// --- Kernel EMA com stride ---
__global__ void ema_kernel_stride(const float *input, float *output,
                                  int len, int window, int total_series)
{
    int series_idx = blockIdx.x;
    int tid        = threadIdx.x;
    int stride     = blockDim.x;
    float alpha    = 2.0f / (window + 1);

    if (series_idx >= total_series) return;

    // Para EMA cada i depende de i-1, então vamos fazer:
    // 1) thread 0 inicializa output[0]
    // 2) __syncthreads() para todos lerem
    // 3) todas percorrem stride (iniciando em tid, mas garantir ordem)
    if (tid == 0) {
        output[ IDX(series_idx, 0, len) ] = input[ IDX(series_idx, 0, len) ];
    }
    __syncthreads();

    // Em seguida cada thread faz sua parte, mas deve respeitar dependência
    // A forma simples: cada thread serialmente percorre i += stride,
    // lendo output[i-1] que já está calculado por thread ou por tid==0
    for (int i = tid; i < len; i += stride) {
        if (i == 0) continue;
        float prev = output[ IDX(series_idx, i - 1, len) ];
        float curr = input[  IDX(series_idx, i, len)     ];
        output[ IDX(series_idx, i, len) ] = alpha * curr + (1 - alpha) * prev;
    }
}

// --- Kernel RSI com stride ---
__global__ void rsi_kernel_stride(const float *input, float *output,
                                  int len, int window, int total_series)
{
    int series_idx = blockIdx.x;
    int tid        = threadIdx.x;
    int stride     = blockDim.x;

    if (series_idx >= total_series) return;

    for (int i = tid; i < len; i += stride) {
        if (i < window) {
            output[ IDX(series_idx, i, len) ] = NAN;
            continue;
        }
        float gain = 0.0f, loss = 0.0f;
        for (int k = i - window + 1; k <= i; ++k) {
            float delta = input[IDX(series_idx, k, len)] - input[IDX(series_idx, k - 1, len)];
            if (delta > 0) gain += delta;
            else          loss -= delta;
        }
        float rs = (loss == 0.0f) ? 100.0f : gain / loss;
        output[ IDX(series_idx, i, len) ] = 100.0f - (100.0f / (1.0f + rs));
    }
}

// --- Kernel Stochastic %K com stride ---
__global__ void stochk_kernel_stride(const float *input, float *output,
                                     int len, int window, int total_series)
{
    int series_idx = blockIdx.x;
    int tid        = threadIdx.x;
    int stride     = blockDim.x;

    if (series_idx >= total_series) return;

    for (int i = tid; i < len; i += stride) {
        if (i < window - 1) {
            output[ IDX(series_idx, i, len) ] = NAN;
            continue;
        }
        float highest = input[ IDX(series_idx, i - window + 1, len) ];
        float lowest  = highest;
        for (int k = i - window + 1; k <= i; ++k) {
            float v = input[ IDX(series_idx, k, len) ];
            if (v > highest) highest = v;
            if (v <  lowest ) lowest  = v;
        }
        float denom = highest - lowest;
        if (denom == 0.0f)
            output[ IDX(series_idx, i, len) ] = 0.0f;
        else {
            float close = input[ IDX(series_idx, i, len) ];
            output[ IDX(series_idx, i, len) ] = (close - lowest) / denom * 100.0f;
        }
    }
}

// --- Wrappers com strike ---
void run_cuda_sma_batch(float **in, float **out,
                        int num_series, int len, int window)
{
    // flatten
    float *h_in  = (float*) malloc(num_series*len*sizeof(float));
    float *h_out = (float*) malloc(num_series*len*sizeof(float));
    for(int s=0;s<num_series;++s)
        memcpy(h_in + s*len, in[s], len*sizeof(float));

    float *d_in, *d_out;
    cudaMalloc(&d_in,  num_series*len*sizeof(float));
    cudaMalloc(&d_out, num_series*len*sizeof(float));
    cudaMemcpy(d_in, h_in, num_series*len*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(num_series);
    sma_kernel_stride<<<grid,block>>>(d_in,d_out,len,window,num_series);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, num_series*len*sizeof(float), cudaMemcpyDeviceToHost);
    for(int s=0;s<num_series;++s)
        memcpy(out[s], h_out + s*len, len*sizeof(float));

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}

void run_cuda_ema_batch(float **in, float **out,
                        int num_series, int len, int window)
{
    float *h_in  = (float*) malloc(num_series*len*sizeof(float));
    float *h_out = (float*) malloc(num_series*len*sizeof(float));
    for(int s=0;s<num_series;++s)
        memcpy(h_in + s*len, in[s], len*sizeof(float));

    float *d_in, *d_out;
    cudaMalloc(&d_in,  num_series*len*sizeof(float));
    cudaMalloc(&d_out, num_series*len*sizeof(float));
    cudaMemcpy(d_in, h_in, num_series*len*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(num_series);
    ema_kernel_stride<<<grid,block>>>(d_in,d_out,len,window,num_series);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, num_series*len*sizeof(float), cudaMemcpyDeviceToHost);
    for(int s=0;s<num_series;++s)
        memcpy(out[s], h_out + s*len, len*sizeof(float));

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}

void run_cuda_rsi_batch(float **in, float **out,
                        int num_series, int len, int window)
{
    float *h_in  = (float*) malloc(num_series*len*sizeof(float));
    float *h_out = (float*) malloc(num_series*len*sizeof(float));
    for(int s=0;s<num_series;++s)
        memcpy(h_in + s*len, in[s], len*sizeof(float));

    float *d_in, *d_out;
    cudaMalloc(&d_in,  num_series*len*sizeof(float));
    cudaMalloc(&d_out, num_series*len*sizeof(float));
    cudaMemcpy(d_in, h_in, num_series*len*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(num_series);
    rsi_kernel_stride<<<grid,block>>>(d_in,d_out,len,window,num_series);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, num_series*len*sizeof(float), cudaMemcpyDeviceToHost);
    for(int s=0;s<num_series;++s)
        memcpy(out[s], h_out + s*len, len*sizeof(float));

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}

void run_cuda_stochk_batch(float **in, float **out,
                           int num_series, int len, int window)
{
    float *h_in  = (float*) malloc(num_series*len*sizeof(float));
    float *h_out = (float*) malloc(num_series*len*sizeof(float));
    for(int s=0;s<num_series;++s)
        memcpy(h_in + s*len, in[s], len*sizeof(float));

    float *d_in, *d_out;
    cudaMalloc(&d_in,  num_series*len*sizeof(float));
    cudaMalloc(&d_out, num_series*len*sizeof(float));
    cudaMemcpy(d_in, h_in, num_series*len*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(num_series);
    stochk_kernel_stride<<<grid,block>>>(d_in,d_out,len,window,num_series);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, num_series*len*sizeof(float), cudaMemcpyDeviceToHost);
    for(int s=0;s<num_series;++s)
        memcpy(out[s], h_out + s*len, len*sizeof(float));

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
}
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
    const char *dir_empresas = "empresas";
    const char *dir_saida = "saida";
    listar_csvs(dir_empresas);

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

    for (int i = 0; i < num_files; i++) {
        sizes[i] = load_csv(filenames[i], series[i], MAX_SIZE);
        if (sizes[i] < 0) { printf("Falha ao carregar %s\n", filenames[i]); sizes[i] = 0; }
    }

    double tempo_total = 0.0;
    float **sma = (float **)malloc(num_files * sizeof(float*));
    float **ema = (float **)malloc(num_files * sizeof(float*));
    float **rsi = (float **)malloc(num_files * sizeof(float*));
    float **stoch = (float **)malloc(num_files * sizeof(float*));

    for (int i = 0; i < num_files; i++) {
        sma[i] = (float *)malloc(MAX_SIZE * sizeof(float));
        ema[i] = (float *)malloc(MAX_SIZE * sizeof(float));
        rsi[i] = (float *)malloc(MAX_SIZE * sizeof(float));
        stoch[i] = (float *)malloc(MAX_SIZE * sizeof(float));
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    run_cuda_sma_batch(series, sma, num_files, MAX_SIZE, 14);
    run_cuda_ema_batch(series, ema, num_files, MAX_SIZE, 14);
    run_cuda_rsi_batch(series, rsi, num_files, MAX_SIZE, 14);
    run_cuda_stochk_batch(series, stoch, num_files, MAX_SIZE, 14);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    tempo_total = elapsed_ms / 1000.0;

    printf("\nTempo total de cálculo (batch CUDA): %.6f segundos\n", tempo_total);

    for (int i = 0; i < num_files; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            resultados[i][j][0] = series[i][j];
            resultados[i][j][1] = sma[i][j];
            resultados[i][j][2] = ema[i][j];
            resultados[i][j][3] = rsi[i][j];
            resultados[i][j][4] = stoch[i][j];
        }
    }

    for (int i = 0; i < num_files; i++) {
        if (sizes[i] > 0) {
            const char *input_path = filenames[i];
            const char *nome_arquivo = strrchr(input_path, '/');
            if (!nome_arquivo) nome_arquivo = input_path; else nome_arquivo++;
            char nome_empresa[64];
            strncpy(nome_empresa, nome_arquivo, strchr(nome_arquivo, '.') - nome_arquivo);
            nome_empresa[strchr(nome_arquivo, '.') - nome_arquivo] = '\0';
            char output_path[256];
            snprintf(output_path, sizeof(output_path), "%s/cuda-batch-%s.csv", dir_saida, nome_empresa);
            salvar_csv(output_path, resultados[i], sizes[i]);
        }
    }

    for (int i = 0; i < num_files; i++) {
        free(series[i]);
        free(sma[i]); free(ema[i]); free(rsi[i]); free(stoch[i]);
        for (int j = 0; j < MAX_SIZE; j++) free(resultados[i][j]);
        free(resultados[i]);
    }
    free(series); free(resultados); free(sma); free(ema); free(rsi); free(stoch); free(sizes);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
