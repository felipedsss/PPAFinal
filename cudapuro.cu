#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#define MAX_SIZE 1300 // Tamanho máximo da série temporal
#define MAX_FILES 1000
#define FILENAME_LEN 256
#define RESULT_COLS 5  // Close, SMA, EMA, RSI, StochK

// Variáveis globais para armazenar os caminhos dos arquivos
char filenames[MAX_FILES][FILENAME_LEN];
int num_files = 0;
// Funções externas (já implementadas com CUDA)

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

void run_cuda_stochastic(const float *prices, float *stoch_k, float *stoch_d, int len, int window) {
    float *d_prices, *d_k, *d_d;
    cudaMalloc(&d_prices, len * sizeof(float));
    cudaMalloc(&d_k, len * sizeof(float));
    cudaMalloc(&d_d, len * sizeof(float));

    cudaMemcpy(d_prices, prices, len * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (len + blockSize - 1) / blockSize;
    stochastic_kernel<<<numBlocks, blockSize>>>(d_prices, d_k, d_d, len, window);
    cudaDeviceSynchronize();         // <<< você precisa disto!

    cudaMemcpy(stoch_k, d_k, len * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(stoch_d, d_d, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_prices);
    cudaFree(d_k);
    cudaFree(d_d);
}

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
void run_cuda_ema(float* input, float* output, int len, int window) {
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

void run_cuda_sma(float* input, float* output, int len, int window);

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
    cudaDeviceSynchronize();         // <<< você precisa disto!

    cudaMemcpy(output, d_output, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


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

void run_cuda_rsi(float* input, float* output, int len, int window) {
    float *d_input, *d_output;

    cudaMalloc((void**)&d_input, len * sizeof(float));
    cudaMalloc((void**)&d_output, len * sizeof(float));

    cudaMemcpy(d_input, input, len * sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (len + threads - 1) / threads;

    rsi_kernel<<<blocks, threads>>>(d_input, d_output, len, window);
    cudaDeviceSynchronize();         // <<< você precisa disto!
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Erro no kernel %s: %s\n",
                __func__, cudaGetErrorString(err));
    }
    cudaMemcpy(output, d_output, len * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int load_csv(const char *filename, float *data, int max_size) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Erro ao abrir o arquivo");
        return -1;
    }

    char line[512];
    int count = 0;
    fgets(line, sizeof(line), fp);  // ignora cabeçalho

    while (fgets(line, sizeof(line), fp) && count < max_size) {
        char *token;
        int column = 0;
        float close_value = 0.0;

        token = strtok(line, ",");
        while (token != NULL) {
            if (column == 4) {
                close_value = strtof(token, NULL);
                data[count++] = close_value;
                break;
            }
            token = strtok(NULL, ",");
            column++;
        }
    }

    fclose(fp);
    return count;
}

void listar_csvs(const char *dirpath) {
    DIR *dir = opendir(dirpath);
    struct dirent *entry;

    if (!dir) {
        perror("Erro ao abrir diretório");
        exit(1);
    }

    while ((entry = readdir(dir))) {
        if (strstr(entry->d_name, ".csv")) {
            snprintf(filenames[num_files], FILENAME_LEN, "%s/%s", dirpath, entry->d_name);
            num_files++;
        }
    }

    closedir(dir);
}

// --- Salvar CSV de saída ---
void salvar_csv(const char *output_path, float **resultados, int tamanho) {
    FILE *out = fopen(output_path, "w");
    if (!out) {
        perror("Erro ao criar arquivo de saída");
        return;
    }

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
double processar_empresa_cuda(float *serie, int tamanho, float **resultado) {
    int period = 14;
    float *d_data, *d_sma, *d_ema, *d_rsi, *d_stoch_k, *d_stoch_d;

    // Alocar memória na GPU
    cudaMalloc((void**)&d_data, tamanho * sizeof(float));
    cudaMalloc((void**)&d_sma, tamanho * sizeof(float));
    cudaMalloc((void**)&d_ema, tamanho * sizeof(float));
    cudaMalloc((void**)&d_rsi, tamanho * sizeof(float));
    cudaMalloc((void**)&d_stoch_k, tamanho * sizeof(float));
    cudaMalloc((void**)&d_stoch_d, tamanho * sizeof(float));

    // Copiar dados para a GPU
    cudaMemcpy(d_data, serie, tamanho * sizeof(float), cudaMemcpyHostToDevice);

    // Medir tempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Executar cálculos na GPU
    run_cuda_sma(d_data, d_sma, tamanho, period);
    run_cuda_ema(d_data, d_ema, tamanho, period);
    run_cuda_rsi(d_data, d_rsi, tamanho, period);
    run_cuda_stochastic(d_data, d_stoch_k, d_stoch_d, tamanho, period);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double tempo_segundos = elapsed_ms / 1000.0;

    // Copiar resultados da GPU para a CPU
    float *h_sma = (float *)malloc(tamanho * sizeof(float));
    float *h_ema = (float *)malloc(tamanho * sizeof(float));
    float *h_rsi = (float *)malloc(tamanho * sizeof(float));
    float *h_stoch_k = (float *)malloc(tamanho * sizeof(float));
    float *h_stoch_d = (float *)malloc(tamanho * sizeof(float));

    cudaMemcpy(h_sma, d_sma, tamanho * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ema, d_ema, tamanho * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rsi, d_rsi, tamanho * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stoch_k, d_stoch_k, tamanho * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stoch_d, d_stoch_d, tamanho * sizeof(float), cudaMemcpyDeviceToHost);

    // Montar resultado final
    for (int i = 0; i < tamanho; i++) {
        resultado[i][0] = serie[i];        // Close
        resultado[i][1] = h_sma[i];        // SMA
        resultado[i][2] = h_ema[i];        // EMA
        resultado[i][3] = h_rsi[i];        // RSI
        resultado[i][4] = h_stoch_k[i];    // Stochastic %K
    }

    // Liberar memória
    free(h_sma);
    free(h_ema);
    free(h_rsi);
    free(h_stoch_k);
    free(h_stoch_d);
    cudaFree(d_data);
    cudaFree(d_sma);
    cudaFree(d_ema);
    cudaFree(d_rsi);
    cudaFree(d_stoch_k);
    cudaFree(d_stoch_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tempo_segundos;
}

int main() {
    const char *dir_empresas = "empresas";
    const char *dir_saida = "saida";

    listar_csvs(dir_empresas);

    // Cria diretório de saída se não existir
    //mkdir(dir_saida, 0777);

    double tempo_total = 0.0;
    // Alocar séries (float * por empresa)
float **series      = (float**) malloc(num_files * sizeof(float*));
int   *sizes        = (int*)    malloc(num_files * sizeof(int));
float ***resultados = (float***)malloc(num_files * sizeof(float**));
for (int i = 0; i < num_files; i++) {
    series[i]     = (float*) malloc(MAX_SIZE * sizeof(float));
    resultados[i] = (float**) malloc(MAX_SIZE * sizeof(float*));
    for (int j = 0; j < MAX_SIZE; j++) {
        resultados[i][j] = (float*) malloc(RESULT_COLS * sizeof(float));
    }
}
    // Carregar todas as séries
    for (int i = 0; i < num_files; i++) {
        sizes[i] = load_csv(filenames[i], series[i], MAX_SIZE);
        if (sizes[i] < 0) {
            printf("Falha ao carregar %s\n", filenames[i]);
            sizes[i] = 0;
        }
    }

        // Processar todas as séries
    for (int i = 0; i < num_files; i++) {
        if (sizes[i] > 0) {
            double tempo = processar_empresa_cuda(series[i], sizes[i], resultados[i]);
            tempo_total += tempo;
        }
    }


    printf("\nTempo total de cálculo (CUDA): %.6f segundos\n", tempo_total);
        // Salvar resultados
    for (int i = 0; i < num_files; i++) {
        if (sizes[i] > 0) {
            // extrair nome do arquivo
            const char *input_path = filenames[i];
            const char *nome_arquivo = strrchr(input_path, '/');
            if (!nome_arquivo) nome_arquivo = input_path;
            else nome_arquivo++;

            // nome sem extensão
            char nome_empresa[64];
            strncpy(nome_empresa, nome_arquivo, strchr(nome_arquivo, '.') - nome_arquivo);
            nome_empresa[strchr(nome_arquivo, '.') - nome_arquivo] = '\0';

            char output_path[256];
            snprintf(output_path, sizeof(output_path), "%s/cuda-sequencial-%s.csv", dir_saida, nome_empresa);

            salvar_csv(output_path, resultados[i], sizes[i]);
            //printf("Salvo: %s\n", output_path);
        }
    }

    // Liberar memória
    for (int i = 0; i < num_files; i++) {
        free(series[i]);
        for (int j = 0; j < MAX_SIZE; j++) {
            free(resultados[i][j]);
        }
        free(resultados[i]);
    }
    free(series);
    free(resultados);
    free(sizes);


    return 0;
}


