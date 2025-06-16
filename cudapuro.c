#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#define MAX_SIZE 10000
#define MAX_FILES 1000
#define FILENAME_LEN 256

// Variáveis globais para armazenar os caminhos dos arquivos
char filenames[MAX_FILES][FILENAME_LEN];
int num_files = 0;
// Funções externas (já implementadas com CUDA)
extern void run_cuda_sma(float* input, float* output, int len, int window);
extern void run_cuda_rsi(float* input, float* output, int len, int window);
extern void run_cuda_ema(float* input, float* output, int len, int window);
extern void run_cuda_stochastic(const float *prices, float *stoch_k, float *stoch_d, int len, int window);


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


double processar_empresa_cuda(const char *input_path, const char *output_path, const char *empresa_nome) {
    float h_data[MAX_SIZE];
    int total = load_csv(input_path, h_data, MAX_SIZE);
    if (total <= 0) {
        printf("Erro ao carregar dados de %s.\n", input_path);
        return 0.0;
    }

    int period = 14;
    float *d_data, *d_sma, *d_ema, *d_rsi, *d_stoch_k, *d_stoch_d;

cudaMalloc((void**)&d_data, total * sizeof(float));
cudaMalloc((void**)&d_sma, total * sizeof(float));
cudaMalloc((void**)&d_ema, total * sizeof(float));
cudaMalloc((void**)&d_rsi, total * sizeof(float));
cudaMalloc((void**)&d_stoch_k, total * sizeof(float));
cudaMalloc((void**)&d_stoch_d, total * sizeof(float));
    cudaMemcpy(d_data, h_data, total * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    run_cuda_sma(d_data, d_sma, total, period);
    run_cuda_ema(d_data, d_ema, total, period);
    run_cuda_rsi(d_data, d_rsi, total, period);
    run_cuda_stochastic(d_data, d_stoch_k, d_stoch_d, total, period);

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    double tempo_segundos = elapsed_ms / 1000.0;

    float h_sma[MAX_SIZE], h_ema[MAX_SIZE], h_rsi[MAX_SIZE], h_stoch_k[MAX_SIZE], h_stoch_d[MAX_SIZE];

    cudaMemcpy(h_sma, d_sma, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ema, d_ema, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rsi, d_rsi, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stoch_k, d_stoch_k, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stoch_d, d_stoch_d, total * sizeof(float), cudaMemcpyDeviceToHost);
    
    FILE *out = fopen(output_path, "w");
    if (!out) {
        perror("Erro ao abrir arquivo de saída");
        return tempo_segundos;
    }

    fprintf(out, "Index,Close,SMA,EMA,RSI,StochK,StochD\n");
    for (int i = 0; i < total; i++) {
        fprintf(out, "%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                i, h_data[i], h_sma[i], h_ema[i], h_rsi[i], h_stoch_k[i], h_stoch_d[i]);
    }

    fclose(out);
    // Libera memória GPU
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

    for (int i = 0; i < num_files; i++) {
        const char *input_path = filenames[i];

        const char *nome_arquivo = strrchr(input_path, '/');
        if (!nome_arquivo) nome_arquivo = input_path;
        else nome_arquivo++;  // pula a barra

        char nome_empresa[64];
        const char *ponto = strchr(nome_arquivo, '.');
        if (ponto) {
            strncpy(nome_empresa, nome_arquivo, ponto - nome_arquivo);
            nome_empresa[ponto - nome_arquivo] = '\0';
        } else {
            strncpy(nome_empresa, nome_arquivo, sizeof(nome_empresa));
            nome_empresa[sizeof(nome_empresa) - 1] = '\0';
        }

        char output_path[256];
        snprintf(output_path, sizeof(output_path), "%s/cuda-%s.csv", dir_saida, nome_empresa);

        //printf("Processando %s com CUDA...\n", nome_empresa);
        double tempo = processar_empresa_cuda(input_path, output_path, nome_empresa);
        tempo_total += tempo;

        //printf("Tempo de execução para %s: %.6f segundos\n", nome_empresa, tempo);
    }

    printf("\nTempo total de cálculo (CUDA): %.6f segundos\n", tempo_total);
    return 0;
}