#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <sys/time.h>

#define MAX_SIZE 1300
#define MAX_FILES 500
#define FILENAME_LEN 128
#define RESULT_COLS 5  // Close, SMA, EMA, RSI, StochK

struct timeval start, end;
char filenames[MAX_FILES][FILENAME_LEN];
int num_files = 0;

// --- Listar arquivos CSV no diretório ---
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
            if (num_files >= MAX_FILES) break;
        }
    }

    closedir(dir);
}

// --- Carregar Close price de CSV para array ---
int load_csv(const char *filename, float *data, int max_size) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror("Erro ao abrir arquivo");
        return -1;
    }

    char line[512];
    int count = 0;
    fgets(line, sizeof(line), fp);  // pular cabeçalho

    while (fgets(line, sizeof(line), fp) && count < max_size) {
        char *token;
        int col = 0;
        float close_val = 0.0f;

        token = strtok(line, ",");
        while (token != NULL) {
            if (col == 4) {  // coluna Close (index 4)
                close_val = strtof(token, NULL);
                data[count++] = close_val;
                break;
            }
            token = strtok(NULL, ",");
            col++;
        }
    }

    fclose(fp);
    return count;
}

// --- Indicadores ---
// SMA simples
float calc_sma(float *data, int idx, int period) {
    if (idx < period - 1) return NAN;
    float sum = 0.0f;
    for (int i = idx - period + 1; i <= idx; i++)
        sum += data[i];
    return sum / period;
}

// EMA exponencial
float calc_ema(float *data, int idx, int period, float prev_ema) {
    float k = 2.0f / (period + 1);
    return data[idx] * k + prev_ema * (1 - k);
}

// RSI
float calc_rsi(float *data, int idx, int period) {
    if (idx < period) return NAN;
    float gain = 0.0f, loss = 0.0f;
    for (int i = idx - period + 1; i <= idx; i++) {
        float diff = data[i] - data[i - 1];
        if (diff > 0)
            gain += diff;
        else
            loss -= diff;
    }
    if (loss == 0) return 100.0f;
    float rs = gain / loss;
    return 100.0f - (100.0f / (1.0f + rs));
}

// Stochastic %K
float calc_stochastic_k(float *data, int idx, int period) {
    if (idx < period - 1) return NAN;
    float high = data[idx], low = data[idx];
    for (int i = idx - period + 1; i <= idx; i++) {
        if (data[i] > high) high = data[i];
        if (data[i] < low)  low = data[i];
    }
    if (high == low) return NAN;
    return 100.0f * (data[idx] - low) / (high - low);
}

// --- Processar uma empresa: preencher resultados [linhas][colunas] ---
void processar_empresa(float *serie, int tamanho, float **resultados) {
    int period = 14;
    float ema = serie[0];

    for (int i = 0; i < tamanho; i++) {
        resultados[i][0] = serie[i];                    // Close
        resultados[i][1] = calc_sma(serie, i, period); // SMA
        ema = (i == 0) ? serie[0] : calc_ema(serie, i, period, ema);
        resultados[i][2] = ema;                         // EMA
        resultados[i][3] = calc_rsi(serie, i, period); // RSI
        resultados[i][4] = calc_stochastic_k(serie, i, period); // StochK
    }
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

int main() {
    const char *dir_empresas = "empresas";
    const char *dir_saida = "saida";

    listar_csvs(dir_empresas);

    // Alocar séries (float * por empresa)
    float **series = malloc(num_files * sizeof(float*));
    int *sizes = malloc(num_files * sizeof(int));

    // Alocar resultados [num_files][MAX_SIZE][RESULT_COLS]
    float ***resultados = malloc(num_files * sizeof(float**));
    for (int i = 0; i < num_files; i++) {
        series[i] = malloc(MAX_SIZE * sizeof(float));
        resultados[i] = malloc(MAX_SIZE * sizeof(float*));
        for (int j = 0; j < MAX_SIZE; j++) {
            resultados[i][j] = malloc(RESULT_COLS * sizeof(float));
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

    gettimeofday(&start, NULL);

    // Processar todas as séries
    for (int i = 0; i < num_files; i++) {
        if (sizes[i] > 0) {
            processar_empresa(series[i], sizes[i], resultados[i]);
        }
    }

    gettimeofday(&end, NULL);
    double tempo = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
    printf("Tempo cálculo indicadores: %.6f s\n", tempo);

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
            snprintf(output_path, sizeof(output_path), "%s/sequencial-%s.csv", dir_saida, nome_empresa);

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
