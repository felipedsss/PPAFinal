#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <sys/time.h>
#include <omp.h>
#define MAX_SIZE 1300
#define MAX_FILES 5000
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


    #pragma omp parallel for schedule(static)

    for (int i = 0; i < num_files; i++) {
            // Processar todas as séries
            int period = 14;
            float k = 2.0f / (period + 1);
            float k_antes = 1.0f - k;
        if (sizes[i] > 0) {
            //const int period = 14;
            float close = series[i][0];
            float ema = close;

            float sum_sma = close;
            float sum_gain = 0.0f, sum_loss = 0.0f;
            float diff = 0.0f;
            float low = close, high = close;
            //para j = 0, ou seja, primeiro elemento
            
             // SMA inicializado com o primeiro Close
            resultados[i][0][0] = close; // Close
            resultados[i][0][1] = sum_sma; // SMA (não calculado)
            resultados[i][0][2] = close; // EMA (inicializado com o primeiro Close)
            resultados[i][0][3] = 50.0f; // RSI (não calculado)
            resultados[i][0][4] = 50.0f; // Stochastic %K
            // para j<period, SMA não é calculado, mas EMA é inicializado
            float gain = 0.0f, loss = 0.0f, rs=0.0f;
            for (int j = 1; j < period && j < sizes[i]; j++) {
                float close = series[i][j];
                resultados[i][j][0] = close; // Close

                // --- SMA cumulativa ---
                sum_sma += close;
                resultados[i][j][1] = sum_sma / (j + 1);

                // --- EMA exponencial com suavização inicial ---
                ema = close * k + ema * k_antes;
                resultados[i][j][2] = ema;

                // --- RSI com period = j ---
                
                for (int k = 1; k <= j; k++) {
                    diff = series[i][k] - series[i][k - 1];
                if (diff > 0) sum_gain += diff;
                else          sum_loss -= diff;
                }
                if (sum_loss == 0.0f) {
                    resultados[i][j][3] = 100.0f; // RSI 100 se não houver perda
                } else {
                    rs = sum_gain / sum_loss;
                    resultados[i][j][3] = 100.0f - (100.0f / (1.0f + rs));
                }
 

                // --- Stochastic %K com period = j ---
                
                for (int k = 0; k <= j; k++) {
                    float val = series[i][k];
                    if (val < low) low = val;
                    if (val > high) high = val;
                }
                resultados[i][j][4] = (high == low) ? 50.0f : (100.0f * (close - low) / (high - low));
            }
            // inicializando parametros do RSI
            sum_gain /= period;
            sum_loss /= period;
            // a partir de j = period, os indicadores são calculados com a janela period
            for (int j = period; j < sizes[i]; j++) {

                float close = series[i][j];
                resultados[i][j][0] = close; // Close

                // --- SMA (rolling sum) ---

                    sum_sma += close - series[i][j - period];
                    resultados[i][j][1] = sum_sma / period;
                

                // --- EMA (exponencial) ---

                ema = close * k + ema * k_antes;
                resultados[i][j][2] = ema;

                // --- RSI (rolling gain/loss pelo método de Wilder) ---

                diff = series[i][j] - series[i][j - 1];
                gain = (diff > 0) ? diff : 0.0f;
                loss = (diff < 0) ? -diff : 0.0f;

                sum_gain = (sum_gain * (period - 1) + gain) / period;
                sum_loss = (sum_loss * (period - 1) + loss) / period;

                rs = (sum_loss == 0.0f) ? 0.0f : (sum_gain / sum_loss);
                resultados[i][j][3] = (sum_loss == 0.0f) ? 100.0f : (100.0f - (100.0f / (1.0f + rs)));

                // --- Stochastic %K ---

               
                    float val_out = series[i][j - period]; // valor que saiu da janela
                    float val_in = close;                  // valor que entrou

                    // Atualiza máximo
                    if (close >= high) {
                        high = close;
                    } else if (val_out == high) {
                        // Recalcular high
                        high = series[i][j - period + 1];
                        for (int m = j - period + 2; m <= j; m++) {
                            if (series[i][m] > high)
                                high = series[i][m];
                        }
                    }

                    // Atualiza mínimo
                    if (close <= low) {
                        low = close;
                    } else if (val_out == low) {
                        // Recalcular low
                        low = series[i][j - period + 1];
                        for (int m = j - period + 2; m <= j; m++) {
                            if (series[i][m] < low)
                                low = series[i][m];
                        }
                    }

                    // Calcular Stochastic %K
                resultados[i][j][4] = (high == low) ? NAN : (100.0f * (close - low) / (high - low));
            }
        }
    }

    gettimeofday(&end, NULL);
    double tempo = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
    printf("Tempo cálculo indicadores: %.6f s\n", tempo);
    /*
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
            snprintf(output_path, sizeof(output_path), "%s/openmp-%s.csv", dir_saida, nome_empresa);

            salvar_csv(output_path, resultados[i], sizes[i]);
            //printf("Salvo: %s\n", output_path);
        }
    }
    */
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
