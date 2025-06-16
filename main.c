#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_SERIES_LENGTH 10000

float series[MAX_SERIES_LENGTH];

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

    // Lê e ignora o cabeçalho
    fgets(line, sizeof(line), fp);

    while (fgets(line, sizeof(line), fp) && count < max_size) {
        char *token;
        int column = 0;
        float close_value = 0.0;

        token = strtok(line, ",");
        while (token != NULL) {
            if (column == 4) {  // geralmente coluna 'Close' está na posição 4 (5ª), que é o caso de petr4.csv
                close_value = strtof(token, NULL);
                printf("%f\n", close_value);  // Debug: imprime o valor de fechamento
                data[count++] = close_value;
                break;  // não precisa continuar a linha
            }
            token = strtok(NULL, ",");
            column++;
        }
    }

    fclose(fp);
    return count;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float *full_series = NULL;
    int total_length = 0;

    if (rank == 0) {
        full_series = (float*)malloc(MAX_SERIES_LENGTH * sizeof(float));
        total_length = load_csv("PETR4.SA_train.csv", full_series, MAX_SERIES_LENGTH);
        if (total_length <= 0) {
            fprintf(stderr, "Erro ao carregar os dados.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("Total de dados carregados: %d\n", total_length);
    }

    MPI_Bcast(&total_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = total_length / size;
    float *prices = (float*)malloc(chunk_size * sizeof(float));
    float *sma = (float*)malloc(chunk_size * sizeof(float));
    float *rsi = (float*)malloc(chunk_size * sizeof(float));
    float *ema = (float*)malloc(chunk_size * sizeof(float));
    float *stoch_k = (float*)malloc(chunk_size * sizeof(float));
    float *stoch_d = (float*)malloc(chunk_size * sizeof(float));

    MPI_Scatter(full_series, chunk_size, MPI_FLOAT, prices, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) free(full_series);

    int window = 14;

    #pragma omp parallel
    {
        #pragma omp single
        {
            run_cuda_sma(prices, sma, chunk_size, window);
            run_cuda_rsi(prices, rsi, chunk_size, window);
            run_cuda_ema(prices, ema, chunk_size, window);
            run_cuda_stochastic(prices, stoch_k, stoch_d, chunk_size, window);
        }
    }

    float local_sum_sma = 0.0f;
    float local_sum_rsi = 0.0f;
    float local_sum_ema = 0.0f;
    // Calcula a soma local dos SMAs, RSIs e EMAs
    for (int i = window; i < chunk_size; i++) {
        local_sum_sma += sma[i];
        local_sum_rsi += rsi[i];
        local_sum_ema += ema[i];
    }

    float global_sum_sma = 0.0f, global_sum_rsi = 0.0f, global_sum_ema = 0.0f;
    MPI_Reduce(&local_sum_sma, &global_sum_sma, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_rsi, &global_sum_rsi, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum_ema, &global_sum_ema, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Juntando resultados de todos os ranks para salvar no CSV
    float *all_prices = NULL;
    float *all_sma = NULL;
    float *all_ema = NULL;
    float *all_rsi = NULL;
    float *all_stoch_k = NULL;
    float *all_stoch_d = NULL;


    if (rank == 0) {
        all_prices = (float*)malloc(total_length * sizeof(float));
        all_sma    = (float*)malloc(total_length * sizeof(float));
        all_ema    = (float*)malloc(total_length * sizeof(float));
        all_rsi    = (float*)malloc(total_length * sizeof(float));
        all_stoch_k = (float*)malloc(total_length * sizeof(float));
        all_stoch_d = (float*)malloc(total_length * sizeof(float));
    }

    MPI_Gather(prices, chunk_size, MPI_FLOAT, all_prices, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(sma, chunk_size, MPI_FLOAT, all_sma, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(ema, chunk_size, MPI_FLOAT, all_ema, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(rsi, chunk_size, MPI_FLOAT, all_rsi, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(stoch_k, chunk_size, MPI_FLOAT, all_stoch_k, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(stoch_d, chunk_size, MPI_FLOAT, all_stoch_d, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);


    if (rank == 0) {
        FILE *fp = fopen("saida_paralela.csv", "w");
        if (!fp) {
            perror("Erro ao criar arquivo de saída");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fprintf(fp, "Index,Close,SMA,EMA,RSI,StochK,StochD\n");
        for (int i = 0; i < total_length; i++) {
            fprintf(fp, "%d,%.2f,", i, all_prices[i]);
            if (i < window - 1) {
                fprintf(fp, ",,,,\n");
            } else {
                fprintf(fp, "%.2f,%.2f,%.2f,%.2f,%.2f\n",
                        all_sma[i], all_ema[i], all_rsi[i],
                        all_stoch_k[i], all_stoch_d[i]);
            }
        }
        fclose(fp);
        printf("Arquivo 'saida_paralela.csv' gerado com sucesso!\n");
    }
    // Libera a memória alocada
    free(prices);
    free(sma);
    free(rsi);
    free(ema);
    free(stoch_k);
    free(stoch_d);
    if (rank == 0) {
        printf("Processamento concluído.\n");
    }
    // Finaliza o MPI
    MPI_Finalize();
    return 0;
}
