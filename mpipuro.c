#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <sys/time.h>
#include <mpi.h>

#define MAX_SIZE 1300
#define MAX_FILES 500
#define FILENAME_LEN 128
#define RESULT_COLS 5

char filenames[MAX_FILES][FILENAME_LEN];
int num_files = 0;

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

void calcular_indicadores(float *serie, int size, float **resultado, int period) {
    float k = 2.0f / (period + 1);
    float k_antes = 1.0f - k;
    float ema = serie[0];
    float sum_sma = serie[0], sum_gain = 0.0f, sum_loss = 0.0f;
    float low = serie[0], high = serie[0];
    resultado[0][0] = serie[0];
    resultado[0][1] = serie[0];
    resultado[0][2] = ema;
    resultado[0][3] = 50.0f;
    resultado[0][4] = 50.0f;

    for (int j = 1; j < period && j < size; j++) {
        float close = serie[j];
        resultado[j][0] = close;
        sum_sma += close;
        resultado[j][1] = sum_sma / (j + 1);
        ema = close * k + ema * k_antes;
        resultado[j][2] = ema;

        float gain = 0.0f, loss = 0.0f;
        for (int p = 1; p <= j; p++) {
            float diff = serie[p] - serie[p - 1];
            if (diff > 0) gain += diff;
            else loss -= diff;
        }
        float rs = (loss == 0.0f) ? 0.0f : gain / loss;
        resultado[j][3] = (loss == 0.0f) ? 100.0f : (100.0f - (100.0f / (1.0f + rs)));

        float local_low = close, local_high = close;
        for (int k = 0; k <= j; k++) {
            float val = serie[k];
            if (val < local_low) local_low = val;
            if (val > local_high) local_high = val;
        }
        resultado[j][4] = (local_high == local_low) ? 50.0f : (100.0f * (close - local_low) / (local_high - local_low));
    }

    sum_gain /= period;
    sum_loss /= period;

    for (int j = period; j < size; j++) {
        float close = serie[j];
        resultado[j][0] = close;
        sum_sma += close - serie[j - period];
        resultado[j][1] = sum_sma / period;
        ema = close * k + ema * k_antes;
        resultado[j][2] = ema;

        float diff = close - serie[j - 1];
        float gain = (diff > 0) ? diff : 0.0f;
        float loss = (diff < 0) ? -diff : 0.0f;
        sum_gain = (sum_gain * (period - 1) + gain) / period;
        sum_loss = (sum_loss * (period - 1) + loss) / period;
        float rs = (sum_loss == 0.0f) ? 0.0f : (sum_gain / sum_loss);
        resultado[j][3] = (sum_loss == 0.0f) ? 100.0f : (100.0f - (100.0f / (1.0f + rs)));

        float val_out = serie[j - period];
        if (close >= high) high = close;
        else if (val_out == high) {
            high = serie[j - period + 1];
            for (int m = j - period + 2; m <= j; m++) if (serie[m] > high) high = serie[m];
        }
        if (close <= low) low = close;
        else if (val_out == low) {
            low = serie[j - period + 1];
            for (int m = j - period + 2; m <= j; m++) if (serie[m] < low) low = serie[m];
        }
        resultado[j][4] = (high == low) ? NAN : (100.0f * (close - low) / (high - low));
    }
}

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

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const int period = 14;
    
    if (rank == 0) {
        listar_csvs("empresas");
        
        // float series[MAX_FILES][MAX_SIZE];
        // int sizes[MAX_FILES];
        // float resultados[MAX_FILES][MAX_SIZE][RESULT_COLS];
            
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

        int size = nprocs;
        int files_per_proc = num_files / size;
        int remainder = num_files % size;

        struct timeval start, end;
        gettimeofday(&start, NULL);

        int offset = 0;
        for (int dest = 1; dest < size; dest++) {
            int count = files_per_proc + (dest < remainder ? 1 : 0);
            MPI_Send(&count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);

            for (int i = 0; i < count; i++) {
                int index = offset + i;
                MPI_Send(&sizes[index], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                MPI_Send(series[index], MAX_SIZE, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
            }
            offset += count;
        }

        // Process your own chunk (rank 0)
        int my_count = files_per_proc + (0 < remainder ? 1 : 0);
        for (int i = 0; i < my_count; i++) {
            int index = i;
            calcular_indicadores(series[index], sizes[index], resultados[index], period);
        }

        // Receive results from other processes
        offset = my_count;
        for (int src = 1; src < size; src++) {
            int count = files_per_proc + (src < remainder ? 1 : 0);
            for (int i = 0; i < count; i++) {
                int index = offset + i;
                MPI_Recv(resultados[index][0], MAX_SIZE * RESULT_COLS, MPI_FLOAT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            offset += count;
        }

        gettimeofday(&end, NULL);
        double tempo = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
        printf("Tempo cálculo indicadores: %.6f s\n", tempo);

        for (int i = 0; i < num_files; i++) {
            const char *nome = strrchr(filenames[i], '/');
            if (!nome) nome = filenames[i]; else nome++;
            char nome_empresa[64];
            strncpy(nome_empresa, nome, strchr(nome, '.') - nome);
            nome_empresa[strchr(nome, '.') - nome] = '\0';
            char output_path[256];
            snprintf(output_path, sizeof(output_path), "saida/mpi-%s.csv", nome_empresa);
            salvar_csv(output_path, resultados[i], sizes[i]);
        }
    } else {
        int my_count;
        MPI_Recv(&my_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < my_count; i++) {
            int size;
            float serie[MAX_SIZE];
            float resultado[MAX_SIZE][RESULT_COLS];

            MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(serie, MAX_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            calcular_indicadores(serie, size, resultado, period);

            MPI_Send(resultado, MAX_SIZE * RESULT_COLS, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
