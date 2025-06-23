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
struct timeval start, end;
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

    int files_per_proc, remainder, my_count;

    if (rank == 0) listar_csvs("empresas");
    MPI_Bcast(&num_files, 1, MPI_INT, 0, MPI_COMM_WORLD);

    files_per_proc = num_files / nprocs;
    remainder = num_files % nprocs;
    my_count = files_per_proc + (rank < remainder ? 1 : 0);

    float *buffer_local = malloc(my_count * MAX_SIZE * RESULT_COLS * sizeof(float));
    float **series = malloc(my_count * sizeof(float*));
    int *sizes = malloc(my_count * sizeof(int));

    for (int i = 0; i < my_count; i++) series[i] = malloc(MAX_SIZE * sizeof(float));

    if (rank == 0) {
        int index = 0;
        for (int dest = 0; dest < nprocs; dest++) {
            int count = files_per_proc + (dest < remainder ? 1 : 0);
            for (int i = 0; i < count; i++, index++) {
                float temp[MAX_SIZE];
                int tam = load_csv(filenames[index], temp, MAX_SIZE);
                if (dest == 0) {
                    memcpy(series[i], temp, MAX_SIZE * sizeof(float));
                    sizes[i] = tam;
                } else {
                    MPI_Send(&tam, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
                    MPI_Send(temp, MAX_SIZE, MPI_FLOAT, dest, 0, MPI_COMM_WORLD);
                }
            }

            gettimeofday(&start, NULL);
        }
    } else {
        for (int i = 0; i < my_count; i++) {
            MPI_Recv(&sizes[i], 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(series[i], MAX_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    for (int i = 0; i < my_count; i++) {
        int size = sizes[i];
        if (size <= 0) {
            printf("[RANK %d] Série %d inválida (size=%d), pulando...\n", rank, i, size);
            continue;
        }

        float resultado[MAX_SIZE][RESULT_COLS];
        float *serie = series[i];

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

        for (int j = 0; j < size; j++) {
            for (int k = 0; k < RESULT_COLS; k++) {
                buffer_local[i * MAX_SIZE * RESULT_COLS + j * RESULT_COLS + k] = resultado[j][k];
            }
        }
    }

    // Coleta dos tamanhos reais
    int *sizes_global = NULL;
    if (rank == 0) sizes_global = malloc(num_files * sizeof(int));

    MPI_Gather(sizes, my_count, MPI_INT, sizes_global, my_count, MPI_INT, 0, MPI_COMM_WORLD);

    // Gatherv
    int *recvcounts = NULL, *displs = NULL;
    float *buffer_global = NULL;
    if (rank == 0) {
        recvcounts = malloc(nprocs * sizeof(int));
        displs = malloc(nprocs * sizeof(int));

        int offset = 0, index = 0;
        for (int i = 0; i < nprocs; i++) {
            int count = files_per_proc + (i < remainder ? 1 : 0);
            int total = 0;
            for (int j = 0; j < count; j++, index++) {
                total += sizes_global[index] * RESULT_COLS;
            }
            recvcounts[i] = total;
            displs[i] = offset;
            offset += total;
        }
        buffer_global = malloc(offset * sizeof(float));
    }

    int sendcount_local = 0;
    for (int i = 0; i < my_count; i++) sendcount_local += sizes[i] * RESULT_COLS;

    MPI_Gatherv(buffer_local, sendcount_local, MPI_FLOAT,
                buffer_global, recvcounts, displs, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        int index = 0;
        gettimeofday(&end, NULL);
            double tempo = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
            printf("Tempo cálculo indicadores: %.6f s\n", tempo);

        for (int i = 0; i < num_files; i++) {
            float **resultado = malloc(sizes_global[i] * sizeof(float*));
            for (int j = 0; j < sizes_global[i]; j++) {
                resultado[j] = malloc(RESULT_COLS * sizeof(float));
                for (int k = 0; k < RESULT_COLS; k++) {
                    resultado[j][k] = buffer_global[index++];
                }
            }
            gettimeofday(&end, NULL);
            const char *nome = strrchr(filenames[i], '/');
            if (!nome) nome = filenames[i]; else nome++;
            char nome_empresa[64];
            const char *dot = strchr(nome, '.');
            size_t len = dot ? (size_t)(dot - nome) : strlen(nome);
            if (len >= sizeof(nome_empresa)) len = sizeof(nome_empresa) - 1;
            strncpy(nome_empresa, nome, len);
            nome_empresa[len] = '\0';

            char output_path[256];
            snprintf(output_path, sizeof(output_path), "saida/mpi-%s.csv", nome_empresa);
            salvar_csv(output_path, resultado, sizes_global[i]);

            for (int j = 0; j < sizes_global[i]; j++) free(resultado[j]);
            free(resultado);
        }
        free(buffer_global);
        free(sizes_global);
        free(recvcounts);
        free(displs);
    }

    for (int i = 0; i < my_count; i++) free(series[i]);
    free(series);
    free(buffer_local);
    free(sizes);

    MPI_Finalize();
    return 0;
}
