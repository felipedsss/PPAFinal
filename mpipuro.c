#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <mpi.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_SIZE 1300
#define MAX_FILES 500
#define FILENAME_LEN 512
#define MAX_NAME 128

typedef struct {
    float sma;
    float ema;
    float rsi;
    float stoch;
} Indicador;

char filenames[MAX_FILES][FILENAME_LEN];
int num_files = 0;

int comparar_nomes(const void *a, const void *b) {
    const char *pa = (const char *)a;
    const char *pb = (const char *)b;
    return strcmp(pa, pb);
}

int listar_csvs(const char *dirpath, char nomes[MAX_FILES][MAX_NAME]) {
    DIR *dir = opendir(dirpath);
    struct dirent *entry;
    int count = 0;

    if (!dir) {
        perror("Erro ao abrir diretório");
        exit(1);
    }

    while ((entry = readdir(dir))) {
        if (entry->d_name[0] == '.') continue; // Ignora ocultos

        if (strstr(entry->d_name, ".csv")) {
            if (count >= MAX_FILES) break;
            snprintf(nomes[count], MAX_NAME, "%s", entry->d_name);
            snprintf(filenames[count], FILENAME_LEN, "%s/%s", dirpath, entry->d_name);
            count++;
        }
    }

    closedir(dir);
    qsort(nomes, count, MAX_NAME, comparar_nomes);
    qsort(filenames, count, FILENAME_LEN, comparar_nomes);
    num_files = count;
    return count;
}

int load_csv(const char *filename, float *data, int max_size) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        perror(filename); // Mostra nome do arquivo com erro
        return -1;
    }

    char line[512];
    int count = 0;
    fgets(line, sizeof(line), fp); // Ignora cabeçalho

    while (fgets(line, sizeof(line), fp) && count < max_size) {
        char *token = strtok(line, ",");
        int column = 0;

        while (token) {
            if (column == 4) {
                data[count++] = strtof(token, NULL);
                break;
            }
            token = strtok(NULL, ",");
            column++;
        }
    }

    fclose(fp);
    return count;
}

// Indicadores
float calc_sma(float *data, int idx, int period) {
    if (idx < period - 1) return NAN;
    float sum = 0.0f;
    for (int i = idx - period + 1; i <= idx; i++)
        sum += data[i];
    return sum / period;
}

float calc_ema(float *data, int idx, int period, float prev_ema) {
    float k = 2.0f / (period + 1);
    return data[idx] * k + prev_ema * (1 - k);
}

float calc_rsi(float *data, int idx, int period) {
    if (idx < period) return NAN;
    float gain = 0.0f, loss = 0.0f;
    for (int i = idx - period + 1; i <= idx; i++) {
        float diff = data[i] - data[i - 1];
        if (diff > 0) gain += diff;
        else loss -= diff;
    }
    if (loss == 0) return 100.0f;
    float rs = gain / loss;
    return 100.0f - (100.0f / (1.0f + rs));
}

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

void processar(float *serie, int size, Indicador *out) {
    int period = 14;
    float ema = serie[0];
    for (int i = 1; i < size; i++) {
        out[i].sma = calc_sma(serie, i, period);
        ema = calc_ema(serie, i, period, ema);
        out[i].ema = ema;
        out[i].rsi = calc_rsi(serie, i, period);
        out[i].stoch = calc_stochastic_k(serie, i, period);
    }
}

// Tags
#define TAG_TAM    1
#define TAG_SERIE  2
#define TAG_NOME   3
#define TAG_RESULT 4
#define TAG_TEMPO  5
#define TAG_FIM    99

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        char nomes[MAX_FILES][MAX_NAME];
        int num_arquivos = listar_csvs("empresas", nomes);
        int enviados = 0, recebidos = 0;
        double tempo_total = 0.0;

        for (int i = 1; i < size && enviados < num_arquivos; i++) {
            float serie[MAX_SIZE];
            int tam = load_csv(filenames[enviados], serie, MAX_SIZE);
            if (tam <= 0) {
                enviados++;
                i--;
                continue;
            }
            MPI_Send(&tam, 1, MPI_INT, i, TAG_TAM, MPI_COMM_WORLD);
            MPI_Send(serie, tam, MPI_FLOAT, i, TAG_SERIE, MPI_COMM_WORLD);
            MPI_Send(nomes[enviados], MAX_NAME, MPI_CHAR, i, TAG_NOME, MPI_COMM_WORLD);
            enviados++;
        }

        while (recebidos < num_arquivos) {
            int tam;
            double tempo;
            Indicador indicadores[MAX_SIZE];
            char nome[MAX_NAME];
            MPI_Status status;

            MPI_Recv(&tam, 1, MPI_INT, MPI_ANY_SOURCE, TAG_TAM, MPI_COMM_WORLD, &status);
            int origem = status.MPI_SOURCE;
            MPI_Recv(indicadores, tam * sizeof(Indicador), MPI_BYTE, origem, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(nome, MAX_NAME, MPI_CHAR, origem, TAG_NOME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&tempo, 1, MPI_DOUBLE, origem, TAG_TEMPO, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            tempo_total += tempo;

            char caminho[512];
            snprintf(caminho, sizeof(caminho), "saida/mpi-%s", nome);
            FILE *f = fopen(caminho, "w");
            if (!f) {
                perror(caminho);
                continue;
            }

            fprintf(f, "Index,SMA,EMA,RSI,StochK\n");
            for (int j = 1; j < tam; j++) {
                fprintf(f, "%d,%.2f,%.2f,%.2f,%.2f\n", j, indicadores[j].sma, indicadores[j].ema, indicadores[j].rsi, indicadores[j].stoch);
            }
            fclose(f);
            recebidos++;

            if (enviados < num_arquivos) {
                float serie[MAX_SIZE];
                int tam = load_csv(filenames[enviados], serie, MAX_SIZE);
                if (tam <= 0) {
                    enviados++;
                    continue;
                }
                MPI_Send(&tam, 1, MPI_INT, origem, TAG_TAM, MPI_COMM_WORLD);
                MPI_Send(serie, tam, MPI_FLOAT, origem, TAG_SERIE, MPI_COMM_WORLD);
                MPI_Send(nomes[enviados], MAX_NAME, MPI_CHAR, origem, TAG_NOME, MPI_COMM_WORLD);
                enviados++;
            } else {
                int zero = 0;
                MPI_Send(&zero, 1, MPI_INT, origem, TAG_FIM, MPI_COMM_WORLD);
            }
        }

        printf("Tempo total dos workers: %.6f segundos\n", tempo_total);
    } else {
        while (1) {
            int tam;
            MPI_Status status;
            MPI_Recv(&tam, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_FIM || tam == 0) break;

            float serie[MAX_SIZE];
            char nome[MAX_NAME];
            MPI_Recv(serie, tam, MPI_FLOAT, 0, TAG_SERIE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(nome, MAX_NAME, MPI_CHAR, 0, TAG_NOME, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            Indicador indicadores[MAX_SIZE];
            double start = MPI_Wtime();
            processar(serie, tam, indicadores);
            double elapsed = MPI_Wtime() - start;

            MPI_Send(&tam, 1, MPI_INT, 0, TAG_TAM, MPI_COMM_WORLD);
            MPI_Send(indicadores, tam * sizeof(Indicador), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
            MPI_Send(nome, MAX_NAME, MPI_CHAR, 0, TAG_NOME, MPI_COMM_WORLD);
            MPI_Send(&elapsed, 1, MPI_DOUBLE, 0, TAG_TEMPO, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;
}
