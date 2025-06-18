#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <sys/time.h>

#define MAX_SIZE 1300
#define MAX_FILES 500
#define FILENAME_LEN 128
#define RESULT_COLS 5

struct timeval start, end;

char filenames[MAX_FILES][FILENAME_LEN];
int num_files = 0;

void listar_csvs(const char *dirpath) {
    DIR *dir = opendir(dirpath);
    struct dirent *entry;
    if (!dir) { perror("Erro ao abrir diretório"); exit(1); }
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
    if (!fp) { perror("Erro ao abrir arquivo"); return -1; }
    char line[512];
    int count = 0;
    fgets(line, sizeof(line), fp);  // pular cabeçalho
    while (fgets(line, sizeof(line), fp) && count < max_size) {
        char *token;
        int col = 0;
        float close_val = 0.0f;
        token = strtok(line, ",");
        while (token != NULL) {
            if (col == 4) { // coluna Close
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

// Função para ler o arquivo .cl
char* read_kernel_source(const char* filename, size_t* length) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        perror("Erro ao abrir kernel");
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    *length = ftell(fp);
    rewind(fp);
    char* source = malloc(*length + 1);
    fread(source, 1, *length, fp);
    source[*length] = '\0';
    fclose(fp);
    return source;
}

void check_error(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s - Error code: %d\n", msg, err);
        exit(EXIT_FAILURE);
    }
}

void processar_empresa_gpu(float* data, int tamanho, float** resultados, cl_context context, cl_command_queue queue, cl_program program) {
    const int period = 14;
    cl_int err;

    // Buffers OpenCL
    cl_mem data_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tamanho * sizeof(float), data, &err);
    check_error(err, "Erro clCreateBuffer data");

    cl_mem sma_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tamanho * sizeof(float), NULL, &err);
    check_error(err, "Erro clCreateBuffer sma");

    cl_mem ema_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, tamanho * sizeof(float), NULL, &err);
    check_error(err, "Erro clCreateBuffer ema");

    cl_mem rsi_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tamanho * sizeof(float), NULL, &err);
    check_error(err, "Erro clCreateBuffer rsi");

    cl_mem stochk_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tamanho * sizeof(float), NULL, &err);
    check_error(err, "Erro clCreateBuffer stochk");

    // Criar kernels
    cl_kernel kernel_sma = clCreateKernel(program, "calc_sma", &err);
    check_error(err, "Erro clCreateKernel SMA");

    cl_kernel kernel_ema = clCreateKernel(program, "calc_ema", &err);
    check_error(err, "Erro clCreateKernel EMA");

    cl_kernel kernel_rsi = clCreateKernel(program, "calc_rsi", &err);
    check_error(err, "Erro clCreateKernel RSI");

    cl_kernel kernel_stochk = clCreateKernel(program, "calc_stochastic_k", &err);
    check_error(err, "Erro clCreateKernel StochK");

    // Setar argumentos SMA
    clSetKernelArg(kernel_sma, 0, sizeof(cl_mem), &data_buf);
    clSetKernelArg(kernel_sma, 1, sizeof(cl_mem), &sma_buf);
    clSetKernelArg(kernel_sma, 2, sizeof(int), &tamanho);
    clSetKernelArg(kernel_sma, 3, sizeof(int), &period);

    // Setar argumentos EMA
    clSetKernelArg(kernel_ema, 0, sizeof(cl_mem), &data_buf);
    clSetKernelArg(kernel_ema, 1, sizeof(cl_mem), &ema_buf);
    clSetKernelArg(kernel_ema, 2, sizeof(int), &tamanho);
    clSetKernelArg(kernel_ema, 3, sizeof(int), &period);

    // Setar argumentos RSI
    clSetKernelArg(kernel_rsi, 0, sizeof(cl_mem), &data_buf);
    clSetKernelArg(kernel_rsi, 1, sizeof(cl_mem), &rsi_buf);
    clSetKernelArg(kernel_rsi, 2, sizeof(int), &tamanho);
    clSetKernelArg(kernel_rsi, 3, sizeof(int), &period);

    // Setar argumentos Stochastic K
    clSetKernelArg(kernel_stochk, 0, sizeof(cl_mem), &data_buf);
    clSetKernelArg(kernel_stochk, 1, sizeof(cl_mem), &stochk_buf);
    clSetKernelArg(kernel_stochk, 2, sizeof(int), &tamanho);
    clSetKernelArg(kernel_stochk, 3, sizeof(int), &period);

    size_t global_work_size = tamanho;

    // Executar kernels
    err = clEnqueueNDRangeKernel(queue, kernel_sma, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    check_error(err, "Erro clEnqueueNDRangeKernel SMA");

    // Para o EMA, porque depende do valor anterior, calcularemos parcialmente na GPU e finalizaremos no CPU
    err = clEnqueueNDRangeKernel(queue, kernel_ema, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    check_error(err, "Erro clEnqueueNDRangeKernel EMA");

    err = clEnqueueNDRangeKernel(queue, kernel_rsi, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    check_error(err, "Erro clEnqueueNDRangeKernel RSI");

    err = clEnqueueNDRangeKernel(queue, kernel_stochk, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    check_error(err, "Erro clEnqueueNDRangeKernel StochK");

    clFinish(queue);

    // Ler resultados
    float *sma = malloc(tamanho * sizeof(float));
    float *ema = malloc(tamanho * sizeof(float));
    float *rsi = malloc(tamanho * sizeof(float));
    float *stochk = malloc(tamanho * sizeof(float));

    err = clEnqueueReadBuffer(queue, sma_buf, CL_TRUE, 0, tamanho * sizeof(float), sma, 0, NULL, NULL);
    check_error(err, "Erro clEnqueueReadBuffer SMA");

    err = clEnqueueReadBuffer(queue, ema_buf, CL_TRUE, 0, tamanho * sizeof(float), ema, 0, NULL, NULL);
    check_error(err, "Erro clEnqueueReadBuffer EMA");

    err = clEnqueueReadBuffer(queue, rsi_buf, CL_TRUE, 0, tamanho * sizeof(float), rsi, 0, NULL, NULL);
    check_error(err, "Erro clEnqueueReadBuffer RSI");

    err = clEnqueueReadBuffer(queue, stochk_buf, CL_TRUE, 0, tamanho * sizeof(float), stochk, 0, NULL, NULL);
    check_error(err, "Erro clEnqueueReadBuffer StochK");

    // Finalizar EMA sequencialmente (dependência temporal)
    for (int i = 1; i < tamanho; i++) {
        float k = 2.0f / (period + 1);
        ema[i] = data[i] * k + ema[i - 1] * (1 - k);
    }

    // Preencher resultados com os dados da GPU e EMA finalizado
    for (int i = 0; i < tamanho; i++) {
        resultados[i][0] = data[i];
        resultados[i][1] = sma[i];
        resultados[i][2] = ema[i];
        resultados[i][3] = rsi[i];
        resultados[i][4] = stochk[i];
    }

    // Liberar memória GPU
    clReleaseMemObject(data_buf);
    clReleaseMemObject(sma_buf);
    clReleaseMemObject(ema_buf);
    clReleaseMemObject(rsi_buf);
    clReleaseMemObject(stochk_buf);

    clReleaseKernel(kernel_sma);
    clReleaseKernel(kernel_ema);
    clReleaseKernel(kernel_rsi);
    clReleaseKernel(kernel_stochk);

    free(sma);
    free(ema);
    free(rsi);
    free(stochk);
}

int main() {
    const char *dir_empresas = "empresas";
    const char *dir_saida = "saida";

    listar_csvs(dir_empresas);

    float **series = malloc(num_files * sizeof(float*));
    int *sizes = malloc(num_files * sizeof(int));
    float ***resultados = malloc(num_files * sizeof(float**));
    for (int i = 0; i < num_files; i++) {
        series[i] = malloc(MAX_SIZE * sizeof(float));
        resultados[i] = malloc(MAX_SIZE * sizeof(float*));
        for (int j = 0; j < MAX_SIZE; j++) {
            resultados[i][j] = malloc(RESULT_COLS * sizeof(float));
        }
    }

    for (int i = 0; i < num_files; i++) {
        sizes[i] = load_csv(filenames[i], series[i], MAX_SIZE);
        if (sizes[i] < 0) {
            printf("Falha ao carregar %s\n", filenames[i]);
            sizes[i] = 0;
        }
    }

    // Setup OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "Erro clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check_error(err, "Erro clGetDeviceIDs");

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "Erro clCreateContext");

    queue = clCreateCommandQueue(context, device, 0, &err);
    check_error(err, "Erro clCreateCommandQueue");

    size_t kernel_source_size;
    char *kernel_source = read_kernel_source("indicadores.cl", &kernel_source_size);
    if (!kernel_source) exit(EXIT_FAILURE);

    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    check_error(err, "Erro clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Erro no build do kernel:\n%s\n", log);
        free(log);
        exit(EXIT_FAILURE);
    }

    free(kernel_source);

    gettimeofday(&start, NULL);

    for (int i = 0; i < num_files; i++) {
        if (sizes[i] > 0) {
            processar_empresa_gpu(series[i], sizes[i], resultados[i], context, queue, program);
        }
    }

    gettimeofday(&end, NULL);
    double tempo = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
    printf("Tempo cálculo indicadores na GPU: %.6f s\n", tempo);

    for (int i = 0; i < num_files; i++) {
        if (sizes[i] > 0) {
            const char *input_path = filenames[i];
            const char *nome_arquivo = strrchr(input_path, '/');
            if (!nome_arquivo) nome_arquivo = input_path;
            else nome_arquivo++;

            char nome_empresa[64];
            strncpy(nome_empresa, nome_arquivo, strchr(nome_arquivo, '.') - nome_arquivo);
            nome_empresa[strchr(nome_arquivo, '.') - nome_arquivo] = '\0';

            char output_path[256];
            snprintf(output_path, sizeof(output_path), "%s/sequencial-%s.csv", dir_saida, nome_empresa);

            FILE *out = fopen(output_path, "w");
            if (!out) {
                perror("Erro ao criar arquivo de saída");
                continue;
            }
            fprintf(out, "Index,Close,SMA,EMA,RSI,StochK\n");
            for (int j = 0; j < sizes[i]; j++) {
                char sma_buf[16] = "", rsi_buf[16] = "", stoch_buf[16] = "";
                if (!isnan(resultados[i][j][1])) sprintf(sma_buf, "%.2f", resultados[i][j][1]);
                if (!isnan(resultados[i][j][3])) sprintf(rsi_buf, "%.2f", resultados[i][j][3]);
                if (!isnan(resultados[i][j][4])) sprintf(stoch_buf, "%.2f", resultados[i][j][4]);

                fprintf(out, "%d,%.2f,%s,%.2f,%s,%s\n",
                    j,
                    resultados[i][j][0],
                    sma_buf,
                    resultados[i][j][2],
                    rsi_buf,
                    stoch_buf);
            }
            fclose(out);
        }
    }

    for (int i = 0; i < num_files; i++) {
        free(series[i]);
        for (int j = 0; j < MAX_SIZE; j++)
            free(resultados[i][j]);
        free(resultados[i]);
    }
    free(series);
    free(resultados);
    free(sizes);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
