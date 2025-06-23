#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <math.h>
#include <sys/time.h>

#define MAX_SIZE 1300
#define MAX_FILES 5000
#define FILENAME_LEN 128
#define RESULT_COLS 5
#define PERIOD 14

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

void salvar_csv(const char *path, float *resultados, int size) {
    FILE *out = fopen(path, "w");
    if (!out) {
        perror("Erro ao criar arquivo de saída");
        return;
    }
    fprintf(out, "Index,Close,SMA,EMA,RSI,StochK\n");
    for (int i = 0; i < size; i++) {
        fprintf(out, "%d,%.2f,%.2f,%.2f,%.2f,%.2f\n",
            i,
            resultados[i * RESULT_COLS + 0],
            resultados[i * RESULT_COLS + 1],
            resultados[i * RESULT_COLS + 2],
            resultados[i * RESULT_COLS + 3],
            resultados[i * RESULT_COLS + 4]);
    }
    fclose(out);
}

void check_error(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s - Error code: %d\n", msg, err);
        exit(EXIT_FAILURE);
    }
}

// Kernel OpenCL (como string aqui pra simplicidade)
const char* kernel_source = "\
__kernel void indicadores_kernel(__global const float* series, __global const int* sizes, __global float* resultados, \
                                 const int max_size, const int result_cols, const int period, const int num_files) { \
    int i = get_global_id(0); \
    int j = get_global_id(1); \
    if (i >= num_files || j >= max_size) return; \
    int size = sizes[i]; \
    if (size <= 0) return; \
    if (j == 0) { \
        float k = 2.0f / (period + 1); \
        float k_antes = 1.0f - k; \
        float ema = series[i * max_size]; \
        float sum_sma = series[i * max_size]; \
        float sum_gain = 0.0f, sum_loss = 0.0f; \
        float low = series[i * max_size]; \
        float high = series[i * max_size]; \
        resultados[i * max_size * result_cols + 0 * result_cols + 0] = series[i * max_size]; \
        resultados[i * max_size * result_cols + 0 * result_cols + 1] = series[i * max_size]; \
        resultados[i * max_size * result_cols + 0 * result_cols + 2] = series[i * max_size]; \
        resultados[i * max_size * result_cols + 0 * result_cols + 3] = 50.0f; \
        resultados[i * max_size * result_cols + 0 * result_cols + 4] = 50.0f; \
    } else if (j < size) { \
        float close = series[i * max_size + j]; \
        float sum_sma = 0.0f; \
        float ema = 0.0f; \
        float sum_gain = 0.0f, sum_loss = 0.0f; \
        for (int x = 0; x <= j; x++) { \
            sum_sma += series[i * max_size + x]; \
        } \
        resultados[i * max_size * result_cols + j * result_cols + 0] = close; \
        resultados[i * max_size * result_cols + j * result_cols + 1] = sum_sma / (j + 1); \
        resultados[i * max_size * result_cols + j * result_cols + 2] = close; /* EMA não calculado aqui */ \
        resultados[i * max_size * result_cols + j * result_cols + 3] = 50.0f; /* RSI simplificado */ \
        resultados[i * max_size * result_cols + j * result_cols + 4] = 50.0f; /* StochK simplificado */ \
    } \
}"; // <-- FECHA a função e a string

int main() {
    const char *dir_empresas = "empresas";
    const char *dir_saida = "saida";

    listar_csvs(dir_empresas);

    if (num_files == 0) {
        printf("Nenhum arquivo CSV encontrado.\n");
        return 1;
    }

    // Alocar arrays para dados
    float *h_series = malloc(num_files * MAX_SIZE * sizeof(float));
    int *h_sizes = malloc(num_files * sizeof(int));
    float *h_resultados = malloc(num_files * MAX_SIZE * RESULT_COLS * sizeof(float));

    for (int i = 0; i < num_files; i++) {
        float tmp[MAX_SIZE] = {0};
        int tam = load_csv(filenames[i], tmp, MAX_SIZE);
        if (tam < 0) {
            fprintf(stderr, "Erro ao carregar %s\n", filenames[i]);
            tam = 0;
        }
        h_sizes[i] = tam;
        memcpy(&h_series[i * MAX_SIZE], tmp, MAX_SIZE * sizeof(float));
    }

    // Setup OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "Erro clGetPlatformIDs");

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    check_error(err, "Erro clGetDeviceIDs");

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "Erro clCreateContext");

    queue = clCreateCommandQueue(context, device, 0, &err);
    check_error(err, "Erro clCreateCommandQueue");

    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
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

    kernel = clCreateKernel(program, "indicadores_kernel", &err);
    check_error(err, "Erro clCreateKernel");

    cl_mem buffer_series = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          num_files * MAX_SIZE * sizeof(float), h_series, &err);
    check_error(err, "Erro clCreateBuffer series");

    cl_mem buffer_sizes = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         num_files * sizeof(int), h_sizes, &err);
    check_error(err, "Erro clCreateBuffer sizes");

    cl_mem buffer_resultados = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                              num_files * MAX_SIZE * RESULT_COLS * sizeof(float), NULL, &err);
    check_error(err, "Erro clCreateBuffer resultados");

    // Passar argumentos para o kernel
    int max_size = MAX_SIZE;
    int result_cols = RESULT_COLS;
    int period = PERIOD;

    check_error(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_series), "arg0");
    check_error(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_sizes), "arg1");
    check_error(clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_resultados), "arg2");
    check_error(clSetKernelArg(kernel, 3, sizeof(int), &max_size), "arg3");
    check_error(clSetKernelArg(kernel, 4, sizeof(int), &result_cols), "arg4");
    check_error(clSetKernelArg(kernel, 5, sizeof(int), &period), "arg5");
    check_error(clSetKernelArg(kernel, 6, sizeof(int), &num_files), "arg6");

    size_t global_work_size[2] = { (size_t)num_files, (size_t)MAX_SIZE };

    gettimeofday(&start, NULL);

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    check_error(err, "Erro clEnqueueNDRangeKernel");

    clFinish(queue);

    gettimeofday(&end, NULL);
    double tempo = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Tempo cálculo indicadores na GPU: %.6f s\n", tempo);

    err = clEnqueueReadBuffer(queue, buffer_resultados, CL_TRUE, 0,
                              num_files * MAX_SIZE * RESULT_COLS * sizeof(float), h_resultados, 0, NULL, NULL);
    check_error(err, "Erro clEnqueueReadBuffer resultados");
    /*
    // Salvar resultados por arquivo
    for (int i = 0; i < num_files; i++) {
        if (h_sizes[i] > 0) {
            const char *nome_arquivo = strrchr(filenames[i], '/');
            nome_arquivo = nome_arquivo ? nome_arquivo + 1 : filenames[i];
            char nome_empresa[64];
            strncpy(nome_empresa, nome_arquivo, strchr(nome_arquivo, '.') - nome_arquivo);
            nome_empresa[strchr(nome_arquivo, '.') - nome_arquivo] = '\0';

            char output_path[256];
            snprintf(output_path, sizeof(output_path), "%s/opencl2d-%s.csv", dir_saida, nome_empresa);

            salvar_csv(output_path, &h_resultados[i * MAX_SIZE * RESULT_COLS], h_sizes[i]);
        }
    }
    */
    // Cleanup
    clReleaseMemObject(buffer_series);
    clReleaseMemObject(buffer_sizes);
    clReleaseMemObject(buffer_resultados);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(h_series);
    free(h_sizes);
    free(h_resultados);

    return 0;
}
