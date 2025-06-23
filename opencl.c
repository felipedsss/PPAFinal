#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <sys/time.h>

#define MAX_SIZE 1300
#define MAX_FILES 5000
#define FILENAME_LEN 128
#define RESULT_COLS 5

struct timeval start, end;

char filenames[MAX_FILES][FILENAME_LEN];
int num_files = 0;

const char *kernel_source =
"__kernel void indicadores(\n"
"    __global const float* data,\n"
"    __global float* sma,\n"
"    __global float* ema,\n"
"    __global float* rsi,\n"
"    __global float* stochk,\n"
"    const int tamanho,\n"
"    const int period) {\n"
"    int i = get_global_id(0);\n"
"    if (i >= tamanho) return;\n"
"\n"
"    float sum = 0.0f;\n"
"    int count = 0;\n"
"    for (int j = i - period + 1; j <= i; j++) {\n"
"        if (j >= 0) {\n"
"            sum += data[j];\n"
"            count++;\n"
"        }\n"
"    }\n"
"    sma[i] = (count > 0) ? (sum / count) : NAN;\n"
"\n"
"    ema[i] = data[i];\n"
"\n"
"    float gain = 0.0f, loss = 0.0f;\n"
"    int valid = 0;\n"
"    for (int j = i - period + 1; j <= i; j++) {\n"
"        if (j > 0) {\n"
"            float diff = data[j] - data[j - 1];\n"
"            if (diff > 0) gain += diff;\n"
"            else loss -= diff;\n"
"            valid++;\n"
"        }\n"
"    }\n"
"    float rs = (loss == 0.0f) ? 0.0f : gain / loss;\n"
"    rsi[i] = (valid < period) ? NAN : (100.0f - (100.0f / (1.0f + rs)));\n"
"\n"
"    float low = data[i], high = data[i];\n"
"    for (int j = i - period + 1; j <= i; j++) {\n"
"        if (j >= 0) {\n"
"            float val = data[j];\n"
"            if (val < low) low = val;\n"
"            if (val > high) high = val;\n"
"        }\n"
"    }\n"
"    stochk[i] = (high == low) ? NAN : (100.0f * (data[i] - low) / (high - low));\n"
"}\n";

void listar_csvs(const char *dirpath) {
    DIR *dir = opendir(dirpath);
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (strstr(entry->d_name, ".csv")) {
            snprintf(filenames[num_files], FILENAME_LEN, "%s/%s", dirpath, entry->d_name);
            num_files++;
        }
    }
    closedir(dir);
}

int load_csv(const char *filename, float *data, int max_size) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;
    char line[512];
    int count = 0;
    fgets(line, sizeof(line), fp);
    while (fgets(line, sizeof(line), fp) && count < max_size) {
        char *token;
        int col = 0;
        token = strtok(line, ",");
        while (token != NULL) {
            if (col == 4) {
                data[count++] = strtof(token, NULL);
                break;
            }
            token = strtok(NULL, ",");
            col++;
        }
    }
    fclose(fp);
    return count;
}

void salvar_csv(const char *path, float **resultados, int size) {
    FILE *out = fopen(path, "w");
    fprintf(out, "Index,Close,SMA,EMA,RSI,StochK\n");
    for (int i = 0; i < size; i++) {
        char sma_buf[16] = "", rsi_buf[16] = "", stoch_buf[16] = "";
        if (!isnan(resultados[i][1])) sprintf(sma_buf, "%.2f", resultados[i][1]);
        if (!isnan(resultados[i][3])) sprintf(rsi_buf, "%.2f", resultados[i][3]);
        if (!isnan(resultados[i][4])) sprintf(stoch_buf, "%.2f", resultados[i][4]);

        fprintf(out, "%d,%.2f,%s,%.2f,%s,%s\n",
            i, resultados[i][0], sma_buf, resultados[i][2], rsi_buf, stoch_buf);
    }
    fclose(out);
}

void check_error(cl_int err, const char* msg) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "%s - Error code: %d\n", msg, err);
        exit(EXIT_FAILURE);
    }
}

void processar_gpu(float* data, int tamanho, float** resultados, cl_context ctx, cl_command_queue queue, cl_program prog) {
    const int period = 14;
    cl_int err;
    size_t global = tamanho;

    cl_mem buf_data = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tamanho * sizeof(float), data, &err);
    cl_mem buf_sma = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, tamanho * sizeof(float), NULL, &err);
    cl_mem buf_ema = clCreateBuffer(ctx, CL_MEM_READ_WRITE, tamanho * sizeof(float), NULL, &err);
    cl_mem buf_rsi = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, tamanho * sizeof(float), NULL, &err);
    cl_mem buf_stoch = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, tamanho * sizeof(float), NULL, &err);

    cl_kernel kernel = clCreateKernel(prog, "indicadores", &err);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_data);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_sma);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_ema);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_rsi);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_stoch);
    clSetKernelArg(kernel, 5, sizeof(int), &tamanho);
    clSetKernelArg(kernel, 6, sizeof(int), &period);

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(queue);

    float *sma = malloc(tamanho * sizeof(float));
    float *ema = malloc(tamanho * sizeof(float));
    float *rsi = malloc(tamanho * sizeof(float));
    float *stoch = malloc(tamanho * sizeof(float));

    clEnqueueReadBuffer(queue, buf_sma, CL_TRUE, 0, tamanho * sizeof(float), sma, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, buf_ema, CL_TRUE, 0, tamanho * sizeof(float), ema, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, buf_rsi, CL_TRUE, 0, tamanho * sizeof(float), rsi, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, buf_stoch, CL_TRUE, 0, tamanho * sizeof(float), stoch, 0, NULL, NULL);

    for (int i = 1; i < tamanho; i++) {
        float k = 2.0f / (period + 1);
        ema[i] = data[i] * k + ema[i - 1] * (1 - k);
    }

    for (int i = 0; i < tamanho; i++) {
        resultados[i][0] = data[i];
        resultados[i][1] = sma[i];
        resultados[i][2] = ema[i];
        resultados[i][3] = rsi[i];
        resultados[i][4] = stoch[i];
    }

    clReleaseMemObject(buf_data);
    clReleaseMemObject(buf_sma);
    clReleaseMemObject(buf_ema);
    clReleaseMemObject(buf_rsi);
    clReleaseMemObject(buf_stoch);
    clReleaseKernel(kernel);

    free(sma); free(ema); free(rsi); free(stoch);
}

int main() {
    const char *dir_empresas = "empresas";
    const char *dir_saida = "saida";

    listar_csvs(dir_empresas);

    float **series = malloc(MAX_FILES * sizeof(float*));
    float ***resultados = malloc(MAX_FILES * sizeof(float**));
    int sizes[MAX_FILES];

    for (int i = 0; i < num_files; i++) {
        series[i] = malloc(MAX_SIZE * sizeof(float));
        resultados[i] = malloc(MAX_SIZE * sizeof(float*));
        for (int j = 0; j < MAX_SIZE; j++)
            resultados[i][j] = malloc(RESULT_COLS * sizeof(float));
        sizes[i] = load_csv(filenames[i], series[i], MAX_SIZE);
    }

    cl_int err;
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);
    program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    gettimeofday(&start, NULL);
    for (int i = 0; i < num_files; i++) {
        if (sizes[i] > 0)
            processar_gpu(series[i], sizes[i], resultados[i], context, queue, program);
    }
    gettimeofday(&end, NULL);
    double tempo = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec)/1e6;
    printf("Tempo GPU: %.6f s\n", tempo);
    /*
    for (int i = 0; i < num_files; i++) {
        if (sizes[i] > 0) {
            const char *nome_arquivo = strrchr(filenames[i], '/');
            nome_arquivo = nome_arquivo ? nome_arquivo + 1 : filenames[i];
            char nome_empresa[64];
            strncpy(nome_empresa, nome_arquivo, strchr(nome_arquivo, '.') - nome_arquivo);
            nome_empresa[strchr(nome_arquivo, '.') - nome_arquivo] = '\0';

            char output_path[256];
            snprintf(output_path, sizeof(output_path), "%s/opencl-%s.csv", dir_saida, nome_empresa);
            salvar_csv(output_path, resultados[i], sizes[i]);
        }
    }
    */
    for (int i = 0; i < num_files; i++) {
        free(series[i]);
        for (int j = 0; j < MAX_SIZE; j++) free(resultados[i][j]);
        free(resultados[i]);
    }
    free(series); free(resultados);

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}
