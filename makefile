# Nome do executável
EXEC = programa

# Fontes
SRC_C = main.c
SRC_CU = sma_kernel.cu rsi_kernel.cu stochastic_kernel.cu ema_kernel.cu



# Objetos
OBJ_C = $(SRC_C:.c=.o)
OBJ_CU = $(SRC_CU:.cu=.o)

# Compiladores
MPICC = mpicc
NVCC = nvcc

# Flags
CFLAGS = -fopenmp -Wall
CUDA_INC = -I/opt/cuda/include
CUDA_LIB = -L/opt/cuda/targets/x86_64-linux/lib -lcudart

# Regra padrão
all: $(EXEC)

# Compilar código CUDA
%.o: %.cu
	$(NVCC) -c $< -o $@

# Compilar e linkar
$(EXEC): $(OBJ_CU) $(OBJ_C)
	$(MPICC) $(CFLAGS) $(OBJ_C) $(OBJ_CU) -o $(EXEC) $(CUDA_INC) $(CUDA_LIB)

# Limpar
clean:
	rm -f *.o $(EXEC)
