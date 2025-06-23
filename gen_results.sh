#/bin/bash
echo "RESULTADOS PARA OPENMP" > results.txt
for t in {1..16}
do
    echo "$t THREAD" >> results.txt
    export OMP_NUM_THREADS=$t
    for i in {1..10}
    do
        ./openmp_exec >> results.txt
    done
done

echo "RESULTADOS PARA OPENCL" >> results.txt
for i in {1..10}
do
    ./gpu_calc >> results.txt
done
echo "RESULTADOS PARA OPENCL 2D" >> results.txt
for i in {1..10}
do
    ./gpu_calc_2d >> results.txt
done
./

echo "RESULTADOS PARA CUDA SEQUENCIAL" >> results.txt
for i in {1..10}
do
    ./cudapuro >> results.txt
done
echo "RESULTADOS PARA CUDA 2D" >> results.txt
for i in {1..10}
do
    ./cudabatch >> results.txt
done

echo "RESULTADOS PARA MPI" >> results.txt
for j in {1..3} # Não funca com -np 4
do
    echo "$j PROCESSOS" >> results.txt
    for i in {1..10}
    do
        mpirun -np $j ./mpi_exec >> results.txt
    done
done