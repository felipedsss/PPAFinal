#/bin/bash
echo "RESULTADOS PARA OPENMP" > results.txt
for t in {1..8}
do
    echo "$t THREAD" >> results.txt
    export OMP_NUM_THREADS=$t
    for i in {1..10}
    do
        ./openmp_exec >> results.txt
    done
done
echo "RESULTADOS PARA MPI" >> results.txt
for i in {1..10}
do
    mpirun -np 4 ./mpi_exec >> results.txt
done
echo "RESULTADOS PARA CUDA" >> results.txt
for i in {1..10}
do
    ./cudapuro >> results.txt
done
echo "RESULTADOS PARA OPENCL" >> results.txt
for i in {1..10}
do
    ./gpu_calc >> results.txt
done
