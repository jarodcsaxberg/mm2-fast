# GPU Optimized mm2-fast
This code implements a GPU optimized version of mm2-fast. The goal was to 
optimize the RMI lookup process with the help of a GPU. To do this, the lookup
process was performed on the GPU in parallel.

## GPU Kernel
The kernel code can be found in the `test-kernel` directory. The code must first load the files ending with .uint64 and .rmi_PARAMETERS with the CPU to then transfer them to the GPU. The .uint64 file contains the sorted array that the RMI function will be checking if the keys are found within, and the .rmi_PARAMETERS file contains the determined parameters for the RMI model. Also necessary for the kernel is the actual keys to search for; this comes from the  file `test/input/positions.raw`. Using the `wbImport()` function to load the this file was tried, but this function does not support 64-bit values. Custom loading logic was developed which read the file line by line to create and set memory using the `std::stoull()` function to convert each line of the file to an unsigned long long integer. This is certainly a bottleneck to the GPU optimization and has room for improvement. 

The implemented kernel functions are tagged with `__global__` and `__device__` in the kernel.cu file. 

There are two slurm output files in the repo. One shows the determined position array with the GPU optimized code and the other shows the throughput calculations. Using the GPU position array, it can be compared with the valid output file. This is done in `tests/output` with the python file. 
