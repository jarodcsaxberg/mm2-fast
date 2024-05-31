#include <wb.h>
#include <hip/hip_runtime.h>
#include <inttypes.h>
#include <fstream>
#include <immintrin.h>
#include <sys/mman.h>
#include <stdio.h>
#include <stdlib.h>
//#include "LISA-hash/lisa_hash.h"

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    hipError_t err = stmt;                                                \
    if (err != hipSuccess) {                                              \
      wbLog(ERROR, "HIP error: ", hipGetErrorString(err));                \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define THREAD_SIZE 256

typedef uint64_t rmi_key_t;

enum query_state {
    GUESS_RMI_ROOT,
    GUESS_RMI_LEAF,
    LAST_MILE
};

typedef struct batchMetadata {
  int64_t qid;
  query_state state;
  rmi_key_t key;
  int64_t modelIndex;
  int64_t first;
  int64_t m;
} BatchMetadata;

//@@ Define constant memory for device kernel here
__constant__ rmi_key_t constant_sorted_array[65536 / sizeof(rmi_key_t)];

//@@ Helper functions for the rmi lookup
__device__ int64_t FCLAMP(double inp, double bound) {
  if (inp < 0.0) return 0;
  return (inp > bound ? bound : (size_t)inp);
}

__device__ int64_t get_guess_root_step(rmi_key_t key, double L0_PARAMETER0, double L0_PARAMETER1, int64_t L1_SIZE) {
  int64_t modelIndex;
  double fpred = std::fma(L0_PARAMETER1, key, L0_PARAMETER0);
  modelIndex = FCLAMP(fpred, L1_SIZE - 1.0);
  return modelIndex;
}

__device__ int64_t get_guess_leaf_step(rmi_key_t key, int64_t modelIndex, int64_t *err, int64_t n, double* L1_PARAMETERS) {
  double fpred = std::fma(L1_PARAMETERS[modelIndex * 3 + 1], key, L1_PARAMETERS[modelIndex * 3]);
  *err = *((uint64_t*) (L1_PARAMETERS + (modelIndex * 3 + 2)));
  int64_t guess = FCLAMP(fpred, n - 1.0);
  return guess;
}

__device__ void last_mile_search_one_step(rmi_key_t key, int64_t &first, int64_t &m) {
  int64_t half = m >> 1;
  int64_t middle = first + half;
  int64_t cond = (key >= constant_sorted_array[middle]);
  first = middle * cond + first * (1 - cond);
  m = (m - half) * cond + half * (1 - cond);
}

__device__ int process_query_one_step(
  BatchMetadata *bm,
  int64_t *pos, 
  int64_t n,
  double L0_PARAMETER0, 
  double L0_PARAMETER1, 
  int64_t L1_SIZE,
  double *L1_PARAMETERS
) {
  if(bm->state == GUESS_RMI_ROOT){
    bm->modelIndex = get_guess_root_step(bm->key, L0_PARAMETER0, L0_PARAMETER1, L1_SIZE);
    bm->state = GUESS_RMI_LEAF;
    // _mm_prefetch((const char *)(&L1_PARAMETERS[bm->modelIndex * 3]), _MM_HINT_T0);
    // _mm_prefetch((const char *)(&L1_PARAMETERS[bm->modelIndex * 3 + 2]), _MM_HINT_T0);
  } else if(bm->state == GUESS_RMI_LEAF) {
    int64_t err;
    int64_t guess = get_guess_leaf_step(bm->key, bm->modelIndex, &err, n, L1_PARAMETERS);
    bm->first = guess - err;
    if(bm->first < 0) bm->first = 0;
    int64_t last = guess + err + 1;
    if(last > n) last = n;
    bm->m = last - bm->first;
    bm->state = LAST_MILE;
    int64_t middle = bm->m >> 1;
    //_mm_prefetch((const char *)(&sorted_array[bm->first + middle]), _MM_HINT_T0);
  } else {
    if(bm->m > 1)
    {
      last_mile_search_one_step(bm->key, bm->first, bm->m);
      int64_t middle = bm->m >> 1;
      //_mm_prefetch((const char *)(&sorted_array[bm->first + middle]), _MM_HINT_T0);
    }
    if(bm->m == 1)
    {
      *pos = bm->first;

      if(constant_sorted_array[*pos] != bm->key)
        *pos = -1;
        
        return 0;
    }
  }
  return 1;
}

//@@ Insert kernel code here
__global__ void rmi_lookup(
  rmi_key_t *inputKeys, 
  double *inputL1_PARAMETERS,
  int64_t *outputPositions, 
  int64_t n,
  double L0_PARAMETER0, 
  double L0_PARAMETER1, 
  int64_t L1_SIZE
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < n) {
    BatchMetadata bm;
    bm.qid = tid;
    bm.state = GUESS_RMI_ROOT;
    bm.key = inputKeys[tid];
    int64_t pos;
    int status = 1;

    //printf("tid %" PRId64 ", bm.state %d, bm.key %" PRIu64 "\n", bm.qid, bm.state, bm.key);
    do {
      status = process_query_one_step(&bm, &pos, n, L0_PARAMETER0, L0_PARAMETER1, L1_SIZE, inputL1_PARAMETERS);
    } while(status);

    outputPositions[tid] = pos;
    //printf("tid %d done, pos = %" PRId64 "\n", tid, pos);
  }
}

bool load_sorted_array(char* path, rmi_key_t **sorted_array, int64_t *n) {
  std::ifstream infile(path, std::ios::in | std::ios::binary);
  if (!infile.good()) {
    printf("%s file not found\n", path);
    exit(0);
  }

  infile.read((char *)(n), sizeof(uint64_t));
  *sorted_array = (rmi_key_t*) malloc((*n) * sizeof(rmi_key_t));
  if (*sorted_array == NULL) return false;
  
  infile.read((char*)(*sorted_array), (*n) * sizeof(rmi_key_t));
  if (!infile.good()) return false;
  
  return true;
}

bool load_rmi(char* path, double* L0_PARAMETER0, double* L0_PARAMETER1, int64_t* L1_SIZE, double** L1_PARAMETERS) {
  std::ifstream infile(path, std::ios::in | std::ios::binary);
  if (!infile.good()) {
    printf("%s file not found\n", path);
    exit(0);
  }

  infile.read((char *)(L0_PARAMETER0), sizeof(double));
  infile.read((char *)(L0_PARAMETER1), sizeof(double));
  infile.read((char *)(L1_SIZE), sizeof(int64_t));

  if (!infile.good()) {
    fprintf(stderr, "failed L0 params and L1_SIZE\n");
    return false;
  }

  *L1_PARAMETERS = (double*) malloc(*L1_SIZE * 3 * sizeof(double));
  if (*L1_PARAMETERS == NULL) {
    fprintf(stderr, "failed malloc\n");
    return false;
  };

  infile.read((char*)(*L1_PARAMETERS), *L1_SIZE * 3 * sizeof(double));
  // this fails in "normal" use case, assume that is okay
  if (!infile.good()) {
    fprintf(stderr, "failed read\n");
    //return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int keysLength;
  rmi_key_t *hostKeys;
  int64_t *hostPositions;
  // host RMI values
  rmi_key_t *sorted_array;
  int64_t n;
  double L0_PARAMETER0 = 0.0;
  double L0_PARAMETER1 = 0.0;
  int64_t L1_SIZE = 0;
  double *L1_PARAMETERS;

  rmi_key_t *deviceKeys;
  double* deviceL1_PARAMETERS;
  int64_t *devicePositions;

  // hardcoded value for now
  char uint64_path[71] = "test/input/MT-human.fa_map-ont_minimizers_key_value_sorted_keys.uint64";

  // load_sorted_array with CPU
  if(!load_sorted_array(uint64_path, &sorted_array, &n)){
    fprintf(stderr, "Failed load_sorted_array\n");
    free(sorted_array);
    exit(-1);
  }
  fprintf(stderr, "Success load_sorted_array, n = %" PRId64 "\n", n);

  // hardcoded value for now
  char rmiparams_path[79] = "test/input/MT-human.fa_map-ont_minimizers_key_value_sorted_keys.rmi_PARAMETERS";

  // load_rmi with CPU
  if(!load_rmi(rmiparams_path, &L0_PARAMETER0, &L0_PARAMETER1, &L1_SIZE, &L1_PARAMETERS)) {
    fprintf(stderr, "Failed load_rmi\n");
    free(sorted_array);
    free(L1_PARAMETERS);
    exit(-2);
  }
  fprintf(stderr, "Success load_rmi\n");

  fprintf(stderr, "L0_PARAMETER0 = %E, L0_PARAMETER1 = %E, L1_SIZE = %ld\n", 
          L0_PARAMETER0, L0_PARAMETER1, L1_SIZE);

  args = wbArg_read(argc, argv);

  // wbImport does not work with uint64_t values, must load manually
  // this is certainly a bottleneck that must be improved upon
  fprintf(stderr, "loading keys...\n");
  std::ifstream infile("test/input/keys.raw", std::ios::in | std::ios::binary);
  std::string line;
  if (getline(infile, line)) { // get the first line (size) and malloc based on this value
    keysLength = std::stoi(line);
    hostKeys = (rmi_key_t*) malloc(keysLength * sizeof(rmi_key_t));
  }

  // Read the rest of the lines and populate the array
  int j = 0;
  while (getline(infile, line)) {
    hostKeys[j] = std::stoull(line);
    j++;
  }
  infile.close();

  hostPositions = (int64_t*) malloc(keysLength * sizeof(int64_t));

  //@@ Allocate GPU memory here
  wbCheck(hipMalloc(&deviceKeys, keysLength * sizeof(rmi_key_t)));
  wbCheck(hipMalloc(&deviceL1_PARAMETERS, L1_SIZE * 3 * sizeof(double)));
  wbCheck(hipMalloc(&devicePositions, keysLength * sizeof(int64_t)));

  //@@ Copy input and kernel to GPU here
  wbCheck(hipMemcpy(deviceKeys, hostKeys, keysLength * sizeof(rmi_key_t), hipMemcpyHostToDevice));
  wbCheck(hipMemcpy(deviceL1_PARAMETERS, L1_PARAMETERS, L1_SIZE * 3 * sizeof(double), hipMemcpyHostToDevice));
  wbCheck(hipMemcpyToSymbol(HIP_SYMBOL(constant_sorted_array), sorted_array, n * sizeof(float)));

  //@@ Initialize grid and block dimensions here
  dim3 threads(THREAD_SIZE);
  dim3 blocks((keysLength + THREAD_SIZE - 1) / THREAD_SIZE);

  //@@ Launch the GPU kernel here
  uint64_t start_time = __rdtsc();
  fprintf(stderr, "TIMER LOG: rmi_lookup start- %" PRIu64 "\n", start_time);
  rmi_lookup<<<blocks, threads>>>(deviceKeys, deviceL1_PARAMETERS, devicePositions, n, L0_PARAMETER0, L0_PARAMETER1, L1_SIZE);
  hipDeviceSynchronize();
  uint64_t end_time = __rdtsc();
  uint64_t runtime = end_time - start_time;
  fprintf(stderr, "TIMER LOG: rmi_lookup end- %" PRIu64 "\n", end_time);
  fprintf(stderr, "TIMER LOG: rmi_lookup time- %" PRIu64 "\n", runtime);

  fprintf(stderr, "TIMER LOG: warmup start- %l" PRIu64 "\n", __rdtsc());
  rmi_lookup<<<blocks, threads>>>(deviceKeys, deviceL1_PARAMETERS, devicePositions, n, L0_PARAMETER0, L0_PARAMETER1, L1_SIZE);
  rmi_lookup<<<blocks, threads>>>(deviceKeys, deviceL1_PARAMETERS, devicePositions, n, L0_PARAMETER0, L0_PARAMETER1, L1_SIZE);
  hipDeviceSynchronize();
  fprintf(stderr, "TIMER LOG: warmup end- %l" PRIu64 "\n", __rdtsc());

  fprintf(stderr, "TIMER LOG: timing start- %l" PRIu64 "\n", __rdtsc());
  float totalTime = 0;
  float milliseconds = 0;
  hipEvent_t start, stop;
  wbCheck(hipEventCreate(&start));
  wbCheck(hipEventCreate(&stop));
  // Measure the average time of the kernel over 10 iterations
  wbCheck(hipEventRecord(start, 0));
  for (int i = 0; i < 10; ++i) {
    rmi_lookup<<<blocks, threads>>>(deviceKeys, deviceL1_PARAMETERS, devicePositions, n, L0_PARAMETER0, L0_PARAMETER1, L1_SIZE);
  }
  wbCheck(hipEventRecord(stop, 0));
  wbCheck(hipEventSynchronize(stop));
  wbCheck(hipEventElapsedTime(&milliseconds, start, stop));
  totalTime += milliseconds;
  fprintf(stderr, "Total time: %f\n", totalTime);
  fprintf(stderr, "TIMER LOG: timing end- %l" PRIu64 "\n", __rdtsc());

  //@@ Copy the device memory back to the host here
  wbCheck(hipMemcpy(hostPositions, devicePositions, keysLength * sizeof(int64_t), hipMemcpyDeviceToHost));

  // uncomment to get output values in slurm file
  // for(int i = 0; i < keysLength; i++) {
  //   fprintf(stderr, "positions[%d] = %" PRId64 "\n", i, hostPositions[i]);
  // }

  //@@ Solution
  wbSolution(args, hostPositions, keysLength);

  //@@ Free device memory
  wbCheck(hipFree(deviceKeys));
  wbCheck(hipFree(deviceL1_PARAMETERS));
  wbCheck(hipFree(devicePositions));

  //@@ Free host memory
  free(hostKeys);
  free(sorted_array);
  free(L1_PARAMETERS);
  free(hostPositions);

  return 0;
}