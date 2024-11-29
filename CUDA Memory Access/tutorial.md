## Tutorial Guide

### Using CUDA Memory Access Functions for Efficient Data Transfer  

#### Introduction  
In this tutorial, you’ll learn how to use CUDA memory access functions effectively to transfer data between the host and device, manage device memory, and optimize transfer performance using streams.

#### Prerequisites  
- CUDA Toolkit installed.
- Familiarity with basic CUDA programming concepts.

#### Step 1: Allocate and Initialize Host and Device Memory  
```cpp
int N = 1024; // Size of the array
float *h_array = (float *)malloc(N * sizeof(float)); // Host array
float *d_array;

cudaMalloc(&d_array, N * sizeof(float)); // Allocate device memory
```

#### Step 2: Copy Data from Host to Device  
Use `cudaMemcpy()` to transfer data from host to device:  
```cpp
cudaMemcpy(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice);
```

#### Step 3: Perform Computation on the Device  
Launch a simple CUDA kernel (example: scaling array elements):  
```cpp
__global__ void scaleArray(float *arr, int size, float factor) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        arr[idx] *= factor;
    }
}

scaleArray<<<N / 256, 256>>>(d_array, N, 2.0f);
```

#### Step 4: Copy Results Back to the Host  
Transfer results back from the device to the host:  
```cpp
cudaMemcpy(h_array, d_array, N * sizeof(float), cudaMemcpyDeviceToHost);
```

#### Step 5: Free Device Memory  
Always clean up allocated device memory to prevent memory leaks:  
```cpp
cudaFree(d_array);
```

#### Step 6: Use Asynchronous Transfers  
Optimize performance by using `cudaMemcpyAsync()` with streams:  
```cpp
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaMemcpyAsync(d_array, h_array, N * sizeof(float), cudaMemcpyHostToDevice, stream);

// Perform computations while the transfer occurs
scaleArray<<<N / 256, 256, 0, stream>>>(d_array, N, 2.0f);

cudaStreamDestroy(stream);
```

---