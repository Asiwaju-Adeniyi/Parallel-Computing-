# Reference Documentation: CUDA Memory Access Functions

## `cudaMemcpy()`
- **Description**: Copies data between host and device memory.
- **Prototype**:
  ```cpp
  cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
  ```
- **Parameters**:
  - `dst`: Destination pointer.
  - `src`: Source pointer.
  - `count`: Number of bytes to copy.
  - `kind`: Transfer direction (`cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost`, `cudaMemcpyDeviceToDevice`).
- **Example**:
  ```cpp
  float *d_array;
  cudaMalloc(&d_array, size);
  cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);
  ```

## `__shared__` Memory Declaration
- **Purpose**: Shared memory allocation within a CUDA kernel.
- **Example**:
  ```cpp
  __shared__ float sharedArray[BLOCK_SIZE];
  ```

## `cudaMemcpyAsync()`
- **Description**: Asynchronous memory copy using streams.
- **Example**:
  ```cpp
  cudaMemcpyAsync(d_array, h_array, size, cudaMemcpyHostToDevice, stream);
  ```

## Common Memory Optimization Functions
| Function          | Purpose                      | Notes                           |
|-------------------|------------------------------|---------------------------------|
| `cudaMalloc()`    | Allocate device memory       | Requires explicit free.         |
| `cudaMemset()`    | Initialize device memory     | For zeroing arrays, etc.        |
| `cudaFree()`      | Free allocated device memory | Free after kernel execution.    |
