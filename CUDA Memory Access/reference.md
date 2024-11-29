## Reference Documentation

### `cudaMemcpy()`  
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
- **Returns**: A `cudaError_t` error code.  

### `cudaMemcpyAsync()`  
- **Description**: Performs non-blocking memory transfers using streams.  
- **Prototype**:  
  ```cpp
  cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0);
  ```  

### `cudaMalloc()`  
- **Description**: Allocates memory on the device.  
- **Prototype**:  
  ```cpp
  cudaError_t cudaMalloc(void **devPtr, size_t size);
  ```  

### `cudaFree()`  
- **Description**: Frees memory previously allocated with `cudaMalloc()`.  
- **Prototype**:  
  ```cpp
  cudaError_t cudaFree(void *devPtr);
  ```  

### `cudaMemset()`  
- **Description**: Initializes or resets device memory.  
- **Prototype**:  
  ```cpp
  cudaError_t cudaMemset(void *devPtr, int value, size_t count);
  ```  

### Best Practices Table  

| Function          | Purpose                      | Notes                           |
|-------------------|------------------------------|---------------------------------|
| `cudaMemcpy()`    | Data transfer (blocking)     | Simple but synchronous.         |
| `cudaMemcpyAsync()` | Data transfer (non-blocking) | Use streams for concurrency.    |
| `cudaMalloc()`    | Memory allocation on device  | Explicit deallocation required. |
| `cudaFree()`      | Free device memory           | Essential to avoid leaks.       |
| `cudaMemset()`    | Initialize device memory     | Ideal for zeroing memory.       |


## New header 
Please, update me; and write in some new files and ensure that you keep me updated.