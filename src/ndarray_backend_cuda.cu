#include <cuda_runtime.h>
#include <nvfunctional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256
#define TILE 16
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}

CudaDims CudaTwoDim(size_t rows, size_t cols) {
  /**
   * Utility function to get cuda dimensions for 2D call
   */
  CudaDims dim;
  size_t num_row_blocks = (rows + TILE - 1) / TILE;
  size_t num_col_blocks = (cols + TILE - 1) / TILE;
  dim.block = dim3(TILE, TILE, 1);
  dim.grid = dim3(num_col_blocks, num_row_blocks, 1);
  return dim;
}

#define MAX_VEC_SIZE 8
template <typename T> 
struct CudaVec {
  uint32_t size;
  T data[MAX_VEC_SIZE];
};
template <typename T> 
CudaVec<T> VecToCuda(const std::vector<T>& x) {
  CudaVec<T> shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides


__device__ size_t GetNonCompactIndex(size_t gid, CudaVec<int> strides, CudaVec<uint32_t> compact_strides, size_t offset) {
  /**
   * Returns an index of a non-compact array with strides for the corresponding item (at location gid)
   * in the compact array out.
   * 
   * Args:
   *   gid: index of a compact array
   *   strides: vector of strides of a non-acompact array
   *   compact_strides: vector of compact strides of out array (derived based on the out shape)
   *   offset: offset of a non-acompact array
   */
  size_t index = offset;
  size_t remaining_elems = gid;
  for(size_t i=0; i<strides.size; i++) {
      index += (remaining_elems / compact_strides.data[i]) * strides.data[i];
      remaining_elems = remaining_elems % compact_strides.data[i];
  }
  return index;
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec<uint32_t> shape,
                              CudaVec<int> strides, CudaVec<uint32_t> compact_strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   compact_strides: vector of compact strides of out array (derived based on the out shape)
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t src_index = GetNonCompactIndex(gid, strides, compact_strides, offset);
    out[gid] = a[src_index];
  }
}

std::vector<uint32_t> GetCompactStrides(std::vector<uint32_t>& shape) {
  /**
   * Returns compact strides based on array shape.
   * 
   * Args:
   *   shape: shapes of an array
   */
  std::vector<uint32_t> compact_strides(shape.size());
  uint32_t stride = 1;
  for(int i=shape.size()-1; i>=0; i--) {
    compact_strides[i] = stride;
    stride *= shape[i];
  } 
  return compact_strides;
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
             std::vector<int> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */
  std::vector<uint32_t> compact_strides = GetCompactStrides(shape);
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda<uint32_t>(shape),
                                         VecToCuda<int>(strides), VecToCuda<uint32_t>(compact_strides), offset);
}

__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec<uint32_t> shape,
                                   CudaVec<int> strides, CudaVec<uint32_t> compact_strides, size_t offset) {
  /**
   * The CUDA kernel for the elementwise set item opeation. This will effectively set items from the 
   * compact *a* array  
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   compact_strides: vector of compact strides of out array (derived based on the out shape)
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t dest_index = GetNonCompactIndex(gid, strides, compact_strides, offset);
    out[dest_index] = a[gid];
  }
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<uint32_t> shape,
                  std::vector<int> strides, size_t offset) {
  /**
   * Set items in a (non-compact) out array using CUDA.  You will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  std::vector<uint32_t> compact_strides = GetCompactStrides(shape);
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda<uint32_t>(shape),
                                              VecToCuda<int>(strides), VecToCuda<uint32_t>(compact_strides), offset);
}

__global__ void ScalarSetitemKernel(size_t size, scalar_t val, scalar_t* out, CudaVec<uint32_t> shape,
                                    CudaVec<int> strides, CudaVec<uint32_t> compact_strides, size_t offset) {
  /**
   * The CUDA kernel for the scalar set item opeation. This will effectively set items in out (non-compact)
   * array to value val.
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    size_t dest_index = GetNonCompactIndex(gid, strides, compact_strides, offset);
    out[dest_index] = val;
  }
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<uint32_t> shape,
                   std::vector<int> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  std::vector<uint32_t> compact_strides = GetCompactStrides(shape);
  CudaDims dim = CudaOneDim(out->size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(size, val, out->ptr, VecToCuda(shape), VecToCuda(strides), 
                                              VecToCuda(compact_strides), offset);
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * b[gid];
}

void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Multiply together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseMulKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] * val;
}

void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Multiply together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarMulKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / b[gid];
}

void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Divide a CUDA array by an another one.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseDivKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] / val;
}

void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Divide a CUDA array by a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarDivKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = pow(a[gid], val);
}

void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Power a CUDA array to a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarPowerKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = max(a[gid], b[gid]);
}

void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Find elementwise maximum of two CUDA arrays
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = max(a[gid], val);
}

void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Find maximum of a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarMaximumKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == b[gid]);
}

void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Find if two CUDA arrays are elementwise equal.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseEqKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] == val);
}

void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Find if a CUDA array is elementwise equal to a scalar.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarEqKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= b[gid]);
}

void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Find if two CUDA arrays are elementwise greater or equal.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseGeKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = (a[gid] >= val);
}

void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Find if a CUDA array is elementwise greater or equal to a scalar.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarGeKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = log(a[gid]);
}

void EwiseLog(const CudaArray& a, CudaArray* out) {
  /**
   * Find elementwise log of a CUDA array.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseLogKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = exp(a[gid]);
}

void EwiseExp(const CudaArray& a, CudaArray* out) {
  /**
   * Find elementwise exp of a CUDA array.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseExpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = tanh(a[gid]);
}

void EwiseTanh(const CudaArray& a, CudaArray* out) {
  /**
   * Find elementwise tanh of a CUDA array.
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseTanhKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);
}

/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Matmul
////////////////////////////////////////////////////////////////////////////////
__global__ void MatmulKernelShared(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t N, size_t P) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ scalar_t shared_a[TILE][TILE];
  __shared__ scalar_t shared_b[TILE][TILE];
  size_t tile_r = threadIdx.y;
  size_t tile_c = threadIdx.x;
  scalar_t out_val = 0.;
  
  for(size_t k=0; k<(N+TILE-1)/TILE; k++) {
    shared_a[tile_r][tile_c] = (row<M && (k*TILE+tile_c)<N)? a[row*N + k*TILE+tile_c]: 0.;
    shared_b[tile_r][tile_c] = ((k*TILE+tile_r)<N && col<P)? b[(k*TILE+tile_r)*P + col]: 0.;
    __syncthreads();

    for(int m=0; m<TILE; m++) {
      out_val += shared_a[tile_r][m] * shared_b[m][tile_c];
    }
    __syncthreads();
  }
  if((row < M) && (col < P)) {
    out[row*P + col] = out_val;
  }
}

__global__ void MatmulKernelRegisters(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t M, size_t N, size_t P) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t start_row = (gid / ((P+TILE-1) / TILE)) * TILE;
  size_t start_col = (gid % ((P+TILE-1) / TILE)) * TILE;
  scalar_t rc[TILE][TILE] = {0.};
  scalar_t ra[TILE], rb[TILE];
  
  for(size_t k=0; k<N; k++) {
    for(int i=0; i<TILE; i++) {
      ra[i] = ((start_row+i) < M)? a[(start_row + i)*N + k]: 0.;
      rb[i] = ((start_col+i) < P)? b[k*P + (start_col+i)]: 0.;
    }
    for(int i=0; i<TILE; i++) {
      for(int j=0; j<TILE; j++) {
        rc[i][j] += (ra[i] * rb[j]);
      }
    }
  }
  
  for(int i=0; i<TILE; i++) {
    for(int j=0; j<TILE; j++) {
      if(((start_row+i) < M) && ((start_col+j) < P)) {
        out[(start_row+i)*P + (start_col+j)] = rc[i][j];
      }
    }
  }
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */
  // CudaDims dim = CudaOneDim(((M + TILE - 1) / TILE) * ((P + TILE - 1) / TILE));
  // MatmulKernelRegisters<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P); 
  CudaDims dim = CudaTwoDim(M, P);
  MatmulKernelShared<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t max_val = 0.;
    for(size_t j=0; j<reduce_size; j++) {
      if (j == 0)
        max_val = a[gid*reduce_size + j];
      else
        max_val = max(max_val, a[gid*reduce_size + j]);
    }
    out[gid] = max_val;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t size, size_t reduce_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    scalar_t sum_val = 0.;
    for(size_t j=0; j<reduce_size; j++) {
        sum_val += a[gid*reduce_size + j];
    }
    out[gid] = sum_val;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
}

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}
