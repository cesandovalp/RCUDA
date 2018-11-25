#include <cstdlib>
#include <iostream>
#include "Matrix.hpp"

using namespace std;
using namespace spann;

__device__ __host__
void Add( Matrix<float>* x, Matrix<float>* y )
{
  int size = x->rows * x->columns;

  for( int i = 0; i < size; ++i )
    x->data[i] += y->data[i];

  //printf( "x->data[0]: %f\n", x->data[0] );
}

// Kernel function to add the elements of two arrays
__global__
void Kernel( int n, Matrix<float>* x, Matrix<float>* y )
{
  int start     = blockIdx.x * blockDim.x + threadIdx.x;
  int increment = blockDim.x * gridDim.x;

  for( int index = start; index < n; index += increment ) 
  {
    Add( &x[index], &y[index] );
  }

  //printf( "x[%d]->data[0]: %f\n", index, x[index].data[0] );
  //x->data[0] = 100;
}

template<typename domain>
void PrintMatrix( Matrix<domain>& M )
{
  for( int i = 0; i < M.rows; ++i )
  {
    for( int j = 0; j < M.columns; ++j )
      std::cout << M[i][j] << "\t|";
    std::cout << std::endl;
  }
}

void hello( int N )
//int main( void )
{
  //int N = 1 << 20;
  //int N = 2560;
  Matrix<float>* host_matrix_a,
               * host_matrix_b,
               * device_matrix_a,
               * device_matrix_b;

  float** device_data_a,
       ** device_tmp_a ,
       ** device_data_b,
       ** device_tmp_b;

  int matrix_rows = 3;
  int matrix_cols = 3;
  int array_size  = matrix_cols * matrix_rows;

  host_matrix_a = new Matrix<float>[N];
  host_matrix_b = new Matrix<float>[N];

  cudaMalloc( (void**) &device_matrix_a, N * sizeof( Matrix<float> ) );
  cudaMalloc( (void**) &device_matrix_b, N * sizeof( Matrix<float> ) );

  device_data_a = new float*[N];
  device_tmp_a  = new float*[N];
  device_data_b = new float*[N];
  device_tmp_b  = new float*[N];

  for( int i = 0; i < N; ++i )
  {
    host_matrix_a[i].SetSize( matrix_rows, matrix_cols, 8 );
    host_matrix_b[i].SetSize( matrix_rows, matrix_cols, 7 );
  }

  // Copying non-pointer data to device object
  cudaMemcpy( device_matrix_a, host_matrix_a, N * sizeof( Matrix<float> ), cudaMemcpyHostToDevice );
  cudaMemcpy( device_matrix_b, host_matrix_b, N * sizeof( Matrix<float> ), cudaMemcpyHostToDevice );

  for( int i = 0; i < N; ++i )
  {
    // Allocate device data   
    cudaMalloc( (void**)& device_data_a[i], array_size * sizeof( float ) );
    cudaMalloc( (void**)& device_tmp_a[i] , array_size * sizeof( float ) );
    cudaMalloc( (void**)& device_data_b[i], array_size * sizeof( float ) );
    cudaMalloc( (void**)& device_tmp_b[i] , array_size * sizeof( float ) );

    // Copy data from host to device
    cudaMemcpy( device_data_a[i], host_matrix_a[i].data, array_size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( device_tmp_a[i] , host_matrix_a[i].tmp , array_size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( device_data_b[i], host_matrix_b[i].data, array_size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( device_tmp_b[i] , host_matrix_b[i].tmp , array_size * sizeof( float ), cudaMemcpyHostToDevice );

    // NOTE: Binding pointers with device
    cudaMemcpy( &( device_matrix_a[i].data ), &device_data_a[i], sizeof( device_matrix_a[i].data ), cudaMemcpyHostToDevice );
    cudaMemcpy( &( device_matrix_a[i].tmp ) , &device_tmp_a[i] , sizeof( device_matrix_a[i].tmp ) , cudaMemcpyHostToDevice );
    cudaMemcpy( &( device_matrix_b[i].data ), &device_data_b[i], sizeof( device_matrix_b[i].data ), cudaMemcpyHostToDevice );
    cudaMemcpy( &( device_matrix_b[i].tmp ) , &device_tmp_b[i] , sizeof( device_matrix_b[i].tmp ) , cudaMemcpyHostToDevice );
  }

  // Run kernel on 1M elements on the GPU
  int block_size = 256;
  int num_blocks = ( N + block_size - 1 ) / block_size;
  printf( "N: %d\nblock_size: %d\nnum_blocks: %d\n", N, block_size, num_blocks );
  Kernel<<<num_blocks, block_size>>>( N, device_matrix_a, device_matrix_b );

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  for( int i = 0; i < N; ++i )
  {
    cudaMemcpy( &device_data_a[i], &( device_matrix_a[i].data ), sizeof( device_data_a[i] ), cudaMemcpyHostToDevice );
    cudaMemcpy( &device_data_b[i], &( device_matrix_b[i].data ), sizeof( device_data_b[i] ), cudaMemcpyHostToDevice );

    cudaMemcpy( host_matrix_a[i].data, device_data_a[i], array_size * sizeof( float ), cudaMemcpyDeviceToHost );
    cudaMemcpy( host_matrix_b[i].data, device_data_b[i], array_size * sizeof( float ), cudaMemcpyDeviceToHost );
  }

  PrintMatrix<float>( host_matrix_a[2] );
  PrintMatrix<float>( host_matrix_b[2] );

  // Free memory
  cudaFree( device_data_a );
  cudaFree( device_data_b );
  cudaFree( device_tmp_a );
  cudaFree( device_tmp_b );

  cout << "CPU Hello World! " << N << endl;
} 
