#include <cstdlib>
#include <iostream>
#include "MatrixOperations.hpp"

using namespace std;
using namespace spann;

struct FFNNData
{
  int  input_size;
  int  output_size;
  int  total_layers;
  int* layers_size;

  MatrixOperations operations;
};

__global__
void Kernel( int n, FFNNData* test, float* input, Matrix<float>* W, Matrix<float>* bias, float* error )
{
  int start     = blockIdx.x * blockDim.x + threadIdx.x;
  int increment = blockDim.x * gridDim.x;

  for( int index = start; index < n; index += increment ) 
  {
    test->operations.Add( W[index], bias[index] );
  }
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
  Matrix<float>* host_W     ,
               * host_bias  ,
               * device_W   ,
               * device_bias;

  float** device_W_data   ,
       ** device_W_tmp    ,
       ** device_bias_data,
       ** device_bias_tmp ;

  float* input,
       * error;

  FFNNData* device_ffnn;
  FFNNData* host_ffnn;

  int matrix_rows = 3;
  int matrix_cols = 3;
  int array_size  = matrix_cols * matrix_rows;

  host_W    = new Matrix<float>[N];
  host_bias = new Matrix<float>[N];

  cudaMalloc( (void**) &device_W   , N * sizeof( Matrix<float> ) );
  cudaMalloc( (void**) &device_bias, N * sizeof( Matrix<float> ) );

  host_ffnn = new FFNNData;
  cudaMalloc( (void**) &device_ffnn, sizeof( FFNNData ) );

  host_ffnn->input_size   = 10;
  host_ffnn->output_size  = 10;
  host_ffnn->total_layers = 10;
  host_ffnn->layers_size  = new int[host_ffnn->total_layers];

  cudaMemcpy( device_ffnn, host_ffnn, sizeof( FFNNData ), cudaMemcpyHostToDevice );

  device_W_data    = new float*[N];
  device_W_tmp     = new float*[N];
  device_bias_data = new float*[N];
  device_bias_tmp  = new float*[N];

  for( int i = 0; i < N; ++i )
  {
    host_W[i].SetSize( matrix_rows, matrix_cols, 8 );
    host_bias[i].SetSize( matrix_rows, matrix_cols, 7 );
  }

  // Copying non-pointer data to device object
  cudaMemcpy( device_W   , host_W   , N * sizeof( Matrix<float> ), cudaMemcpyHostToDevice );
  cudaMemcpy( device_bias, host_bias, N * sizeof( Matrix<float> ), cudaMemcpyHostToDevice );

  for( int i = 0; i < N; ++i )
  {
    // Allocate device data   
    cudaMalloc( (void**)& device_W_data[i]   , array_size * sizeof( float ) );
    cudaMalloc( (void**)& device_W_tmp[i]    , array_size * sizeof( float ) );
    cudaMalloc( (void**)& device_bias_data[i], array_size * sizeof( float ) );
    cudaMalloc( (void**)& device_bias_tmp[i] , array_size * sizeof( float ) );

    // Copy data from host to device
    cudaMemcpy( device_W_data[i  ] , host_W[i].data   , array_size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( device_W_tmp[i]    , host_W[i].tmp    , array_size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( device_bias_data[i], host_bias[i].data, array_size * sizeof( float ), cudaMemcpyHostToDevice );
    cudaMemcpy( device_bias_tmp[i] , host_bias[i].tmp , array_size * sizeof( float ), cudaMemcpyHostToDevice );

    // NOTE: Binding pointers with device
    cudaMemcpy( &( device_W[i].data )   , &device_W_data[i]   , sizeof( device_W[i].data )   , cudaMemcpyHostToDevice );
    cudaMemcpy( &( device_W[i].tmp )    , &device_W_tmp[i]    , sizeof( device_W[i].tmp )    , cudaMemcpyHostToDevice );
    cudaMemcpy( &( device_bias[i].data ), &device_bias_data[i], sizeof( device_bias[i].data ), cudaMemcpyHostToDevice );
    cudaMemcpy( &( device_bias[i].tmp ) , &device_bias_tmp[i] , sizeof( device_bias[i].tmp ) , cudaMemcpyHostToDevice );
  }

  // Run kernel on 1M elements on the GPU
  int block_size = 256;
  int num_blocks = ( N + block_size - 1 ) / block_size;
  printf( "N: %d\nblock_size: %d\nnum_blocks: %d\n", N, block_size, num_blocks );
  Kernel<<<num_blocks, block_size>>>( N, device_ffnn, input, device_W, device_bias, error );

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  for( int i = 0; i < N; ++i )
  {
    cudaMemcpy( &device_W_data[i]   , &( device_W[i].data )   , sizeof( device_W_data[i] )   , cudaMemcpyHostToDevice );
    cudaMemcpy( &device_bias_data[i], &( device_bias[i].data ), sizeof( device_bias_data[i] ), cudaMemcpyHostToDevice );

    cudaMemcpy( host_W[i].data   , device_W_data[i]   , array_size * sizeof( float ), cudaMemcpyDeviceToHost );
    cudaMemcpy( host_bias[i].data, device_bias_data[i], array_size * sizeof( float ), cudaMemcpyDeviceToHost );
  }

  PrintMatrix<float>( host_W[2] );
  PrintMatrix<float>( host_bias[2] );

  // Free memory
  cudaFree( device_W_data );
  cudaFree( device_bias_data );
  cudaFree( device_W_tmp );
  cudaFree( device_bias_tmp );

  cout << "CPU Hello World! " << N << endl;
} 
