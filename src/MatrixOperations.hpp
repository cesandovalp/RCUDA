#pragma once

#include "Matrix.hpp"

using spann::Matrix;

class MatrixOperations
{
  public:
    __device__ __host__
    void HadamardProduct( Matrix<float>&, Matrix<float>& );
    __device__ __host__
    void Transpose( Matrix<float>& );
    __device__ __host__
    void Add( Matrix<float>&, Matrix<float>& );
    __device__ __host__
    void Subtract( Matrix<float>&, Matrix<float>& );
    __device__ __host__
    void Subtract( Matrix<float>&, float* );
    __device__ __host__
    void Subtract( Matrix<float>&, float*, Matrix<float>& );
    __device__ __host__
    void Apply( Matrix<float>&, float (*function)( float ) );
    __device__ __host__
    void Assign( Matrix<float>&, Matrix<float>& );
    __device__ __host__
    void Multiplication( Matrix<float>&, Matrix<float>&, Matrix<float>& );
    __device__ __host__
    void Multiplication( float*, Matrix<float>&, Matrix<float>& );
    __device__ __host__
    void Copy( Matrix<float>&, float* );
    __device__ __host__
    void Fill( Matrix<float>&, float );
    __device__ __host__
    float Get( Matrix<float>&, int, int );
};