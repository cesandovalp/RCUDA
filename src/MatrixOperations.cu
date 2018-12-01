#include "MatrixOperations.hpp"

void MatrixOperations::HadamardProduct( Matrix<float>& a, Matrix<float>& b )
{
  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] *= b.data[i];
}

void MatrixOperations::Transpose( Matrix<float>& matrix )
{
  int columns = matrix.columns;
  matrix.columns = matrix.rows;
  matrix.rows    = columns;

  for( int i = 0; i < matrix.columns; ++i )
    for( int j = 0; j < matrix.rows; ++j )
      ( matrix.data + ( i * matrix.columns ) )[j] = ( matrix.data + ( j * matrix.columns ) )[i];
}

void MatrixOperations::Apply( Matrix<float>& matrix, float (*function)( float ) )
{
  for( int i = 0; i < matrix.rows * matrix.columns; ++i )
    matrix.data[i] = function( matrix.data[i] );
}

void MatrixOperations::Add( Matrix<float>& a, Matrix<float>& b )
{
  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] += b.data[i];
}

void MatrixOperations::Subtract( Matrix<float>& a, Matrix<float>& b )
{
  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] -= b.data[i];
}

void MatrixOperations::Subtract( Matrix<float>& a, float* b )
{
  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] -= b[i];
}

void MatrixOperations::Assign( Matrix<float>& a, Matrix<float>& b )
{
  for( int i = 0; i < a.rows * a.columns; ++i )
    a.data[i] = b.data[i];
}

void MatrixOperations::Multiplication( Matrix<float>& a, Matrix<float>& b, Matrix<float>& result )
{
  for( int i = 0; i < a.rows; ++i )
    for( int j = 0; j < b.columns; ++j )
      for( int k = 0; k < a.columns; ++k )
        result.data[i * b.columns + j] += Get( a, i, k ) * Get( b, k, j );
}

void MatrixOperations::Multiplication( float* a, Matrix<float>& b, Matrix<float>& result )
{
  for( int i = 0; i < result.rows; ++i )
    for( int j = 0; j < b.columns; ++j )
      for( int k = 0; k < result.columns; ++k )
        result.data[i * b.columns + j] += a[i] * Get( b, k, j );
}

void MatrixOperations::Copy( Matrix<float>& matrix, float* array )
{
  for( int i = 0; i < matrix.rows * matrix.columns; ++i )
    matrix.data[i] = array[i];
}

void MatrixOperations::Fill( Matrix<float>& matrix, float value )
{
  for( int i = 0; i < matrix.rows * matrix.columns; ++i )
    matrix.data[i] = value;
}

float MatrixOperations::Get( Matrix<float>& matrix, int row, int column )
{
  return *( matrix.data + ( row * matrix.columns ) + column );
}