// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

extern __shared__ _FTYPE_ sharemem[];

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{

    int I = blockIdx.y * blockDim.y + threadIdx.y;
    int J = blockIdx.x * blockDim.x + threadIdx.x;

    if ((I < N) && (J < N))
    {
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++)
        {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
            // printf("%f %f %f\n", a, b, _c);
        }
        C[I * N + J] = _c;
    }
}

#else
// You should be changing the kernel here for the non naive implementation.
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B)
{
    // printf("TILEDIM_M %d\n", TILEDIM_M);
    _FTYPE_ *__restrict__ As = &sharemem[0];
    _FTYPE_ *__restrict__ Bs = &sharemem[TILEDIM_M * TILEDIM_K];

    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = blockIdx.x;
    int I = by * TILEDIM_K + ty;
    int J = bx * TILEDIM_N + tx;
    double cij0, cij1, cij2, cij3  = 0;

    // int I =  blockIdx.y*blockDim.y + threadIdx.y;
    // int J =  blockIdx.x*blockDim.x + threadIdx.x;
    int TW = TILEDIM_K < N ? TILEDIM_K : N;
    // TILEDIM_K = TW;TILEDIM_M = TILEDIM_N = TW;

    if((I < N) && (J < N))
    {
        //     _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N / TW; k++)
        {
            // printf("%f %f\n", A[(I * N + k * TILEDIM_K + tx)], B[(k * TILEDIM_K + ty) * N + J]);
            //As[ty * TW + tx] = A[(I * N + k * TW + tx)];
            //Bs[ty * TW + tx] = B[(k * TW + ty) * N + J];
			As[ty * TW + tx] = A[((by * TILEDIM_K * N) + (N * ty) + k * TW + tx)];
			Bs[ty * TW + tx] = B[(k * TW * N) + (N * ty) + J];
			
			As[(ty + 8) * TW + tx] = A[((by * TILEDIM_K * N) + (N * (ty + 8)) + k * TW + tx)];
			Bs[(ty + 8) * TW + tx] = B[(k * TW * N) + (N * (ty + 8)) + J];

			As[(ty + 16) * TW + tx] = A[((by * TILEDIM_K * N) + (N * (ty + 16)) + k * TW + tx)];
			Bs[(ty + 16) * TW + tx] = B[(k * TW * N) + (N * (ty + 16)) + J];
			
			As[(ty + 24) * TW + tx] = A[((by * TILEDIM_K * N) + (N * (ty + 24)) + k * TW + tx)];
			Bs[(ty + 24) * TW + tx] = B[(k * TW * N) + (N * (ty + 24)) + J];



            __syncthreads();

            for (int kk = 0; kk < TW; kk++)
            {
                cij0 += As[ty * TW + kk] * Bs[kk * TW + tx];
				cij1 += As[(ty + 8) * TW + kk] * Bs[kk * TW + tx];
				cij2 += As[(ty + 16) * TW + kk] * Bs[kk * TW + tx];
				cij3 += As[(ty + 24) * TW + kk] * Bs[kk * TW + tx];
                // printf("%d %f %f %f\n", (ty * TILEDIM_K + (kk - tx)), As[ty * TILEDIM_K + kk], Bs[kk * TILEDIM_N + tx], cij);
            }

            __syncthreads();
        }

        //C[I * N + J] = cij;
		
		C[(by * TILEDIM_K * N) + (N * ty) + J] = cij0;
		C[(by * TILEDIM_K * N) + (N * (ty+8)) + J] = cij1;
		C[(by * TILEDIM_K * N) + (N * (ty+16)) + J] = cij2;
		C[(by * TILEDIM_K * N) + (N * (ty+24)) + J] = cij3;
		
    }
}
#endif