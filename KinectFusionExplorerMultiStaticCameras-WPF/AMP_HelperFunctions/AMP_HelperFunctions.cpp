#include "stdafx.h"
#include "amp.h"

using namespace concurrency;

extern "C" __declspec ( dllexport ) void _stdcall square_array(float* arr, int n)
{
    // Create a view over the data on the CPU
    array_view<float,1> dataView(n, &arr[0]);

    // Run code on the GPU
    parallel_for_each(dataView.extent, [=] (index<1> idx) restrict(amp)
    {
        dataView[idx] = dataView[idx] * dataView[idx];
    });

    // Copy data from GPU to CPU
    dataView.synchronize();
}

const int BLOCK_DIM = 32;


void sum_kernel_tiled(tiled_index<BLOCK_DIM> t_idx, array<int, 1> &A, int stride_size) restrict(amp)
{
    tile_static int localA[BLOCK_DIM];

    index<1> globalIdx = t_idx.global * stride_size;
    index<1> localIdx = t_idx.local;

    localA[localIdx[0]] =  A[globalIdx];
    
    t_idx.barrier.wait();

    // Aggregate all elements in one tile into the first element.
    for (int i = BLOCK_DIM / 2; i > 0; i /= 2) 
    {
        if (localIdx[0] < i) 
        {

            localA[localIdx[0]] += localA[localIdx[0] + i];
        }

        t_idx.barrier.wait();
    }
    
    if (localIdx[0] == 0)
    {
        A[globalIdx] = localA[0];
    }
}

int size_after_padding(int n)
{
    // The extent might have to be slightly bigger than num_stride to 
    // be evenly divisible by BLOCK_DIM. You can do this by padding with zeros.
    // The calculation to do this is BLOCK_DIM * ceil(n / BLOCK_DIM)
    return ((n - 1) / BLOCK_DIM + 1) * BLOCK_DIM;
}
extern "C" __declspec ( dllexport ) void _stdcall findAllocatedUnits(int* inArr, int n, int* outArr, int m) 
{
	// Create a view over the data on the CPU
    array_view<int,1> input(n, &inArr[0]);
	array_view<int,1> sum(m, &outArr[0]);

parallel_for_each( 
        // Define the compute domain, which is the set of threads that are created.
        sum.extent, 
        // Define the code to run on each thread on the accelerator.
        [=](index<1> idx) restrict(amp)
    {
		if(input[idx] == 0x8000)
		{
			sum[idx] = 0;
		}
		else
		{
			sum[idx] = 1;
		}
    }
    );
}
extern "C" __declspec ( dllexport ) void _stdcall reduction_sum_gpu_kernel(int* inArr, int n,int* result) 
{
	array<int, 1> input(n, &inArr[0]);
    int len = input.extent[0];

    ////Tree-based reduction control that uses the CPU.
    //for (int stride_size = 1; stride_size < len; stride_size *= BLOCK_DIM) 
    //{
    //    // Number of useful values in the array, given the current
    //    // stride size.
    //    int num_strides = len / stride_size;  

    //    extent<1> e(size_after_padding(num_strides));
    //    
    //    // The sum kernel that uses the GPU.
    //    parallel_for_each(extent<1>(e).tile<BLOCK_DIM>(), [&input, stride_size] (tiled_index<BLOCK_DIM> idx) restrict(amp)
    //    {
    //        //sum_kernel_tiled(idx, input, stride_size);
    //    });
    //}

    array_view<int, 1> output = input.section(extent<1>(1));
    *result = len;//output[0];
	//return output[0];
	
}