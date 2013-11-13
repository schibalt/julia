//
// GPU Julia set application from Sanders and Kandrot (p. 54)
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
//#include <cutil.h>
#include "cpu_bitmap.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using std::cout;
using std::endl;

#define XDIM 700
#define YDIM 700

struct cuComplex {

	float r;
	float i;

	__device__ cuComplex( ) : r(), i() {}

	__device__ cuComplex( float a, float b) : r(a), i(b) {}

	__device__ float magnitude2(void) {
		return r * r + i * i;
	}

	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r + a.r, i + a.i);
	}

	__device__ cuComplex operator-(const cuComplex& a) {
		return cuComplex(r - a.r, i - a.i);
	}
};

__device__ cuComplex magnitude2(cuComplex a, cuComplex c){
	return a * a + c;
}

__device__ cuComplex magnitude3(cuComplex a, cuComplex c){
	return a * a * a + c;
}

__device__ cuComplex braunlformula(cuComplex a, cuComplex c){
	return a * a * a + c * a - c;
}

__device__ int function(int x, int y, int set, unsigned char magnitude, bool usebraunlformula, bool mandelbrot) {

	const float scale = 1.5;
	float jx = scale * (float) (XDIM / 2 - x) / (XDIM / 2);
	float jy = scale * (float) (YDIM / 2 - y) / (YDIM / 2);

	cuComplex c;

	switch (set) {
	case 0:
		c.r = -.8;
		c.i = .156;
		//= cuComplex(-0.8, 0.156);
		break;
	case 1:
		c.r = -.6;
		c.i = 0;
		//c = cuComplex(-0.6, 1);
		break;
	case 2:
		c.r = -.123;
		c.i = .745;
		//c = cuComplex(-0.123, 0.745);
		break;
	case 3:
		c.r = -.391;
		c.i = -.587;
		//c = cuComplex(-0.391, 0.587);
		break;
	case 4:
		c.r = .285;
		c.i = .01;
		//c = cuComplex(-0.123, 0.745);
		break;
	case 5:
		c.r = .25;
		c.i = 0;
		//c = cuComplex(-0.391, 0.587);
		break;
	}

	//cuComplex a(jx, jy);
	cuComplex a;

	int magfloor = 2;
	float magceiling = 3;

	cuComplex (*formula)(cuComplex, cuComplex);

	if(magnitude == 2)
		formula = magnitude2;
	else if(usebraunlformula)
		formula = braunlformula;
	else
		formula = magnitude3;

	if(mandelbrot){
		magfloor = 4;
		a.r = 0;
		a.i = 0;
		c.r = jx;
		c.i = jy;
	}
	else{
		a.r = jx;
		a.i = jy;
		a = formula(a, c);
	}

	for (int i = 0; i < 47; i++) {

		int colors = 7;
		for(int color = 0; color <= colors; color++){
			float weight = (float)(colors - color) / colors;
			float threshhold = ((magceiling - magfloor) * weight) + magfloor;
			//float threshhold = magceiling;

			float magnitude2 = a.magnitude2();

			if (magnitude2 > threshhold )
				return color;
		}

		a = formula(a, c);
	}
	return 7;
}

__global__ void kernel(unsigned char *ptr, unsigned char set, unsigned char magnitude, unsigned char usebraunlformula, unsigned char mandelbrot) {
	// map from threadIdx/blockIdx to pixel position
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	// now calculate the value at that position
	int value = function(x, y, set, magnitude, usebraunlformula, mandelbrot);

	int red, green, blue;

	switch(value){

		//black
	case 0:
		red=0;
		blue=0;
		green=0;
		break;

		//violet
	case 1:
		red=143;
		blue=0;
		green=255;
		break;

		//indigo
	case 2:
		red=75;
		blue=0;
		green=130;
		break;

		//blue
	case 3:
		red=0;
		blue=0;
		green=255;
		break;

		//green
	case 4:
		red=0;
		blue=255;
		green=0;
		break;

		//yellow
	case 5:
		red=255;
		blue=255;
		green=0;
		break;

		//orange
	case 6:
		red=255;
		blue=127;
		green=0;
		break;

		//red
	case 7:
		red=255;
		blue=0;
		green=0;
		break;
	}

	ptr[offset * 4 + 0] = red;
	ptr[offset * 4 + 1] = blue;
	ptr[offset * 4 + 2] = green;
	ptr[offset * 4 + 3] = 255;
}

int main(int argc, char* argv[]) {

	CPUBitmap bitmap(XDIM, YDIM);
	unsigned char *dev_bitmap;

	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
	dim3 grid(XDIM, YDIM);

	unsigned char set = atoi(argv[1]);
	unsigned char magnitude = atoi(argv[2]);
	unsigned char usebraunlformula = atoi(argv[3]);
	unsigned char mandelbrot = atoi(argv[4]);

	kernel<<<grid,1>>>(dev_bitmap, set, magnitude, usebraunlformula, mandelbrot);

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, 
		bitmap.image_size(), 
		cudaMemcpyDeviceToHost);

	bitmap.display_and_exit();
	cudaFree(dev_bitmap);
}