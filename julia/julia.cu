//
// GPU Julia set application from Sanders and Kandrot (p. 54)
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
//#include <cutil.h>
#include "cpu_bitmap.h"

using std::cout;
using std::endl;

#define DIM 700

struct cuComplex {

	float r;
	float i;

	__device__ cuComplex( ) : r(), i() {}

	__device__ cuComplex( float a, float b) : r(a), i(b) {}

	__device__ float magnitude2(void) {
		return r*r + i*i;
	}

	__device__ cuComplex operator*(const cuComplex& a) {
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}

	__device__ cuComplex operator+(const cuComplex& a) {
		return cuComplex(r+a.r, i+a.i);
	}
};

__device__ int function(int x, int y,int set) {

	const float scale = 1.5;
	float jx = scale * (float) (DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float) (DIM / 2 - y) / (DIM / 2);

	cuComplex c;

	switch (set) {
	case 0:
		c.r=-.8;
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
	cuComplex a(jx, jy);

	//return a.magnitude2() % 8;

	for (int i = 0; i < 47; i++) {

		a = a * a + c;

		int colors = 7;
		for(int color = 0; color <= colors; color++){
			float weight = (float)(colors - color) / colors;
			float magfloor = 1.5;
			float magceiling = 1000;
			float threshhold = ((magceiling - magfloor) * weight) + magfloor;
			//float threshhold = magceiling;

			float magnitude2 = a.magnitude2();

			if (magnitude2 > threshhold )
				return color;
		}
	}
	return 7;
}

__global__ void kernel(unsigned char *ptr,int set) {
	// map from threadIdx/blockIdx to pixel position
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;

	// now calculate the value at that position
	int value = function(x, y,set);

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

	int width = 750;
	int height = 550;
	CPUBitmap bitmap(width, height);
	unsigned char *dev_bitmap;

	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
	dim3 grid(width, height);

	int set = atoi(argv[1]);

	kernel<<<grid,1>>>(dev_bitmap,set);

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, 
		bitmap.image_size(), 
		cudaMemcpyDeviceToHost);

	bitmap.display_and_exit();
	cudaFree(dev_bitmap);
}