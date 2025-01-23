#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// C++ Std Usings

__constant__ const double infinity = std::numeric_limits<double>::infinity();
__constant__ const double pi = 3.1415926535897932385;

__host__ __device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ inline double random_double(curandState* local_rand_state) {
    return curand_uniform_double(local_rand_state);
}

// cuRAND를 사용한 [min, max) 범위의 난수 생성 함수
__device__ inline double random_double(double min, double max, curandState* local_rand_state) {
    return min + (max - min) * random_double(local_rand_state);
}

// Common Headers

#include "interval.h"
#include "ray.h"
#include "vec3.h"

#endif