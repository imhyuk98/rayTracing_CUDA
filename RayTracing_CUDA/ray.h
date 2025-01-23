#ifndef RAYH
#define RAYH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"

class ray {
public:
    __host__ __device__ ray() {}
    __host__ __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}
    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ vec3 direction() const { return dir; }
    __host__ __device__ point3 point_at_parameter(float t) const { return orig + t * dir; }

    point3 orig;
    vec3 dir;
};

#endif