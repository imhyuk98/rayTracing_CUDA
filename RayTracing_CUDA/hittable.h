#pragma once
#include "ray.h"
#include "interval.h"

class material;

class hit_record {
public:
    point3 p;
    vec3 normal;
    material* mat_ptr;
    float t;
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};