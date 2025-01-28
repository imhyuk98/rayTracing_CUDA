#pragma once
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
    bool front_face;

    __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        front_face = dot(r.direction(), outward_normal) < 0;        // negative value indicates opposite direction of the ray and the normal 
        normal = front_face ? outward_normal : -outward_normal;     // negative normal vector indicates a ray starting from inside
    }
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};