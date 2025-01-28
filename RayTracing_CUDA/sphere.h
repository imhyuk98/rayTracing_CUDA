#ifndef SPHEREH
#define SPHEREH

#include "hittable.h"

class sphere : public hittable {
public:
    __device__ sphere() {}
    __device__ sphere(point3 center, float radius, material* m) : center(center), radius(std::fmax(0.0f, radius)), mat_ptr(m) {}
    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const;
    vec3 center;
    float radius;
    material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, interval ray_t, hit_record& rec) const {
    vec3 oc = center - r.origin();
    auto a = r.direction().length_squared();
    auto h = dot(r.direction(), oc);
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = h * h - a * c;  // h = b/2
    if (discriminant < 0)
        return false;

    auto sqrtd = sqrtf(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (h - sqrtd) / a;
    if (!ray_t.surrounds(root)) {
        root = (h + sqrtd) / a;
        if (!ray_t.surrounds(root))
            return false;
    }

    rec.t = root;                                       // t
    rec.p = r.at(rec.t);                                // intersection point
    vec3 outward_normal = (rec.p - center) / radius;    // normalize with radius
    rec.normal = (rec.p - center) / radius;
    rec.mat_ptr = mat_ptr;
    return true;
}


#endif