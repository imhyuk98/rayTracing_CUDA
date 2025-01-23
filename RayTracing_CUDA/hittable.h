#ifndef hittableH
#define hittableH

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
};

class hittable {
public:
    __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
};

#endif