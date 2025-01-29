#pragma once
#include <math.h>
#include <stdlib.h>
#include <iostream>

// Utility Function (old rtweekend.h & color.h)
__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * 3.1415926535897932385 / 180.0;
}

__device__ inline float linear_to_gamma(float linear_component)
{
    if (linear_component > 0)
        return sqrt(linear_component);

    return 0;
}

__device__ inline float random_float(curandState* state) {
    return curand_uniform(state);
}

__device__ inline float random_float(float min, float max, curandState* state) {
    // Returns a random real in [min,max).
    return min + (max - min) * curand_uniform(state);
}


class vec3 {
public:
    float e[3];

    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3& operator+=(const vec3& v2);
    __host__ __device__ inline vec3& operator-=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const vec3& v2);
    __host__ __device__ inline vec3& operator/=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);

    __host__ __device__ inline float length() const { return sqrtf(length_squared()); }
    __host__ __device__ inline float length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __host__ __device__ inline bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        auto s = 1e-8;
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
    }

    __device__ static vec3 random(curandState* rand_state) { return vec3(random_float(rand_state), random_float(rand_state), random_float(rand_state)); }
    __device__ static vec3 random(float min, float max, curandState* rand_state) { return vec3(random_float(min, max, rand_state), random_float(min, max, rand_state), random_float(min, max, rand_state)); }
};


// point3 and color is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;
using color = vec3;


// Vector Utility Functions
inline std::istream& operator>>(std::istream& is, vec3& t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t) {
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}


__host__ __device__ inline vec3& vec3::operator+=(const vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__device__ inline vec3 random_unit_vector(curandState* rand_state) {
    while (true) {
        auto p = vec3::random(-1.0f, 1.0f, rand_state);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1) return p / std::sqrt(lensq);  // Checking whether point P is inside the sphere
    }
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState* rand_state) {
    vec3 on_unit_sphere = random_unit_vector(rand_state);
    if (dot(on_unit_sphere, normal) > 0.0)  // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ vec3 random_in_unit_disk(curandState* local_rand_state) {
    while (true) {
        auto p = vec3(random_float(-1.0f, 1.0f, local_rand_state), random_float(-1.0f, 1.0f, local_rand_state), 0);
        if (p.length_squared() < 1)
            return p;
    }
}

__device__ vec3 sample_square(curandState* local_rand_state) {
    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    return vec3(random_float(local_rand_state) - 0.5, random_float(local_rand_state) - 0.5, 0);
}

__device__ point3 defocus_disk_sample(point3 center, vec3 defocus_disk_u, vec3 defocus_disk_v, curandState* local_rand_state) {
    // Returns a random point in the camera defocus disk.
    vec3 p = random_in_unit_disk(local_rand_state);
    return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = fminf(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrtf(fabsf(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ inline bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ inline float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}