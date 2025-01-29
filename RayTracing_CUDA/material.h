#pragma once

class hit_record;





class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state) const {
        return false;
    }
};

class lambertian : public material {
public:
    color albedo;
    __device__ lambertian(const color& albedo) : albedo(albedo) {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state) const {
        vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero()) scatter_direction = rec.normal;

        scattered = ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
};

class metal : public material {
public:
    color albedo;
    float fuzz;

    __device__ metal(const color& albedo, float fuzz) : albedo(albedo), fuzz(fuzz < 1 ? fuzz : 1) {}

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state) const {
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        reflected = unit_vector(reflected) + (fuzz * random_unit_vector(rand_state));      // The bigger fuzz factor, the fuzzier the reflection will be (so zero is no perturbation)
        scattered = ray(rec.p, reflected);
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);                               // Prevent cases where the direction of the fuzzed relection ray points inside the surface
    }
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* local_rand_state) const {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = color(1.0, 1.0, 1.0);
        vec3 refracted;
        float reflect_prob;
        float cosine;

        if (dot(r_in.direction(), rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }

        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;

        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);

        return true;
    }

    float ref_idx;
};