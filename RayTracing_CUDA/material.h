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
    __device__ dielectric(float refraction_index) : refraction_index(refraction_index) {}

    __device__ bool scatter(const ray& r_in, const hit_record& rec,
        color& attenuation, ray& scattered,
        curandState* rand_state) const override {
        attenuation = color(1.0f, 1.0f, 1.0f);

        // 법선 방향 설정
        vec3 outward_normal = rec.front_face ? rec.normal : -rec.normal;

        // 굴절율 비율 설정
        float refraction_ratio = rec.front_face ? (1.0f / refraction_index) : refraction_index;

        // cos(theta) 계산
        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fminf(dot(-unit_direction, outward_normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

        // 굴절 불가능 여부 확인
        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;

        // 방향 선택: 굴절 또는 반사
        vec3 direction;
        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(rand_state)) {
            // 반사
            direction = reflect(unit_direction, outward_normal);
        }
        else {
            // 굴절
            direction = refract(unit_direction, outward_normal, refraction_ratio);
        }

        scattered = ray(rec.p, direction);
        return true;
    }


    // Refractive index in vacuum or air, or the ratio of the material's refractive index over
    // the refractive index of the enclosing media
    __device__ float reflectance(float cosine, float refraction_index) const {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * powf((1 - cosine), 5.0f);
    }

    float refraction_index;
};