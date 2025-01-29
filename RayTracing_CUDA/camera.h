#ifndef CAMERA_H
#define CAMERA_H

#include "material.h"

class camera {
public:
    int    image_height;   // Rendered image height
    float  pixel_samples_scale;  // Color scale factor for a sum of pixel samples
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    vec3   u, v, w;              // Camera frame basis vectors
    vec3   defocus_disk_u;       // Defocus disk horizontal radius
    vec3   defocus_disk_v;       // Defocus disk vertical radius

    float  defocus_angle = 0;  // Variation angle of rays through each pixel
    float  focus_dist = 10;    // Distance from camera lookfrom point to plane of perfect focus
        
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float focus_dist, int image_width, int image_height) {
        initialize(lookfrom, lookat, vup, vfov, aspect, focus_dist, image_width, image_height);
    }
    __device__ void initialize(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float focus_dist, int image_width, int image_height);
    __device__ color ray_color(const ray& r, hittable** world);
    __device__ ray get_ray(float col, float row, curandState* local_rand_state);
};

__device__ void camera::initialize(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float focus_dist, int image_width, int image_height) {
    
    center = lookfrom;
    auto theta = degrees_to_radians(vfov);
    auto h = tan(theta / 2);
    auto viewport_height = 2 * h * focus_dist;
    auto viewport_width = viewport_height * (double(image_width) / image_height);

    // Calculate camera basis vectors
    w = unit_vector(lookfrom - lookat);       // Camera's backward direction
    u = unit_vector(cross(vup, w));           // Camera's right direction
    v = cross(w, u);                          // Camera's up direction

    // Calculate viewport edges
    vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
    vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

    // Pixel deltas
    pixel_delta_u = viewport_u / image_width;      // Pixel width
    pixel_delta_v = viewport_v / image_height;     // Pixel height

    // Upper-left corner of the viewport
    auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Calculate the camera defocus disk basis vectors.
    auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
}


__device__ ray camera::get_ray(float col, float row, curandState* local_rand_state) {
    auto offset = sample_square(local_rand_state);
    auto pixel_sample = pixel00_loc
        + ((col + offset.x()) * pixel_delta_u)
        + ((row + offset.y()) * pixel_delta_v);

    auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(center, defocus_disk_u, defocus_disk_v, local_rand_state);
    vec3 ray_direction = pixel_sample - ray_origin;

    return ray(ray_origin, ray_direction);
}

__device__ color ray_color(const ray& r, hittable** world, curandState* local_rand_state) {
    int maxDepth = 50;      // Ray tree maximum depth

    ray cur_ray = r;
    color cur_attenuation = color(1.0, 1.0, 1.0);

    for (int i = 0; i < maxDepth; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, interval(0.001f, FLT_MAX), rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}
#endif