#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "ray.h"

class camera {
public:
    int    image_height;   // Rendered image height
    point3 center;         // Camera center
    point3 pixel00_loc;    // Location of pixel 0, 0
    vec3   pixel_delta_u;  // Offset to pixel to the right
    vec3   pixel_delta_v;  // Offset to pixel below
    vec3   unit_direction;
    float  aspect_ratio = 16.0/9.0;  // Ratio of image width over height
    int    image_width = 1200;  // Rendered image width in pixel count

    __device__ camera() {
        initialize();
    }
    __device__ void initialize();
    __device__ color ray_color(const ray& r, hittable** world);
    __device__ ray get_ray(float col, float row);
};

__device__ void camera::initialize() {
    // Camera settings
    point3 lookfrom(0, 0, 0);  // Camera position
    point3 lookat(0, 0, -1);   // Scene center
    vec3   vup(0, 1, 0);       // Up direction

    image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    center = lookfrom;

    // Determine viewport dimensions
    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * (double(image_width) / image_height);

    // Calculate camera basis vectors
    vec3 w = unit_vector(lookfrom - lookat);       // Camera's backward direction
    vec3 u = unit_vector(cross(vup, w));           // Camera's right direction
    vec3 v = cross(w, u);                          // Camera's up direction

    // Calculate viewport edges
    auto viewport_u = viewport_width * u;          // Horizontal axis of the viewport
    auto viewport_v = viewport_height * v;         // Vertical axis of the viewport

    // Pixel deltas
    pixel_delta_u = viewport_u / image_width;      // Pixel width
    pixel_delta_v = viewport_v / image_height;     // Pixel height

    // Upper-left corner of the viewport
    auto viewport_upper_left =
        center - w * focal_length - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Default ray direction for debugging
    unit_direction = unit_vector(lookat - lookfrom);
}


__device__ ray camera::get_ray(float col, float row) {
    vec3 ray_origin = center;
    vec3 cur_pixel = pixel00_loc + col * pixel_delta_u + row * pixel_delta_v;
    vec3 ray_direction = cur_pixel - ray_origin;

    return ray(ray_origin, ray_direction);
}

__device__ color ray_color(const ray& r, hittable** world) {
    hit_record rec;

    ray cur_ray = r;

    if ((*world)->hit(cur_ray, interval(0, infinity), rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }

    vec3 unit_direction = unit_vector(cur_ray.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

#endif