#include "utile.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"

#define RND (curand_uniform(&local_rand_state))      
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

// Initialize cuRAND state for a single thread
__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);            // Custom seed : 1984
    }
}

// Initialize cuRAND state for each pixel, ensuring a unique random sequence for each pixel
__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;      // range [0, 1200]
    int j = threadIdx.y + blockIdx.y * blockDim.y;      // range [0, 679]

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);    // Each thread gets different seed number (Each ramdom number generation pattern must be independent per each thread)
}

__global__ void render(vec3* fb, int max_x, int max_y, int samples_per_pixel, camera** cam, hittable ** world, curandState* rand_state) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if ((col >= max_x) || (row >= max_y)) return;

    int pixel_index = row * max_x + col;
    curandState local_rand_state = rand_state[pixel_index];
    color pixel_color(0, 0, 0); 

    for (int sample = 0; sample < samples_per_pixel; sample++) {
        ray r = (*cam)->get_ray(col, row, &local_rand_state);
        pixel_color += ray_color(r, world, &local_rand_state);
    }

    pixel_color /= float(samples_per_pixel);
    
    pixel_color[0] = linear_to_gamma(pixel_color[0]);
    pixel_color[1] = linear_to_gamma(pixel_color[1]);
    pixel_color[2] = linear_to_gamma(pixel_color[2]);
        
    fb[pixel_index] = pixel_color;
}

__global__ void create_world(hittable** objects, hittable** world, camera** cam, int image_width, int image_height, curandState* rand_state, int num_hittables) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;
        // Sphere Setup
        curandState local_rand_state = *rand_state;
        objects[0] = new sphere(point3(0, -1000.0, -1), 1000, new lambertian(color(0.5, 0.5, 0.5)));

        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                point3 center(a + 0.9 * RND, 0.2, b + 0.9 * RND);

                if (choose_mat < 0.8) {
                    // diffuse
                    objects[++i] = new sphere(center, 0.2, new lambertian(color(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95) {
                    // metal
                    objects[++i] = new sphere(center, 0.2, new metal(color(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    // glass
                    objects[++i] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }

        objects[++i] = new sphere(point3(0, 1, 0), 1.0, new dielectric(1.5));
        objects[++i] = new sphere(point3(-4, 1, 0), 1.0, new lambertian(color(0.4, 0.2, 0.1)));
        objects[++i] = new sphere(point3(4, 1, 0), 1.0, new metal(color(0.7, 0.6, 0.5), 0.0));

        *rand_state = local_rand_state;
        *world = new hittable_list(objects, num_hittables);

        // Camera Setup
        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        vec3 vup(0, 1, 0),
        float dist_to_focus = 10.0;
        float aperture = 0.1;
        float vfov = 20.0f;
        float aspect = float(image_width) / float(image_height);

        *cam = new camera(lookfrom, lookat, vup, vfov, aspect, dist_to_focus, image_width, image_height);
    }
}

__global__ void free_world(hittable** d_object_list, hittable** world, camera** cam, int num_hittables) {
    for (int i = 0; i < num_hittables; i++) {
        delete ((sphere*)d_object_list[i])->mat_ptr;
        delete d_object_list[i];
    }
    delete* world;
    delete* cam;
}

int main() {
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 1200;
    int tx = 8;                                  
    int ty = 8;
    int samples_per_pixel = 500;

    // Calculate the image height, and ensure that it's at least 1.
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "with one thread per pixel.\n";

    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);

    // Allocate random states
    curandState* rand_rend;                // For rendering        
    checkCudaErrors(cudaMalloc((void**)&rand_rend, num_pixels * sizeof(curandState)));
    curandState* rand_world;                // For world creation
    checkCudaErrors(cudaMalloc((void**)&rand_world, 1 * sizeof(curandState)));

    // Allocate memory on CPU and GPU
    vec3* d_fb;                        // Device memory
    checkCudaErrors(cudaMalloc((void**)&d_fb, fb_size));
    vec3* h_fb = (vec3*)malloc(fb_size); // Host memory

    rand_init << <1, 1 >> > (rand_world);   // 2nd random state initialization
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hittable** d_object_list;
    int num_hittables = 22 * 22 + 1 + 3;         // Total object number
    checkCudaErrors(cudaMalloc((void**)&d_object_list, num_hittables * sizeof(hittable*)));

    hittable** world;
    checkCudaErrors(cudaMalloc((void**)&world, sizeof(hittable*)));

    camera** cam;
    checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera*)));

    // make our world of hittables
    create_world << <1, 1 >> > (d_object_list, world, cam, image_width, image_height, rand_world, num_hittables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // CUDA Events for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Start recording time
    checkCudaErrors(cudaEventRecord(start, 0));

    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);

    render_init << <blocks, threads >> > (image_width, image_height, rand_rend);   // 1st random state initialization
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render << <blocks, threads >> > (d_fb, image_width, image_height, samples_per_pixel, cam, world, rand_rend);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy results back to host

    checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));

    /////////////////////////////////////////////////////////////////////////////
    // Output File
    /////////////////////////////////////////////////////////////////////////////

    FILE* f = fopen("image3.ppm", "w");
    std::fprintf(f, "P3\n%d %d\n%d\n", image_width, image_height, 255);
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            auto ir = h_fb[pixel_index].r();
            auto ig = h_fb[pixel_index].g();
            auto ib = h_fb[pixel_index].b();


            static const interval intensity(0.000, 0.999);
            int rbyte = int(256 * intensity.clamp(ir));
            int gbyte = int(256 * intensity.clamp(ig));
            int bbyte = int(256 * intensity.clamp(ib));
            std::fprintf(f, "%d %d %d ", rbyte, gbyte, bbyte);
        }
    }
    std::clog << "\rDone.                 \n";

    // Stop recording time
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cerr << "Rendering took " << milliseconds / 1000.0f << " seconds.\n";

    // Destroy CUDA Events
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_object_list, world, cam, num_hittables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(d_object_list));
    checkCudaErrors(cudaFree(rand_rend));
    checkCudaErrors(cudaFree(rand_world));
    checkCudaErrors(cudaFree(d_fb));
    free(h_fb);

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}
