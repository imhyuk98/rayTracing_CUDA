#include "utile.h"

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__constant__ int d_samples_per_pixel;
__constant__ float d_pixel_samples_scale;

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

__global__ void render(vec3* fb, int max_x, int max_y, int samples_per_pixel, camera** cam, hittable ** d_world, curandState* rand_state) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int depth = 50;
    int pixel_index = row * max_x + col;
    curandState local_rand_state = rand_state[pixel_index];

    if ((col >= max_x) || (row >= max_y)) return;

    float u = float(col) / float(max_x);
    float v = float(row) / float(max_y);

    color pixel_color(0, 0, 0); 
    for (int sample = 0; sample < samples_per_pixel; sample++) {
        ray r = (*cam)->get_ray(col, row, &local_rand_state);
        pixel_color += ray_color(r, d_world, &local_rand_state);
    }
    pixel_color /= float(samples_per_pixel);
    
    pixel_color[0] = linear_to_gamma(pixel_color[0]);
    pixel_color[1] = linear_to_gamma(pixel_color[1]);
    pixel_color[2] = linear_to_gamma(pixel_color[2]);
        
    fb[pixel_index] = pixel_color;
}

//__global__ void render(vec3* fb, int max_x, int max_y, camera** cam, hittable** d_world) {
//    // 1D 스레드와 블록 구성에서 글로벌 인덱스 계산
//    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
//
//    if (thread_id >= max_x * max_y) return;
//
//    // 1D 인덱스를 2D (col, row)로 변환w 1
//    int col = thread_id % max_x;  // 열 = thread_id를 이미지 가로 크기로 나눈 나머지
//    int row = thread_id / max_x;  // 행 = thread_id를 이미지 가로 크기로 나눈 몫
//
//    // 픽셀 인덱스 계산 (1D 배열 접근용)
//    int pixel_index = row * max_x + col;
//
//    // u와 v 계산
//    float u = float(col) / float(max_x);
//    float v = float(row) / float(max_y);
//
//    // Ray 생성 및 색상 계산
//    ray r = (*cam)->get_ray(col, row);
//    fb[pixel_index] = ray_color(r, d_world);
//}

__global__ void create_world(hittable** objects, hittable** d_world, camera** cam) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        objects[0] = new sphere(vec3(0, 0, -1), 0.5);
        objects[1] = new sphere(vec3(0, -100.5, -1), 100);
        *d_world = new hittable_list(objects, 2);
        *cam = new camera();
    }
}

__global__ void free_world(hittable** d_object_list, hittable** d_world, camera** cam) {
    delete d_object_list[0];
    delete d_object_list[1];
    delete* d_world;
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
    curandState* d_rand_state_1;                // For rendering        
    checkCudaErrors(cudaMalloc((void**)&d_rand_state_1, num_pixels * sizeof(curandState)));
    curandState* d_rand_state_2;                // For world creation
    checkCudaErrors(cudaMalloc((void**)&d_rand_state_2, 1 * sizeof(curandState)));

    // Allocate memory on CPU and GPU
    vec3* d_fb;                        // Device memory
    checkCudaErrors(cudaMalloc((void**)&d_fb, fb_size));

    rand_init << <1, 1 >> > (d_rand_state_2);   // 2nd random state initialization
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hittable** d_object_list;
    int num_hittables = 3;         // Total object number
    checkCudaErrors(cudaMalloc((void**)&d_object_list, num_hittables * sizeof(hittable*)));

    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

    camera** cam;
    checkCudaErrors(cudaMalloc((void**)&cam, sizeof(camera*)));

    // make our world of hittables
    create_world << <1, 1 >> > (d_object_list, d_world, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // CUDA Events for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Start recording time
    checkCudaErrors(cudaEventRecord(start, 0));

    // Render our buffer
    // 안 되는거
    //int threads_per_block = 256; // Optimal number of threads per block
    //int blocks_per_grid = (num_pixels + threads_per_block - 1) / threads_per_block; // Calculate grid size
    //render << <blocks_per_grid, threads_per_block >> > (d_fb, image_width, image_height, cam, d_world);

    // 일반적c

    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);

    render_init << <blocks, threads >> > (image_width, image_height, d_rand_state_1);   // 1st random state initialization
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render << <blocks, threads >> > (d_fb, image_width, image_height, samples_per_pixel, cam, d_world, d_rand_state_1);

    // 안 되는거 보완
    //int threads_per_block = 256;
    //int blocks_per_grid = (image_width * image_height + threads_per_block - 1) / threads_per_block;
    //render << <blocks_per_grid, threads_per_block >> > (d_fb, image_width, image_height, cam, d_world);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Copy results back to host
    vec3* h_fb = (vec3*)malloc(fb_size); // Host memory
    checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));


    /////////////////////////////////////////////////////////////////////////////
    // Output File
    /////////////////////////////////////////////////////////////////////////////

    FILE* f = fopen("image3.ppm", "w");
    std::fprintf(f, "P3\n%d %d\n%d\n", image_width, image_height, 255);
    for (int j = image_height - 1; j >= 0; j--) {
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

    /////////////////////////////////////////////////////////////////////////////
    // Output Console
    /////////////////////////////////////////////////////////////////////////////

    //std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    //for (int j = image_height - 1; j >= 0; j--) {
    //    for (int i = 0; i < image_width; i++) {
    //        size_t pixel_index = j * image_width + i;
    //        int ir = int(255.99 * h_fb[pixel_index].r());
    //        int ig = int(255.99 * h_fb[pixel_index].g());
    //        int ib = int(255.99 * h_fb[pixel_index].b());
    //        std::cout << ir << " " << ig << " " << ib << "\n";
    //    }
    //}

    //std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    //for (int j = image_height - 1; j >= 0; j--) {
    //    for (int i = 0; i < image_width; i++) {
    //        size_t pixel_index = j * image_width + i;
    //        auto ir = h_fb[pixel_index].r();
    //        auto ig = h_fb[pixel_index].g();
    //        auto ib = h_fb[pixel_index].b();

    //        static const interval intensity(0.000, 0.999);
    //        int rbyte = int(255.99 * intensity.clamp(ir));
    //        int gbyte = int(255.99 * intensity.clamp(ig));
    //        int bbyte = int(255.99 * intensity.clamp(ib));
    //        std::cout << rbyte << " " << gbyte << " " << bbyte << "\n";
    //    }
    //}

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
    free_world << <1, 1 >> > (d_object_list, d_world, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_object_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_fb));

    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
}
