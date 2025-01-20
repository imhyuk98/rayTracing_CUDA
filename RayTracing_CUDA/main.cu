#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <time.h>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void render(float* fb, int max_x, int max_y) {
    int index_i = threadIdx.x + blockIdx.x * blockDim.x;
    int index_j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((index_i >= max_x) || (index_j >= max_y)) return;
    int pixel_index = index_j * max_x * 3 + index_i * 3;
    fb[pixel_index + 0] = float(index_i) / max_x;
    fb[pixel_index + 1] = float(index_j) / max_y;
    fb[pixel_index + 2] = 0.2;
}

int main() {
    int image_width = 1200;
    int image_height = 600;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = image_width * image_height;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    // allocate FB
    float* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // CUDA Events for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Start recording time
    checkCudaErrors(cudaEventRecord(start, 0));

    // Render our buffer
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render << <blocks, threads >> > (fb, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

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

    // Output FB as Image
    std::cout << "P3\n" << image_width << " " << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; j--) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * 3 * image_width + i * 3;
            float r = fb[pixel_index + 0];
            float g = fb[pixel_index + 1];
            float b = fb[pixel_index + 2];
            int ir = int(255.99 * r);
            int ig = int(255.99 * g);
            int ib = int(255.99 * b);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaFree(fb));
}
