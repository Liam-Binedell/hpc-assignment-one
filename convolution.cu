#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define BUF 64

typedef struct {
    int width;
    int height;
    int max_val;
    unsigned char *data;
} Pgm_Image;

__constant__ int d_sharpen_filter[9];
__constant__ int d_emboss_filter[25];

void copy_image(Pgm_Image *dest, Pgm_Image *src) {
    int size;
    size = src->width * src->height;
    dest->width = src->width;
    dest->height = src->height;
    dest->max_val = src->max_val;
    dest->data = (unsigned char *)malloc(size);
    memcpy(dest->data, src->data, size);
}

// Me doing this the C way because I can't find documentation
// on loading .pgm files with the cuda sdk.
int load_pgm_image(Pgm_Image *image, const char *path) {
    int size;
    char c, magic[3];
    FILE *f;

    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open input file\n");
        return 1;
    }

    // Scan first two characters to identify if file is binary pgm file.
    // First two characters will be "P5" if this is true
    fscanf(f, "%2s", magic);
    if (strcmp("P5", magic)) {
        fprintf(stderr, "Not a binary PGM file");
        return 1;
    }

    // This should consume all the comments leading up to the info that we want.
    fscanf(f, " %c", &c);
    while (c == '#') {
        while (fgetc(f) != '\n')
            ;
        fscanf(f, "%c", &c);
    }
    ungetc(c, f);

    // These three fields are vital metadata about the file that will be useful
    // for copying image over to device memory.
    fscanf(f, "%d %d %d", &image->width, &image->height, &image->max_val);
    fgetc(f);

    // Copies image data into Pgm_Image struct and closes input file.
    size = image->width * image->height;
    image->data = (unsigned char *)malloc(size);
    fread(image->data, sizeof(unsigned char), size, f);
    fclose(f);
    return 0;
}

int write_pgm_image(Pgm_Image *image, const char *path) {
    int size;
    FILE *f;

    f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open output file\n");
        return 1;
    }

    fprintf(f, "%s\n", "P5");
    fprintf(f, "%d %d %d\n", image->width, image->height, image->max_val);

    size = image->width * image->height;
    fwrite(image->data, sizeof(unsigned char), size, f);
    fclose(f);
    return 0;
}

/*

  KERNELS

 */

void sharpen_image_serial(Pgm_Image *image) {
    int size, sum, index, filter_index, conv_index;
    int sharpen_filter[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
    unsigned char *output;

    size = image->width * image->height;
    output = (unsigned char *)malloc(size);

    for (int row = 0; row < image->height; row++) {
        for (int col = 0; col < image->width; col++) {
            index = row * image->width + col;

            sum = 0;
            filter_index = 0;
            for (int i = row - 1; i <= row + 1; i++) {
                for (int j = col - 1; j <= col + 1; j++) {
                    conv_index = i * image->width + j;
                    if (i < 0 || i >= image->height || j < 0 ||
                        j >= image->width) {
                        filter_index++;
                        continue;
                    }
                    sum += image->data[conv_index] *
                           sharpen_filter[filter_index++];
                }
            }
            output[index] = sum > 255 ? 255 : sum < 0 ? 0 : sum;
        }
    }

    free(image->data);
    image->data = output;
}

__global__ void sharpen_image_global(int width, int height, unsigned char *data, unsigned char *output) {
    int x, y, index, sum, filter_index, conv_index;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    index = y * width + x;

    sum = 0;
    filter_index = 0;
    for (int i = y - 1; i <= y + 1; i++) {
        for (int j = x - 1; j <= x + 1; j++) {
            conv_index = i * width + j;
            if (i < 0 || i >= height || j < 0 || j >= width) {
                filter_index++;
                continue;
            }
            sum += data[conv_index] * d_sharpen_filter[filter_index++];
        }
    }
    output[index] = sum > 255 ? 255 : sum < 0 ? 0 : sum;
}

__global__ void sharpen_image_shared_memory(int width, int height, unsigned char *data, unsigned char *output) {
    extern __shared__ unsigned char tile[];

    /* DISCLAIMER */
    // The following section of code was provided by Claude.ai because I'm lazy.
    // Each thread is responsible for loading its own pixel into shared memory
    // while also loading halo pixels to facilitate the indices that will be
    // accessed by the convolution filter.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

// Tile dimensions include halo
    int tile_width = blockDim.x + 2;

// Load the pixel this thread is responsible for (offset by 1 for halo)
    int tile_x = tx + 1;
    int tile_y = ty + 1;

// Load centre pixel
    tile[tile_y * tile_width + tile_x] = (x < width && y < height) ? data[y * width + x] : 0;

// Load left halo
    if (tx == 0)
        tile[tile_y * tile_width + 0] = (x > 0 && y < height) ? data[y * width + (x - 1)] : 0;

// Load right halo
    if (tx == blockDim.x - 1)
        tile[tile_y * tile_width + tile_x + 1] = (x + 1 < width && y < height) ? data[y * width + (x + 1)] : 0;

// Load top halo
    if (ty == 0)
        tile[0 * tile_width + tile_x] = (y > 0 && x < width) ? data[(y - 1) * width + x] : 0;

// Load bottom halo
    if (ty == blockDim.y - 1)
        tile[(tile_y + 1) * tile_width + tile_x] = (y + 1 < height && x < width) ? data[(y + 1) * width + x] : 0;

// Load corners
    if (tx == 0 && ty == 0)
        tile[0] = (x > 0 && y > 0) ? data[(y - 1) * width + (x - 1)] : 0;
    if (tx == blockDim.x - 1 && ty == 0)
        tile[tile_width - 1] = (x + 1 < width && y > 0) ? data[(y - 1) * width + (x + 1)] : 0;
    if (tx == 0 && ty == blockDim.y - 1)
        tile[(tile_y + 1) * tile_width] = (x > 0 && y + 1 < height) ? data[(y + 1) * width + (x - 1)] : 0;
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1)
        tile[(tile_y + 1) * tile_width + tile_width - 1] = (x + 1 < width && y + 1 < height) ? data[(y + 1) * width + (x + 1)] : 0;

    // END OF CLAUDE CODE
    __syncthreads();
    int index, sum, filter_index;
    index = y * width + x;
    sum = 0;
    filter_index = 0;
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            if (y + i < 0 || y + i >= height || x + j < 0 || x + j >= width) {
                filter_index++;
                continue;
            }
            sum += tile[(tile_y + i) * tile_width + (tile_x + j)] * d_sharpen_filter[filter_index++];
        }
    }
    output[index] = sum > 255 ? 255 : sum < 0 ? 0 : sum;
}

void emboss_image_serial(Pgm_Image *image) {
    int size, sum, idx, filter_index, index;
    int emboss_filter[25] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1};
    unsigned char *output;

    size = image->width * image->height;
    output = (unsigned char *)malloc(size);

    for (int row = 0; row < image->height; row++) {
        for (int col = 0; col < image->width; col++) {
            idx = row * image->width + col;

            sum = 0;
            filter_index = 0;
            for (int i = row - 2; i <= row + 2; i++) {
                for (int j = col - 2; j <= col + 2; j++) {
                    index = i * image->width + j;
                    if (i < 0 || i >= image->height || j < 0 || j >= image->width) {
                        filter_index++;
                        continue;
                    }
                    sum += image->data[index] * emboss_filter[filter_index++];
                }
            }
            output[idx] = sum > 255 ? 255 : sum < 0 ? 0 : sum;
        }
    }

    free(image->data);
    image->data = output;
}

__global__ void emboss_image_global(int width, int height, unsigned char *data, unsigned char *output) {
    int x, y, idx, sum, filter_index, index;
    x = blockIdx.x * blockDim.x + threadIdx.x;
    y = blockIdx.y * blockDim.y + threadIdx.y;
    idx = y * width + x;

    sum = 0;
    filter_index = 0;
    for (int i = y - 2; i <= y + 2; i++) {
        for (int j = x - 2; j <= x + 2; j++) {
            index = i * width + j;
            if (i < 0 || i >= height || j < 0 || j >= width) {
                filter_index++;
                continue;
            }
            sum += data[index] * d_emboss_filter[filter_index++];
        }
    }
    output[idx] = sum > 255 ? 255 : sum < 0 ? 0 : sum;
}

__global__ void emboss_image_shared(int width, int height, unsigned char *data, unsigned char *output) {
    extern __shared__ unsigned char tile[];

    /* DISCLAIMER */
    // The following section of code was provided by Claude.ai because I'm lazy.
    // Each thread is responsible for loading its own pixel into shared memory
    // while also loading halo pixels to facilitate the indices that will be
    // accessed by the convolution filter.

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Tile dimensions include halo of 2 on each side
    int tile_width = blockDim.x + 4;
    int tile_height = blockDim.y + 4;

    // Position inside tile (halo offset = 2)
    int tile_x = tx + 2;
    int tile_y = ty + 2;

    // Load centre pixel (safe for out-of-bounds)
    tile[tile_y * tile_width + tile_x] = (x < width && y < height) ? data[y * width + x] : 0;

    // Load left halo (2 pixels)
    if (tx == 0) {
        tile[tile_y * tile_width + 0] = (x > 0 && y < height) ? data[y * width + (x - 1)] : 0;
        if (tx == 0) // second left halo pixel
            tile[tile_y * tile_width + 1] = (x > 1 && y < height) ? data[y * width + (x - 2)] : 0;
    }
    // Load right halo (2 pixels)
    if (tx == blockDim.x - 1) {
        tile[tile_y * tile_width + tile_x + 1] = (x + 1 < width && y < height) ? data[y * width + (x + 1)] : 0;
        tile[tile_y * tile_width + tile_x + 2] = (x + 2 < width && y < height) ? data[y * width + (x + 2)] : 0;
    }

    // Load top halo (2 rows)
    if (ty == 0) {
        tile[(tile_y - 1) * tile_width + tile_x] = (y > 0 && x < width) ? data[(y - 1) * width + x] : 0;
        tile[(tile_y - 2) * tile_width + tile_x] = (y > 1 && x < width) ? data[(y - 2) * width + x] : 0;
    }
    // Load bottom halo (2 rows)
    if (ty == blockDim.y - 1) {
        tile[(tile_y + 1) * tile_width + tile_x] = (y + 1 < height && x < width) ? data[(y + 1) * width + x] : 0;
        tile[(tile_y + 2) * tile_width + tile_x] = (y + 2 < height && x < width) ? data[(y + 2) * width + x] : 0;
    }

    // Corners (top‑left 2x2, top‑right 2x2, bottom‑left 2x2, bottom‑right 2x2)
    if (tx == 0 && ty == 0) {
        tile[0] = (x > 0 && y > 0) ? data[(y - 1) * width + (x - 1)] : 0;
        tile[1] = (x > 1 && y > 0) ? data[(y - 1) * width + (x - 2)] : 0;
        tile[tile_width] = (x > 0 && y > 1) ? data[(y - 2) * width + (x - 1)] : 0;
        tile[tile_width + 1] = (x > 1 && y > 1) ? data[(y - 2) * width + (x - 2)] : 0;
    }
    if (tx == blockDim.x - 1 && ty == 0) {
        tile[tile_width - 2] = (x + 1 < width && y > 0) ? data[(y - 1) * width + (x + 1)] : 0;
        tile[tile_width - 1] = (x + 2 < width && y > 0) ? data[(y - 1) * width + (x + 2)] : 0;
        tile[2 * tile_width - 2] = (x + 1 < width && y > 1) ? data[(y - 2) * width + (x + 1)] : 0;
        tile[2 * tile_width - 1] = (x + 2 < width && y > 1) ? data[(y - 2) * width + (x + 2)] : 0;
    }
    if (tx == 0 && ty == blockDim.y - 1) {
        tile[(tile_height - 2) * tile_width] = (x > 0 && y + 1 < height) ? data[(y + 1) * width + (x - 1)] : 0;
        tile[(tile_height - 2) * tile_width + 1] = (x > 1 && y + 1 < height) ? data[(y + 1) * width + (x - 2)] : 0;
        tile[(tile_height - 1) * tile_width] = (x > 0 && y + 2 < height) ? data[(y + 2) * width + (x - 1)] : 0;
        tile[(tile_height - 1) * tile_width + 1] = (x > 1 && y + 2 < height) ? data[(y + 2) * width + (x - 2)] : 0;
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        tile[(tile_height - 2) * tile_width + (tile_width - 2)] = (x + 1 < width && y + 1 < height) ? data[(y + 1) * width + (x + 1)] : 0;
        tile[(tile_height - 2) * tile_width + (tile_width - 1)] = (x + 2 < width && y + 1 < height) ? data[(y + 1) * width + (x + 2)] : 0;
        tile[(tile_height - 1) * tile_width + (tile_width - 2)] = (x + 1 < width && y + 2 < height) ? data[(y + 2) * width + (x + 1)] : 0;
        tile[(tile_height - 1) * tile_width + (tile_width - 1)] = (x + 2 < width && y + 2 < height) ? data[(y + 2) * width + (x + 2)] : 0;
    }

    // END OF CLAUDE CODE //

    __syncthreads();
    int index, sum, filter_index;
    index = y * width + x;
    sum = 0;
    filter_index = 0;
    for (int i = -2; i <= 2; i++) {
        for (int j = -2; j <= 2; j++) {
            if (y + i < 0 || y + i >= height || x + j < 0 || x + j >= width) {
                filter_index++;
                continue;
            }
            sum += tile[(tile_y + i) * tile_width + (tile_x + j)] * d_emboss_filter[filter_index++];
        }
    }
    output[index] = sum > 255 ? 255 : sum < 0 ? 0 : sum;
}

void average_image_serial(Pgm_Image *image, int k) {
    int width = image->width;
    int height = image->height;
    int size = width * height;
    unsigned char *output = (unsigned char *)malloc(size);
    int offset = k / 2;

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int sum = 0;
            for (int i = -offset; i <= offset; i++) {
                for (int j = -offset; j <= offset; j++) {
                    int r = row + i;
                    int c = col + j;
                    if (r >= 0 && r < height && c >= 0 && c < width)
                        sum += image->data[r * width + c];
                }
            }
            output[row * width + col] = (unsigned char)(sum / (k * k));
        }
    }
    free(image->data);
    image->data = output;
}

__global__ void average_image_global(int width, int height, unsigned char *data, unsigned char *output, int k) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int offset = k / 2;
    int sum = 0;
    int count = 0;
    for (int i = -offset; i <= offset; i++) {
        for (int j = -offset; j <= offset; j++) {
            int r = y + i;
            int c = x + j;
            if (r >= 0 && r < height && c >= 0 && c < width) {
                sum += data[r * width + c];
                count++;
            }
        }
    }
    output[y * width + x] = (unsigned char)(sum / (k * k));
}

__global__ void average_image_shared(int width, int height, unsigned char *data, unsigned char *output, int k) {
    extern __shared__ unsigned char tile[];
    int offset = k / 2;
    int tile_width = blockDim.x + 2 * offset;
    int tile_height = blockDim.y + 2 * offset;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    // Cooperatively load entire tile including halo using strided loop
    int thread_id = ty * blockDim.x + tx;
    int block_size = blockDim.x * blockDim.y;
    int tile_size = tile_width * tile_height;

    for (int i = thread_id; i < tile_size; i += block_size) {
        int tile_row = i / tile_width;
        int tile_col = i % tile_width;
        int global_row = blockIdx.y * blockDim.y + tile_row - offset;
        int global_col = blockIdx.x * blockDim.x + tile_col - offset;

        if (global_row >= 0 && global_row < height && global_col >= 0 && global_col < width)
            tile[i] = data[global_row * width + global_col];
        else
            tile[i] = 0;
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    int sum = 0;
    for (int i = -offset; i <= offset; i++) {
        for (int j = -offset; j <= offset; j++) {
            int r = y + i;
            int c = x + j;
            if (r >= 0 && r < height && c >= 0 && c < width)
                sum += tile[(ty + offset + i) * tile_width + (tx + offset + j)];
        }
    }
    output[y * width + x] = sum / (k * k);
}

/*

  KERNEL CALLS

 */

double sharpness_serial(Pgm_Image *image) {
    struct timeval start, end;

    gettimeofday(&start, NULL);
    sharpen_image_serial(image);
    gettimeofday(&end, NULL);

    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

double sharpness_global(Pgm_Image *image) {
    struct timeval start, end;
    int h_filter[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
    int size;
    unsigned char *d_data, *d_output;
    dim3 n_blocks, n_threads;

    size = image->width * image->height;
    n_blocks = dim3(32, 32);
    n_threads = dim3(16, 16);

    cudaMemcpyToSymbol(d_sharpen_filter, h_filter, 9 * sizeof(int));
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, image->data, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, size);

    gettimeofday(&start, NULL);
    sharpen_image_global<<<n_blocks, n_threads>>>(image->width, image->height, d_data, d_output);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaMemcpy(image->data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_output);

    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

double sharpness_shared(Pgm_Image *image) {
    struct timeval start, end;
    int h_filter[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
    int size;
    unsigned char *d_data, *d_output;
    dim3 n_blocks, n_threads;

    size = image->width * image->height;
    n_threads = dim3(16, 16);
    n_blocks = dim3((image->width + 15) / 16, (image->height + 15) / 16);

    cudaMemcpyToSymbol(d_sharpen_filter, h_filter, 9 * sizeof(int));
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, image->data, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, size);

    gettimeofday(&start, NULL);
    sharpen_image_shared_memory<<<n_blocks, n_threads, 18 * 18 * sizeof(unsigned char)>>>(image->width, image->height, d_data, d_output);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaMemcpy(image->data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_output);
    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

double emboss_serial(Pgm_Image *image) {
    struct timeval start, end;

    gettimeofday(&start, NULL);
    emboss_image_serial(image);
    gettimeofday(&end, NULL);

    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

double emboss_global(Pgm_Image *image) {
    struct timeval start, end;
    int h_filter[25] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1};
    int size;
    unsigned char *d_data, *d_output;
    dim3 n_blocks, n_threads;

    size = image->width * image->height;
    n_blocks = dim3(32, 32);
    n_threads = dim3(16, 16);

    cudaMemcpyToSymbol(d_emboss_filter, h_filter, 25 * sizeof(int));
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, image->data, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, size);

    gettimeofday(&start, NULL);
    emboss_image_global<<<n_blocks, n_threads>>>(image->width, image->height, d_data, d_output);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaMemcpy(image->data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_output);

    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

double emboss_shared(Pgm_Image *image) {
    struct timeval start, end;
    int h_filter[25] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1};
    int size;
    unsigned char *d_data, *d_output;
    dim3 n_blocks, n_threads;

    size = image->width * image->height;
    n_threads = dim3(16, 16);
    n_blocks = dim3((image->width + 15) / 16, (image->height + 15) / 16);

    cudaMemcpyToSymbol(d_emboss_filter, h_filter, 25 * sizeof(int));
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, image->data, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, size);

    gettimeofday(&start, NULL);
    emboss_image_shared<<<n_blocks, n_threads, 20 * 20 * sizeof(unsigned char)>>>(image->width, image->height, d_data, d_output);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaMemcpy(image->data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_output);
    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

double average_serial(Pgm_Image *image, int k) {
    struct timeval start, end;

    gettimeofday(&start, NULL);
    average_image_serial(image, k);
    gettimeofday(&end, NULL);

    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

double average_global(Pgm_Image *image, int k) {
    struct timeval start, end;
    int size;
    unsigned char *d_data, *d_output;
    dim3 n_blocks, n_threads;

    size = image->width * image->height;
    n_threads = dim3(16, 16);
    n_blocks = dim3((image->width + 15) / 16, (image->height + 15) / 16);

    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, image->data, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, size);

    gettimeofday(&start, NULL);
    average_image_global<<<n_blocks, n_threads>>>(image->width, image->height, d_data, d_output, k);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaMemcpy(image->data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_output);
    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

double average_shared(Pgm_Image *image, int k) {
    struct timeval start, end;
    int size, offset;
    unsigned char *d_data, *d_output;
    dim3 n_blocks, n_threads;

    size = image->width * image->height;
    offset = k / 2;
    n_threads = dim3(16, 16);
    n_blocks = dim3((image->width + 15) / 16, (image->height + 15) / 16);
    size_t mem_size = (16 + 2 * offset) * (16 + 2 * offset) * sizeof(unsigned char);

    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, image->data, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output, size);

    gettimeofday(&start, NULL);
    average_image_shared<<<n_blocks, n_threads, mem_size>>>(image->width, image->height, d_data, d_output, k);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaMemcpy(image->data, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_output);
    return (end.tv_sec - start.tv_sec) +
           (end.tv_usec - start.tv_usec) / 1000000.0;
}

int main(int argc, char **argv) {
    Pgm_Image image, copy;
    char average_path[BUF];
    copy.data = NULL;

    if (argc != 2) {
        printf("Usage: %s [input_file].pgm\n", argv[0]);
        return 1;
    }

    if (load_pgm_image(&image, argv[1]) != 0) {
        return 1;
    }

    /*
     SHARPENING
     */
    copy_image(&copy, &image);
    printf("Serial sharpening time: %f seconds\n", sharpness_serial(&copy));
    write_pgm_image(&copy, "output/sharpened_serial.pgm");

    free(copy.data);
    copy_image(&copy, &image);
    printf("Cuda global sharpening time: %f seconds\n", sharpness_global(&copy));
    write_pgm_image(&copy, "output/sharpened_global.pgm");

    free(copy.data);
    copy_image(&copy, &image);
    printf("Cuda shared sharpening time: %f seconds\n", sharpness_shared(&copy));
    write_pgm_image(&copy, "output/sharpened_shared.pgm");
    printf("\n\n");

    /*
     EMBOSSING
     */
    free(copy.data);
    copy_image(&copy, &image);
    printf("Serial embossing time: %f seconds\n", emboss_serial(&copy));
    write_pgm_image(&copy, "output/embossed_serial.pgm");

    free(copy.data);
    copy_image(&copy, &image);
    printf("Cuda global embossing time: %f seconds\n", emboss_global(&copy));
    write_pgm_image(&copy, "output/embossed_global.pgm");

    free(copy.data);
    copy_image(&copy, &image);
    printf("Cuda shared embossing time: %f seconds\n", emboss_shared(&copy));
    write_pgm_image(&copy, "output/embossed_shared.pgm");
    printf("\n\n");

    /*
     AVERAGING
     */
    for (int k = 3; k <= 25; k += 2) {
        free(copy.data);
        copy_image(&copy, &image);
        printf("Serial averaging time with kernel size %d: %f seconds\n", k, average_serial(&copy, k));
        snprintf(average_path, BUF, "output/averages/averaged_serial_%dx%d.pgm", k, k);
        write_pgm_image(&copy, average_path);

        free(copy.data);
        copy_image(&copy, &image);
        printf("Cuda global averaging time with kernel size %d: %f seconds\n", k, average_global(&copy, k));
        snprintf(average_path, BUF, "output/averages/averaged_global_%dx%d.pgm", k, k);
        write_pgm_image(&copy, average_path);

        free(copy.data);
        copy_image(&copy, &image);
        printf("Cuda shared averaging time with kernel size %d: %f seconds\n", k, average_shared(&copy, k));
        snprintf(average_path, BUF, "output/averages/averaged_shared_%dx%d.pgm", k, k);
        write_pgm_image(&copy, average_path);
        printf("\n");
    }

    free(image.data);
    free(copy.data);
    return 0;
}
