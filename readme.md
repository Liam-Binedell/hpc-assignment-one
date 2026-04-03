# Problem 1: Image Convolution Using CUDA C/C++

## Requirements

- NVIDIA CUDA Toolkit (nvcc)
- A CUDA-capable GPU
- GNU Make

## Files

| File | Description |
|------|-------------|
| `convolution.cu` | Source code containing serial, global memory, and shared memory implementations |
| `makefile` | Build configuration |
| `run.sh` | Script to build and run the program |
| `gimp_run.sh` | Script to build, run, and open output images in GIMP |
| `lena_bw.pgm` | Input test image (512x512 greyscale) |

## Compiling

```bash
make
```

To clean the build:

```bash
make clean
```

## Running

```bash
./run.sh
```

This will compile the program and run it on `lena_bw.pgm`. Alternatively, run the binary directly:

```bash
./convolution lena_bw.pgm
```

## Output

The program applies three convolution filters using all three implementations (serial, CUDA global memory, CUDA shared memory) and writes the results to PGM files in the current directory:

| File | Description |
|------|-------------|
| `sharpened_serial.pgm` | Sharpening filter, serial |
| `sharpened_global.pgm` | Sharpening filter, CUDA global memory |
| `sharpened_shared.pgm` | Sharpening filter, CUDA shared memory |
| `embossed_serial.pgm` | Emboss filter, serial |
| `embossed_global.pgm` | Emboss filter, CUDA global memory |
| `embossed_shared.pgm` | Emboss filter, CUDA shared memory |
| `averaged_serial_NxN.pgm` | Averaging filter (serial) for N in {3,5,7,...,25} |
| `averaged_global_NxN.pgm` | Averaging filter (CUDA global) for N in {3,5,7,...,25} |
| `averaged_shared_NxN.pgm` | Averaging filter (CUDA shared) for N in {3,5,7,...,25} |

Execution times for each filter and implementation are printed to stdout.

## Viewing Output Images

GIMP can be used to view PGM files. To build, run, and open selected outputs in GIMP automatically:

```bash
./gimp_run.sh
```
