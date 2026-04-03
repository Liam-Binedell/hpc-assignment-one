#!/usr/bin/env sh

make clean
make
./convolution lena_bw.pgm
gimp sharpened_global.pgm > /dev/null 2>&1 &
gimp sharpened_shared.pgm > /dev/null 2>&1 &
