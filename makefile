NVCC = nvcc
CFLAGS = -O2
TARGET = convolution
SRCS = convolution.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)
