NVCC ?= nvcc
NVCCFLAGS = -std=c++11 -Iinclude
LDFLAGS = -lm

HEADERS = include/kernels.cuh include/fitness.cuh include/config.hpp
SRCS = $(wildcard src/*.cu) $(wildcard src/*.cpp)
OBJS = $(patsubst %.cu,%.o,$(filter %.cu,$(SRCS))) \
       $(patsubst %.cpp,%.o,$(filter %.cpp,$(SRCS)))

TARGET = ga

.PHONY: all clean check-toolchain

all: check-toolchain $(TARGET)

check-toolchain:
	@command -v $(NVCC) >/dev/null 2>&1 || { \
		echo "Error: compiler '$(NVCC)' not found."; \
		echo "Install CUDA (nvcc) or run 'make NVCC=/path/to/nvcc'."; \
		exit 1; \
	}

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)