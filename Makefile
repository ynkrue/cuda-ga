NVCC      ?= nvcc
NVCCFLAGS  = -std=c++11 -Iinclude
LDFLAGS    = -lm

HEADERS   = include/kernels.cuh include/crossover.cuh include/utils.hpp
APP_SRCS  = $(wildcard src/*.cu) $(wildcard src/*.cpp)
APP_OBJS  = $(patsubst %.cu,%.o,$(filter %.cu,$(APP_SRCS))) \
            $(patsubst %.cpp,%.o,$(filter %.cpp,$(APP_SRCS)))

TARGET    = bin/ga

.PHONY: all clean

all: $(TARGET)

bin/ga: $(APP_OBJS) | bin
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

bin:
	mkdir -p bin

%.o: %.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(APP_OBJS) $(TARGET)
