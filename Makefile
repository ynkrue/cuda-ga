NVCC ?= nvcc
NVCCFLAGS = -std=c++11 -Iinclude
LDFLAGS = -lm

HEADERS = include/kernels.cuh include/fitness.cuh include/config.hpp
APP_SRCS = $(wildcard src/*.cu) $(wildcard src/*.cpp)
APP_OBJS = $(patsubst %.cu,%.o,$(filter %.cu,$(APP_SRCS))) \
	    $(patsubst %.cpp,%.o,$(filter %.cpp,$(APP_SRCS)))
TEST_SRC = $(wildcard test/test_lj.cu)
TEST_OBJ = $(patsubst %.cu,%.o,$(TEST_SRC))

TARGET = bin/ga $(if $(TEST_SRC),bin/test_lj)

.PHONY: all clean check-toolchain

all: check-toolchain $(TARGET)

bin/ga: $(APP_OBJS)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

bin/test_lj: $(TEST_OBJ)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TEST_OBJ): $(TEST_SRC) $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(APP_OBJS) $(TEST_OBJ) $(TARGET)