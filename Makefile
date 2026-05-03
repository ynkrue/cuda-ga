CXX = g++
CXXFLAGS = -std=c++11 -Wall
LDFLAGS = -lm

SRCS = src/main.cpp
OBJS = $(SRCS:.cpp=.o)

TARGET = ga

all: $(TARGET)
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)