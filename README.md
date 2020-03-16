# ellipse_detector4aMatch-2020
Single camera ellipse detector

# Operating system & Environment
Linux only for now

# Dependency
opencv-3.4.6 or higher

# How to compile
1、Create a CMakeLists.txt at the workspace dir, and fill it with words below:
```
cmake_minimum_required(VERSION 2.8)
project( detect )
find_package( OpenCV 3.4.6 REQUIRED )
add_executable( detect detect.cpp )
target_link_libraries( detect ${OpenCV_LIBS})
```
2、Open the terminal and input:
```
cmake .
make
```
3、Wait for compiling

# How to use
In terminal inputs:
```
./detect
```

# Going to do
1、Adopt kNN algorithm

