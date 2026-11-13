cmake_minimum_required(VERSION 3.10)

project(homework_1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#mavx2 для SIMD; O3 для оптимизации
#fno-tree-vectorize выключает автовекторизацию,
#иначе компилятор сам векторизует "скалярные" циклы (и тогда нет смысла сравнивать)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -O3 -fno-tree-vectorize")
find_package(OpenMP REQUIRED)

add_executable(homework main.cpp)

target_link_libraries(homework PRIVATE OpenMP::OpenMP_CXX)