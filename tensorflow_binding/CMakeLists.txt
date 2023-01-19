cmake_minimum_required(VERSION 3.21)

project(warp_rnnt_tf)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --std=c++14 -O2")
set(CMAKE_C_COMPILLER "usr/libexec/gcc")
include_directories(include)

FIND_PACKAGE(CUDA 6.5)
MESSAGE(STATUS "CUDA found ${CUDA_FOUND}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")

IF(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5)
  SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
ENDIF()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

IF (CUDA_VERSION GREATER 7.6)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
ENDIF()

IF (CUDA_VERSION GREATER 8.9)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
ENDIF()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
set(CMAKE_SKIP_RPATH TRUE)

MESSAGE(STATUS "Building shared library with GPU support")
CUDA_ADD_LIBRARY(warp_rnnt_core SHARED core.cu core_gather.cu)
