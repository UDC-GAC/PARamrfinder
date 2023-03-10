cmake_minimum_required(VERSION 3.0)
project(PARamrfinder)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-O3")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)
find_package(GSL REQUIRED)
find_package(ZLIB REQUIRED)
find_package(HTSlib REQUIRED)

include_directories(${MPI_INCLUDE_PATH})
include_directories(${GSL_INCLUDE_DIRS})
include_directories(${HTSlib_INCLUDE_DIRS})
include_directories(${ZLIB_INCLUDE_DIRS})

enable_testing()

add_subdirectory(src/smithlab_cpp)
add_subdirectory(src/common)
add_subdirectory(src/amrfinder)
