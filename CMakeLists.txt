cmake_minimum_required(VERSION 3.21)
project(various_image_processing)

option(BUILD_SAMPLES "Build samples" ON)
option(BUILD_TESTS   "Build library" ON)

if (BUILD_SAMPLES)
    add_subdirectory(sample)
endif()

if (BUILD_TESTS)
    add_subdirectory(test)
endif()
