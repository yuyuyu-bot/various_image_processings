cmake_minimum_required(VERSION 3.21)
project(main LANGUAGES CXX CUDA)

# OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV version: " ${OpenCV_VERSION})
else()
    message(FATAL_ERROR "OpenCV not found.")
endif()

# source files
file(GLOB cpp_src  ./*.cpp)
file(GLOB cuda_src ./*.cu)
set(src ${cpp_src} ${cuda_src})

# include directories
set(inc
    ../include
    ../3rdparty/toml11
    ${OpenCV_INCLUDE_DIRS}
)

# libraries
set(lib
    ${OpenCV_LIBRARIES}
)

add_executable(${PROJECT_NAME} ${src})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME} PRIVATE ${inc})
target_link_libraries(${PROJECT_NAME} PRIVATE ${lib})
