cmake_minimum_required(VERSION 3.21)
project(test LANGUAGES CXX CUDA)

# OpenCV
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV version: " ${OpenCV_VERSION})
else()
    message(FATAL_ERROR "OpenCV not found.")
endif()

# googletest
find_package(GTest REQUIRED)
if (GTest_FOUND)
else()
    message(FATAL_ERROR "googletest not found.")
endif()

# source files
file(GLOB src ${CMAKE_SOURCE_DIR}/src/*.c*)
file(GLOB test_src ./*.c*)
set(src ${src} ${test_src})

# include directories
set(inc
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)

# libraries
set(lib
    ${OpenCV_LIBRARIES}
    GTest::GTest
    GTest::Main
)

add_executable(${PROJECT_NAME} ${src})

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_include_directories(${PROJECT_NAME} PRIVATE ${inc})
target_link_libraries(${PROJECT_NAME} PRIVATE ${lib})
