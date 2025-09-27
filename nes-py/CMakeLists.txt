cmake_minimum_required(VERSION 3.15)
project(nes_py)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific settings
if(MSVC)
    # Windows with Visual Studio
    add_compile_options(/bigobj)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
elseif(APPLE)
    # macOS specific settings
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum macOS version")
    # Use clang++ on macOS
    if(NOT CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "clang++")
    endif()
elseif(UNIX)
    # Linux/Unix specific settings
    # Use g++ on Linux
    if(NOT CMAKE_CXX_COMPILER)
        set(CMAKE_CXX_COMPILER "g++")
    endif()
endif()

# Collect source files
file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/nes_py/nes/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nes_py/nes/src/mappers/*.cpp"
)

# Create the extension module
python_add_library(lib_nes_env MODULE ${SOURCES} WITH_SOABI)

# Set properties
target_include_directories(lib_nes_env PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/nes_py/nes/include"
)

target_compile_options(lib_nes_env PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O3 -pipe>
    $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

# Install the extension module
install(TARGETS lib_nes_env DESTINATION nes_py)