cmake_minimum_required(VERSION 3.2)
project(msc)

set(CMAKE_AUTOMOC ON)
set(SOURCE_FILES main.c)

find_package(PythonLibs REQUIRED)

include_directories(${CMAKE_BINARY_DIR})
include_directories(${PYTHON_INCLUDE_DIRS})

add_executable(main ${SOURCE_FILES})

target_link_libraries(main ${PYTHON_LIBRARIES})
