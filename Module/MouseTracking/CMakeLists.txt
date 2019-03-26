# cmake_minimum_required(VERSION <specify CMake version here>)
project(mousetracking)
cmake_minimum_required(VERSION 3.10)

add_subdirectory(pybind11)

pybind11_add_module(mousetracking src/mousetracking.cpp)
target_link_libraries(mousetracking PRIVATE pybind11::module)
set_target_properties(mousetracking PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")