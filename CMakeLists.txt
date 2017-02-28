# -*- mode: cmake; indent-tabs-mode-nil; -*-
project(mxpfit C CXX Fortran)

cmake_minimum_required (VERSION 2.8.8)

option(mxpfit_build_test "Build tests for mxpfit-cpp" ON)
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation (requires Doxygen)" ON)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. (you may need to remove CMakeCache.txt ")
endif()

#===============================================================================
# Additional CMake module path
#===============================================================================
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#===============================================================================
# Project info
#===============================================================================

# Project include directories, exectable output path, etc.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)

#===============================================================================
# Compiler information
#===============================================================================
include(CXXCompilerSetup)

#===============================================================================
# Python Interpreter
#===============================================================================
# find_package(PythonInterp)

# if (NOT PYTHONINTERP_FOUND)
#   message(FATAL_ERROR "Cannot find python interpreter.")
# endif()

#===============================================================================
# REQUIRED external packages
#===============================================================================

# # External package directory
# set(EXTERNAL_PROJECT_DIR ${PROJECT_SOURCE_DIR}/external)
# #add_subdirectory(external)
# include_directories(${EXTERNAL_PROJECT_DIR}/include)
# link_directories(${EXTERNAL_PROJECT_DIR}/lib)

# include( ExternalProject )
#
# # {fmt} --- a C++ formatting library
# ExternalProject_Add(
#   fmt
#   PREFIX ${EXTERNAL_PROJECT_DIR}
#   GIT_REPOSITORY https://github.com/fmtlib/fmt.git
#   GIT_TAG 3.0.0
#   CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECT_DIR} -DFMT_DOC=OFF
#   TEST_AFTER_INSTALL 1
#   )
#
# # spdlog --- very fast C++ logging library
# ExternalProject_Add(
#   spdlog
#   PREFIX ${EXTERNAL_PROJECT_DIR}
#   GIT_REPOSITORY https://github.com/gabime/spdlog.git
#   GIT_TAG "master"
#   CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECT_DIR}
#   TEST_COMMAND ""
#   )
#
# # cereal --- a C++11 library for serialization
# ExternalProject_Add(
#   cereal
#   PREFIX ${EXTERNAL_PROJECT_DIR}
#   GIT_REPOSITORY https://github.com/USCiLab/cereal.git
#   GIT_TAG "master"
#   CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECT_DIR} -DJUST_INSTALL_CEREAL=ON
#   TEST_COMMAND ""
#   )

# # cpptoml --- A header-only library for parsing TOML configuration files.
# ExternalProject_Add(
#   cpptoml
#   PREFIX ${EXTERNAL_PROJECT_DIR}
#   GIT_REPOSITORY https://github.com/skystrife/cpptoml.git
#   GIT_TAG "master"
#   CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECT_DIR} 
#   TEST_COMMAND ""
#   )

# # docopt.cpp --- docopt creates beautiful command-line interfaces
# ExternalProject_Add(
#   docopt.cpp
#   PREFIX ${EXTERNAL_PROJECT_DIR}
#   GIT_REPOSITORY https://github.com/docopt/docopt.cpp.git
#   GIT_TAG "master"
#   CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECT_DIR} -DWITH_TESTS=ON
#   )

# # cmdline -- a simple command line parser for C++
# ExternalProject_Add(
#   cmdline
#   PREFIX ${EXTERNAL_PROJECT_DIR}
#   GIT_REPOSITORY https://github.com/tanakh/cmdline.git
#   GIT_TAG "master"
#   CONFIGURE_COMMAND ""
#   BUILD_COMMAND ""
#   INSTALL_COMMAND ${CMAKE_COMMAND} -E copy <SOURCE_DIR>/cmdline.h
#                   ${EXTERNAL_PROJECT_DIR}/include/cmdline/cmdline.h
#   TEST_COMMAND ""
#   TEST_COMMAND ""
#   )

# Eigen
find_package(Eigen3)
if (EIGEN3_FOUND)
  include_directories(${EIGEN3_INCLUDE_DIR})
  message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")
else()
  message(FATAL_ERROR "Cannot find Eigen3 library. Project will not be built.")
endif()

# FFTW3
set(FFTW3_USE_MULTITHREADED OFF)
find_package(FFTW3 COMPONENTS double)
if (FFTW3_FOUND)
  include_directories(${FFTW3_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Cannot find FFTW3 libraries. Project will not be built.")
endif()

#===============================================================================
# Build sources
#===============================================================================
add_library(mxpfit INTERFACE)
target_include_directories(mxpfit INTERFACE include/)

#===============================================================================
# Building API documentation with Doxygen
#===============================================================================
find_package(Doxygen)

if(BUILD_DOCUMENTATION)
  if(NOT DOXYGEN_FOUND)
     message(FATAL_ERROR "Doxygen is needed to build the documentation.")
  endif()

  set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

  configure_file(${doxyfile_in} ${doxyfile} @ONLY)

  add_custom_target(
    doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API documentation with Doxygen" VERBATIM
    )
endif()

#=============================================================================
# Testing
#=============================================================================
include(CTest)
if (mxpfit_build_test)
  enable_testing() # must be called from the root CMakeLists
  add_subdirectory(test)
endif()

#=============================================================================
# Install
#=============================================================================
# header files
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include DESTINATION include)
# documents
if(BUILD_DOCUMENTATION)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
endif()
