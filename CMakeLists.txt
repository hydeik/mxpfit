# -*- mode: cmake; indent-tabs-mode-nil; -*-
project(mxpfit CXX)

cmake_minimum_required (VERSION 3.1)

option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TEST "Build tests" OFF)
option(BUILD_DOCUMENTATION "Build API documents (requires Doxygen)" OFF)

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

#===============================================================================
# Compiler information
#===============================================================================
include(CXXCompilerSetup)

#===============================================================================
# Git
#===============================================================================
find_package(Git REQUIRED)

#===============================================================================
# REQUIRED external packages
#===============================================================================

# External package directory
set(EXTERNAL_PROJECT_DIR ${PROJECT_SOURCE_DIR}/external)
#add_subdirectory(external)
include_directories(${EXTERNAL_PROJECT_DIR}/include)
#link_directories(${EXTERNAL_PROJECT_DIR}/lib)
include( ExternalProject )

# Catch --  C++ Automated Test Cases in Headers
ExternalProject_Add(
  catch
  PREFIX ${EXTERNAL_PROJECT_DIR}
  GIT_REPOSITORY https://github.com/philsquared/Catch.git
  GIT_TAG "master"
  TIMEOUT 10
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${EXTERNAL_PROJECT_DIR}
  )

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

# Boost
find_package(Boost 1.50.0)
if(Boost_FOUND)
  include_directories(${Boost_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Cannot find boost libraries. Project will not be built.")
endif()


#===============================================================================
# Build sources
#===============================================================================
add_library(mxpfit INTERFACE)
target_include_directories(mxpfit INTERFACE include/)

#===============================================================================
# Building API documentation with Doxygen
#===============================================================================

if(BUILD_DOCUMENTATION)
  find_package(Doxygen)
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
# Build example programs
#=============================================================================
if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

#=============================================================================
# Build example programs
#=============================================================================
if (BUILD_TEST)
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
