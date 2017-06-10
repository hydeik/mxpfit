# -*- mode: cmake; indent-tabs-mode-nil; -*-
project(mxpfit C CXX Fortran)

cmake_minimum_required (VERSION 3.2.0)

include(GNUInstallDirs)

option(BUILD_EXAMPLES "Build example programs" OFF)
option(BUILD_TEST "Build tests" OFF)
option(BUILD_DOCUMENTATION "Build API documents (requires Doxygen)" OFF)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "\
In-source builds not allowed. Please make a new directory \
(called a build directory) and run CMake from there. \
(you may need to remove CMakeCache.txt ")
endif()

# Additional CMake module path
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Compiler setup
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- External packages
# Eigen
find_package(Eigen3 REQUIRED)
if (EIGEN3_FOUND)
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# FFTW3
set(FFTW3_USE_MULTITHREADED OFF)
find_package(FFTW3 REQUIRED COMPONENTS double)
if (FFTW3_FOUND)
  include_directories(${FFTW3_INCLUDE_DIRS})
endif()

# # Boost
# find_package(Boost 1.50.0)
# if(Boost_FOUND)
#   include_directories(${Boost_INCLUDE_DIRS})
# else()
#   message(FATAL_ERROR "Cannot find boost libraries. Project will not be built.")
# endif()

# --- Library definitions
# directory structure
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

add_library(mxpfit INTERFACE)
target_include_directories(mxpfit INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:include>
  )

target_compile_features(mxpfit
  INTERFACE cxx_alias_templates cxx_auto_type cxx_constexpr cxx_decltype
  cxx_defaulted_functions cxx_defaulted_move_initializers cxx_lambdas
  )

install(TARGETS mxpfit EXPORT mxpfitConfig INCLUDES DESTINATION "include")
install(EXPORT mxpfitConfig DESTINATION "lib/cmake/mxpfit")
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION "share/doc")
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
  enable_testing()
  find_package(catch REQUIRED CONFIG)
  add_subdirectory(test)
endif()
