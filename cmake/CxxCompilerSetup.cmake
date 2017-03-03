# -*- mode: cmake; indent-tabs-mode-nil; -*-

#===============================================================================
# Compiler information
#===============================================================================
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX14)

if(COMPILER_SUPPORTS_CXX14)
  set(CXX_COMPILER_FLAG_STD "-std=c++14")
else()
  message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(CXX_WARNING_FLAGS "-Wall -Wextra-tokens -Wformat -Wformat-security-Wstrict-aliasing -Wcast-qual -Wcast-align -Wno-long-long -Wpointer-arith -Wuninitialized -Wwrite-strings")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(CXX_WARNING_FLAGS "-stdlib=libc++ -Wall -Wextra -Wno-c++98-compat")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CXX_WARNING_FLAGS "-stdlib=libc++ -Wall -Wextra -Wno-c++98-compat")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CXX_WARNING_FLAGS "-Wall -Wextra -Wformat=2 -Wstrict-aliasing=2 -Wcast-qual -Wcast-align -Wno-long-long -Wwrite-strings -Wconversion -Wfloat-equal -Wpointer-arith -Wswitch-enum")
endif()

#
# Build type
#
string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)

if(cmake_build_type_tolower STREQUAL "debug")
  set(CMAKE_BUILD_TYPE "DEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_COMPILER_FLAG_STD} ${CXX_WARNING_FLAGS}")
else()
  set(CMAKE_BUILD_TYPE "RELEASE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_COMPILER_FLAG_STD} ${CXX_WARNING_FLAGS}")
endif()

list(REMOVE_DUPLICATES CMAKE_CXX_FLAGS)

message(STATUS "Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler ID   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Compiler flags: ${CMAKE_CXX_FLAGS}")

