#.rst:
# FindFFTW3
# ---------
#
# Find FFTW3 include dirs and libraries
#
# Usage:
#
#   set(FFTW3_USE_MULTITHREADED ON)
#   find_package(FFTW3 [COMPONENTS [single double long-double]])
#
# This module finds the headers and required component libraries of FFTW3.
# It sets the following variables:
#
#  FFTW3_FOUND        ... true if FFTW3 is found on the system
#  FFTW3_LIBRARIES    ... list of all libraries found
#  FFTW3_INCLUDE_DIRS ... include directory
#
#
# This module reads hints from the following variables
#
#   FFTW3_USE_STATIC_LIBS   ... if true, only static libraries are found
#   FFTW3_USE_MULTITHREADED ... if true, multi-threded libraries are also found
#   FFTW3_ROOT              ... if set, the libraries are exclusively searched
#                               under this path
#   FFTW3_LIBRARY           ... FFTW3 library to use
#   FFTW3_INCLUDE_DIR       ... FFTW3 include directory
#

# If environment variable FFTW3_DIR or FFTW3_ROOT is specified, it has same
# effect as FFTW_ROOT
if (NOT $ENV{FFTW_ROOT} STREQUAL "")
  set (_FFTW3_ROOT ${FFTW3_ROOT})
elseif (NOT $ENV{FFTW3_DIR} STREQUAL "")
  set (_FFTW3_ROOT $ENV{FFTW3_DIR} )
endif()

# Check if we can use PkgConfig
find_package(PkgConfig)
# Determine from PKG
if (PKG_CONFIG_FOUND AND NOT _FFTW3_ROOT)
  pkg_check_modules(PKG_FFTW3 QUIET "fftw3")
endif()

#  static or dynamically linked library
if (${FFTW3_USE_STATIC_LIBS})
  set(CMAKE_1FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX})
else()
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_SHARED_LIBRARY_SUFFIX})
endif()

# Trying to find muti-threaded library by default
if (NOT DEFINED FFTW3_USE_MULTITHREADED)
  set(FFTW3_USE_MULTITHREADED OFF)
endif()

# List of library components to be find
if(FFTW3_FIND_COMPONENTS MATCHES "^$")
  # Use double precision by default.
  set(_components "double")
else()
  set(_components ${FFTW3_FIND_COMPONENTS})
endif()

# Loop over each component.
set(_libraries)
foreach(_comp ${_components})
  if(_comp STREQUAL "single")
    list(APPEND _libraries fftw3f)
  elseif(_comp STREQUAL "double")
    list(APPEND _libraries fftw3)
  elseif(_comp STREQUAL "long-double")
    list(APPEND _libraries fftw3l)
  else()
    message(FATAL_ERROR "FindFFTW3: unknown component `${_comp}' specified. "
      "Valid components are `single', `double', and `long-double'.")
  endif()
endforeach()

# If using FFTW3_USE_MULTITHREADED is ON, we need to link against threaded
# libraries as well.
set(_threaded_libraries)
if(FFTW3_USE_MULTITHREADED)
  foreach(_lib ${_libraries})
    list(APPEND _thread_libs ${_lib}_threads)
  endforeach()
  set(_libraries ${_thread_libs} ${_libraries})
endif()

# Keep a list of variable names that we need to pass on to
# find_package_handle_standard_args().
set(_check_list)

# Search for all requested libraries.
foreach(_lib ${_libraries})
  string(TOUPPER ${_lib} _LIB)
  # HINTS are checked before PATHS, that's why we call
  # find_library twice, to give priority to LD_LIBRARY_PATH or
  # user-defined paths over pkg-config process.
  # This first call should find _library in env. variables.
  find_library(${_LIB}_LIBRARY
    ${_lib}
    PATHS ${_FFTW3_ROOT}
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
    )
  find_library(${_LIB}_LIBRARY
    ${_lib}
    PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
    PATH_SUFFIXES "lib" "lib64"
    )

  mark_as_advanced(${_LIB}_LIBRARY)
  list(APPEND FFTW3_LIBRARIES ${${_LIB}_LIBRARY})
  list(APPEND _check_list ${_LIB}_LIBRARY)
endforeach(_lib ${_libraries})

find_path(FFTW3_INCLUDE_DIR
  NAMES  "fftw3.h"
  HINTS  ${_FFTW3_ROOT}
  PATHS ${PKG_FFTW3_LIBRARY_DIRS} ${LIB_INSTALL_DIR}
  PATH_SUFFIXES "include"
  )

mark_as_advanced(FFTW3_INCLUDE_DIR)
list(APPEND _check_list FFTW3_INCLUDE_DIR)

# Handle the QUIETLY and REQUIRED arguments and set FFTW_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFTW3 DEFAULT_MSG ${_check_list})

if(FFTW3_FOUND)
  set(FFTW3_INCLUDE_DIRS ${FFTW3_INCLUDE_DIR})
endif()
