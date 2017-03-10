add_executable(example_fast_esprit fast_esprit.cpp)
target_link_libraries(example_fast_esprit ${FFTW3_LIBRARIES})

add_executable(sph_bessel_kernel sph_bessel_kernel.cpp)
