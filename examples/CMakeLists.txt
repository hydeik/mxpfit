add_executable(example_fast_esprit fast_esprit.cpp)
target_link_libraries(example_fast_esprit ${FFTW3_LIBRARIES})

add_executable(example_balanced_truncation balanced_truncation.cpp)
add_executable(example_aak_reduction aak_reduction.cpp)

add_executable(bessel_j_kernel bessel_j_kernel.cpp)
add_executable(bessel_j_kernel2 bessel_j_kernel2.cpp)
target_link_libraries(bessel_j_kernel2 ${FFTW3_LIBRARIES})

add_executable(sph_bessel_kernel sph_bessel_kernel.cpp)

add_executable(pow_kernel pow_kernel.cpp)


add_executable(esprit_gaussian esprit_gaussian.cpp)
target_link_libraries(esprit_gaussian ${FFTW3_LIBRARIES})
