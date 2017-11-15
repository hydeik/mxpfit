# Examples for the fast ESPRIT method
add_executable(example_fast_esprit fast_esprit.cpp)
target_link_libraries(example_fast_esprit ${FFTW3_LIBRARIES})

add_executable(esprit_gaussian esprit_gaussian.cpp)
target_link_libraries(esprit_gaussian ${FFTW3_LIBRARIES})

# Examples for reduction algorithms
add_executable(example_balanced_truncation balanced_truncation.cpp)
add_executable(example_aak_reduction aak_reduction.cpp)

# Exponential sum approximation of analytical functions
add_executable(approx_power approx_power.cpp)
add_executable(approx_sph_bessel approx_sph_bessel.cpp)

# add_executable(bessel_j_kernel bessel_j_kernel.cpp)

