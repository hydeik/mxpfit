# Examples for the fast ESPRIT method
add_executable(example_esprit_compare esprit_compare.cpp)
target_link_libraries(example_esprit_compare Eigen3::Eigen mxpfit::FFTW3)

add_executable(example_fast_esprit fast_esprit.cpp)
target_link_libraries(example_fast_esprit Eigen3::Eigen mxpfit::FFTW3)

add_executable(esprit_gaussian esprit_gaussian.cpp)
target_link_libraries(esprit_gaussian Eigen3::Eigen mxpfit::FFTW3)

# Examples for reduction algorithms
add_executable(example_balanced_truncation balanced_truncation.cpp)
target_link_libraries(example_balanced_truncation Eigen3::Eigen)

# add_executable(example_aak_reduction aak_reduction.cpp)
# target_link_libraries(example_aak_reduction Eigen3::Eigen)

# Exponential sum approximation of analytical functions
add_executable(approx_power approx_power.cpp)
target_link_libraries(approx_power Eigen3::Eigen)

add_executable(approx_sph_bessel approx_sph_bessel.cpp)
target_link_libraries(approx_sph_bessel Eigen3::Eigen)
