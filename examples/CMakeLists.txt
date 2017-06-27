add_executable(example_fast_esprit fast_esprit.cpp)
target_link_libraries(example_fast_esprit Eigen3::Eigen FFTW3::fftw)

add_executable(example_balanced_truncation balanced_truncation.cpp)
target_link_libraries(example_balanced_truncation Eigen3::Eigen)

#add_executable(bessel_j_kernel bessel_j_kernel.cpp)
#add_executable(sph_bessel_kernel sph_bessel_kernel.cpp)
