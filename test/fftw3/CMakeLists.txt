# -*- mode: cmake; indent-tabs-mode-nil; -*-

add_executable(test_fftw3_shared_plan shared_plan.cpp)
target_link_libraries(test_fftw3_shared_plan Eigen3::Eigen mxpfit::FFTW3)
