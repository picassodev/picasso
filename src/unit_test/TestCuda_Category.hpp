#ifndef HARLOW_TEST_CUDA_CATEGORY_HPP
#define HARLOW_TEST_CUDA_CATEGORY_HPP

#define TEST_CATEGORY cuda
#define TEST_EXECSPACE Kokkos::Cuda
#define TEST_MEMSPACE Kokkos::CudaSpace
#define TEST_DEVICE Kokkos::Device<Kokkos::OpenMP,Kokkos::HostSpace>

#endif // end HARLOW_TEST_CUDA_CATEGORY_HPP
