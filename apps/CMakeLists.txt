add_executable(zcgvs_search_cli search_cli.cpp)
target_link_libraries(zcgvs_search_cli PRIVATE zcgvs_core)

# Runs on synthetic data, so it needs no models and no corpus.
add_executable(zcgvs_bench_kernels bench_kernels.cu)
target_link_libraries(zcgvs_bench_kernels PRIVATE zcgvs_core)
