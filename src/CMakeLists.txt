add_library(zcgvs_core STATIC
    core/cuda_context.cpp
    data/embedding_store.cpp
    data/corpus.cpp
    data/lexical_index.cpp
    cuda/similarity.cu
    cuda/topk.cu
    cuda/convert.cu
    cuda/rescore.cu
    cuda/lexical.cu
    cuda/cublas_similarity.cpp
    search/searcher.cpp
    text/tokenizer.cpp
    onnx/onnx_session.cpp
    embed/onnx_query_embedder.cpp
    rerank/onnx_reranker.cpp
    rerank/reranker_factory.cpp
    engine/engine.cpp)

target_include_directories(zcgvs_core PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(zcgvs_core PUBLIC
    CUDA::cudart
    CUDA::cublas
    ONNXRuntime::ONNXRuntime)

if(ZCGVS_ENABLE_NVTX)
    target_compile_definitions(zcgvs_core PUBLIC ZCGVS_WITH_NVTX)
    if(TARGET CUDA::nvtx3)
        target_link_libraries(zcgvs_core PUBLIC CUDA::nvtx3)
    endif()
endif()

set_target_properties(zcgvs_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF)
