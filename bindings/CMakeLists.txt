find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(zcgvs_engine python/module.cpp)

target_link_libraries(zcgvs_engine PRIVATE zcgvs_core)

set_target_properties(zcgvs_engine PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH ON)

install(TARGETS zcgvs_engine LIBRARY DESTINATION .)
