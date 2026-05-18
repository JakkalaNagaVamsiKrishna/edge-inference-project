cmake_minimum_required(VERSION 3.20)
project(edge_cv_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Optimization flags ────────────────────────────────────────────────────────
# -O3              : Maximum scalar optimization
# -march=native    : Use all CPU features of the BUILD host
# -ffast-math      : Relax IEEE754 strictness for speed (safe for inference)
# For ARM cross-compilation the Dockerfile overrides -march to armv8-a+simd
if(NOT CMAKE_CROSSCOMPILING)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -march=native")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Config passed from CI / Dockerfile ───────────────────────────────────────
# These become compile-time constants in main.cpp
set(MODEL_PATH  "/app/model/student_int8.onnx" CACHE STRING "Path to ONNX model")
set(SERVER_PORT "8080"                          CACHE STRING "HTTP server port")
set(INPUT_H     "224"                           CACHE STRING "Input image height")
set(INPUT_W     "224"                           CACHE STRING "Input image width")
set(INPUT_C     "3"                             CACHE STRING "Input channels")

add_definitions(
  -DMODEL_PATH="${MODEL_PATH}"
  -DSERVER_PORT=${SERVER_PORT}
  -DINPUT_H=${INPUT_H}
  -DINPUT_W=${INPUT_W}
  -DINPUT_C=${INPUT_C}
)

# ── ONNX Runtime ──────────────────────────────────────────────────────────────
# Expects ONNXRUNTIME_ROOT to point to the extracted ONNX Runtime release.
# The Dockerfile downloads the correct ARM64 or x86_64 build automatically.
if(NOT DEFINED ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "/opt/onnxruntime")
endif()

find_path(ORT_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
  REQUIRED
)
find_library(ORT_LIB
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}/lib
  REQUIRED
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(edge_cv_server
  src/main.cpp
)

target_include_directories(edge_cv_server PRIVATE
  ${ORT_INCLUDE_DIR}
  ${CMAKE_SOURCE_DIR}/include       # httplib.h and json.hpp live here
)

target_link_libraries(edge_cv_server PRIVATE
  ${ORT_LIB}
  pthread
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS edge_cv_server DESTINATION /app/bin)