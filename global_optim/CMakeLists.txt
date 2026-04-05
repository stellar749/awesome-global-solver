cmake_minimum_required(VERSION 3.16)
project(global_optim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Eigen ────────────────────────────────────────────────────────────────────
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# ── Library target (header-only for now) ─────────────────────────────────────
add_library(global_optim INTERFACE)
target_include_directories(global_optim INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(global_optim INTERFACE Eigen3::Eigen)
target_compile_options(global_optim INTERFACE -Wall -Wextra -O2)

# ── Tests ─────────────────────────────────────────────────────────────────────
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

option(BUILD_PLAYGROUND "Build playground executables" ON)
if(BUILD_PLAYGROUND)
  add_subdirectory(playground)
endif()
