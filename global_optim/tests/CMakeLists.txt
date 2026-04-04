include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
# Prevent GTest from overriding our compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ── Helper to add a test executable ──────────────────────────────────────────
function(add_gtest target source)
  add_executable(${target} ${source})
  target_link_libraries(${target} PRIVATE global_optim GTest::gtest_main)
  add_test(NAME ${target} COMMAND ${target})
endfunction()

# ── Phase 1 tests ─────────────────────────────────────────────────────────────
add_gtest(test_random        test_random.cpp)
add_gtest(test_benchmark_l1  test_benchmark_l1.cpp)
