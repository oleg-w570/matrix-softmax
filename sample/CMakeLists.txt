file(GLOB HDRS "*.hpp")
file(GLOB SRCS "*.cpp")

add_executable(softmax_benchmark ${SRCS} ${HDRS})
target_link_libraries(softmax_benchmark PRIVATE softmax_implementations)
target_link_libraries(softmax_benchmark PRIVATE OpenMP::OpenMP_CXX)
target_include_directories(softmax_benchmark PRIVATE ${INCLUDE})