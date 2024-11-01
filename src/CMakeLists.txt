file(GLOB HDRS "${INCLUDE}/*.hpp")
file(GLOB SRCS "*.cpp")

add_library(softmax_implementations STATIC ${SRCS} ${HDRS})
target_include_directories(softmax_implementations PRIVATE ${INCLUDE})
target_link_libraries(softmax_implementations PRIVATE OpenMP::OpenMP_CXX)
