cmake_minimum_required(VERSION 3.13)
project(shl C CXX)

file (STRINGS "version" SHL_VERSION)
set(SHL_MAJOR_VERSION 2)

enable_language(ASM)

# SHL debug module
add_definitions(-DSHL_DEBUG)
# SHL disable xtheadvdot extension
add_definitions(-DSHL_DISABLE_VDOT)
# SHL Trace
add_definitions(-DSHL_TRACE)

set(SHL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SHL_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/source)

file(GLOB_RECURSE GLOB_SHL_NN2_SRCS source/nn2/*.c source/utils/*.c)
file(GLOB_RECURSE GLOB_SHL_REF_SRCS source/reference/*.c)
file(GLOB_RECURSE GLOB_SHL_GREF_SRCS source/graph_ref/*.c)
file(GLOB_RECURSE GLOB_SHL_THEAD_RVV_SRCS source/thead_rvv/*.c)
list(APPEND GLOB_SHL_SRCS ${GLOB_SHL_NN2_SRCS} ${GLOB_SHL_REF_SRCS} ${GLOB_SHL_GREF_SRCS} ${GLOB_SHL_THEAD_RVV_SRCS})
list(APPEND SHL_INCLUDE_DIRS ${SHL_INCLUDE_DIR} ${SHL_INCLUDE_DIR}/csinn ${SHL_INCLUDE_DIR}/graph ${SHL_INCLUDE_DIR}/backend)

add_library(shl STATIC ${GLOB_SHL_SRCS})
set(SHL_LIBS shl)

include_directories(${SHL_INCLUDE_DIRS})
include_directories(${SHL_SRC_DIR})

target_include_directories(shl PUBLIC ${SHL_INCLUDE_DIRS})
target_include_directories(shl PRIVATE ${SHL_SRC_DIR})

set(SHL_RVV_BUILD_FLAGS -ffp-contract=off -march=rv64gcv0p7_zfh_xtheadc -mabi=lp64d -DSHL_BUILD_RVV -DSHL_BUILD_REF -DSHL_BUILD_GREF -fopenmp)
target_compile_options(shl PRIVATE ${SHL_RVV_BUILD_FLAGS})

set_target_properties(shl PROPERTIES FOLDER "shl")
set_property(TARGET shl PROPERTY POSITION_INDEPENDENT_CODE ON)
