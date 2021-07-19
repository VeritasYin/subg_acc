cmake_minimum_required(VERSION 3.19)
project(gather_op C)

set(CMAKE_C_STANDARD 99)

add_executable(gather_op gcom_acc.c utils.h)
add_executable(uhash_test59 uhash_test59.c)
add_executable(uhash_example uhash_example.c)