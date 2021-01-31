cmake_minimum_required(VERSION 3.14)

project(
    vulkan_study
    DESCRIPTION "Diving into Vulkan programming."
    HOMEPAGE_URL "https://github.com/unsip/vulkan_study"
    VERSION 0.0.1
    LANGUAGES CXX C
  )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

include(FindVulkan)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_subdirectory(src)
