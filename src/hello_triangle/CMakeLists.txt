add_executable(
    triangle
    main.cpp
    )

target_link_libraries(
    triangle
    PRIVATE
        Vulkan::Vulkan
        glfw
    )
