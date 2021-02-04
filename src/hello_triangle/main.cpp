#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>
#include <glm/glm.hpp>

#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <set>
#include <optional>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <cstring>

// @todo: Enable all validation layers and debug info.

namespace {
constexpr std::size_t WIDTH = 1920;
constexpr std::size_t HEIGHT = 1080;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif


struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;

    static VkVertexInputBindingDescription get_binding_description()
    {
        VkVertexInputBindingDescription descr{};
        descr.binding = 0;
        descr.stride = sizeof(Vertex);
        descr.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return descr;
    }

    static std::array<VkVertexInputAttributeDescription, 2> get_attr_descriptions()
    {
        std::array<VkVertexInputAttributeDescription, 2> attr_descr{};
        attr_descr[0].binding = 0;
        attr_descr[0].location = 0;
        attr_descr[0].format = VK_FORMAT_R32G32_SFLOAT;
        attr_descr[0].offset = offsetof(Vertex, pos);
        attr_descr[1].binding = 0;
        attr_descr[1].location = 1;
        attr_descr[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attr_descr[1].offset = offsetof(Vertex, color);

        return attr_descr;
    }
};


const std::vector<Vertex> VERTICES = {
    {{0.0f, -0.5f}, {1.0f, 0.0f, 1.0f}}
  , {{0.5f, 0.5f},  {0.0f, 1.0f, 0.0f}}
  , {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

// @todo: Verbose error message.
static std::vector<char> load_shader(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open())
        throw std::runtime_error("Failed to open file!");

    std::size_t sz = static_cast<std::size_t>(file.tellg());
    std::vector<char> buf(sz);
    file.seekg(0);
    file.read(buf.data(), sz);

    return buf;
}

struct QueueFamilyIndices
{
    std::optional<std::uint32_t> graphicsFamily;
    std::optional<std::uint32_t> presentFamily;
    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};


struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};


// TODO: Add debug info support (see Message Callbacks in Vulkan Tutorial).
class TriangleApp
{
private:
    GLFWwindow*                     m_window = nullptr;
    std::size_t                     m_width;
    std::size_t                     m_height;
    VkInstance                      m_instance;
    VkPhysicalDevice                m_phy_dev = VK_NULL_HANDLE;     // Implicitly destroyed with VkInstance.
    VkDevice                        m_dev;
    VkQueue                         m_graphics_queue;
    VkQueue                         m_present_queue;
    VkSurfaceKHR                    m_surface;
    VkSwapchainKHR                  m_swapchain;
    std::vector<VkImage>            m_swapchain_imgs;
    VkFormat                        m_swapchain_img_format;
    VkExtent2D                      m_swapchain_extent;
    std::vector<VkImageView>        m_swapchain_img_views;
    VkRenderPass                    m_render_pass;
    VkPipelineLayout                m_pipeline_layout;
    VkPipeline                      m_pipeline;
    std::vector<VkFramebuffer>      m_swapchain_framebuffers;
    VkCommandPool                   m_cmd_pool;
    std::vector<VkCommandBuffer>    m_cmd_buffers;
    std::vector<VkSemaphore>        m_img_available_semaphores;
    std::vector<VkSemaphore>        m_render_finished_semaphores;
    std::vector<VkFence>            m_in_flight_fences;
    std::vector<VkFence>            m_imgs_in_flight;
    std::size_t                     m_curr_frame = 0;
    bool                            m_framebuffer_resized = false;
    VkBuffer                        m_vert_buffer;
    VkDeviceMemory                  m_vert_buffer_memory;

public:
    TriangleApp(std::size_t width, std::size_t height)
        : m_width(width), m_height(height)
    {}

    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    static void framebufferResizeCallback(GLFWwindow* window, int, int)
    {
        auto app = reinterpret_cast<TriangleApp*>(glfwGetWindowUserPointer(window));
        app->m_framebuffer_resized = true;
    }

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);   // Suppress OpenGL context.
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);     // Disable window resizing.
        m_window = glfwCreateWindow(m_width, m_height, "Vulkan Triangle", nullptr, nullptr);
        glfwSetWindowUserPointer(m_window, this);
        glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);
    }

    void initVulkan()
    {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createVertexBuffer();
        createCommandBuffers();
        createSyncPrimitives();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(m_window))
        {
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(m_dev);
    }

    void drawFrame()
    {
        vkWaitForFences(m_dev, 1, &m_in_flight_fences[m_curr_frame], VK_TRUE, UINT64_MAX);
        std::uint32_t img_idx;
        VkResult res = vkAcquireNextImageKHR(
            m_dev
          , m_swapchain
          , UINT64_MAX
          , m_img_available_semaphores[m_curr_frame]
          , VK_NULL_HANDLE
          , &img_idx
        );

        if (res == VK_ERROR_OUT_OF_DATE_KHR)
        {
            recreateSwapChain();
            return;
        }
        else if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Failed to acquire swapchain image!");

        // Check if a previous frame is using this image (i.e. there is its fence to wait on).
        if (m_imgs_in_flight[img_idx] != VK_NULL_HANDLE)
            vkWaitForFences(m_dev, 1, &m_imgs_in_flight[img_idx], VK_TRUE, UINT64_MAX);
        // Mark the image as now being used by a current frame.
        m_imgs_in_flight[img_idx] = m_in_flight_fences[m_curr_frame];

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore wait_sems[] = {m_img_available_semaphores[m_curr_frame]};
        VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submit_info.waitSemaphoreCount = 1;
        submit_info.pWaitSemaphores = wait_sems;
        submit_info.pWaitDstStageMask = wait_stages;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &m_cmd_buffers[img_idx];
        VkSemaphore signal_sems[] = {m_render_finished_semaphores[m_curr_frame]};
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = signal_sems;

        vkResetFences(m_dev, 1, &m_in_flight_fences[m_curr_frame]);
        if (vkQueueSubmit(m_graphics_queue, 1, &submit_info, m_in_flight_fences[m_curr_frame]) != VK_SUCCESS)
            throw std::runtime_error("Failed to submit draw command buffer!");

        VkPresentInfoKHR present_info{};
        present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        present_info.waitSemaphoreCount = 1;
        present_info.pWaitSemaphores = signal_sems;

        VkSwapchainKHR swapchains[] = {m_swapchain};
        present_info.swapchainCount = 1;
        present_info.pSwapchains = swapchains;
        present_info.pImageIndices = &img_idx;
        res = vkQueuePresentKHR(m_present_queue, &present_info);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || m_framebuffer_resized)
        {
            m_framebuffer_resized = false;
            recreateSwapChain();
        }
        else if (res != VK_SUCCESS)
            throw std::runtime_error("Failed to present swapchain image!");

        m_curr_frame = (m_curr_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void cleanup()
    {
        cleanupSwapchain();
        vkDestroyBuffer(m_dev, m_vert_buffer, nullptr);
        vkFreeMemory(m_dev, m_vert_buffer_memory, nullptr);
        for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
        {
            vkDestroySemaphore(m_dev, m_render_finished_semaphores[i], nullptr);
            vkDestroySemaphore(m_dev, m_img_available_semaphores[i], nullptr);
            vkDestroyFence(m_dev, m_in_flight_fences[i], nullptr);
        }
        vkDestroyCommandPool(m_dev, m_cmd_pool, nullptr);
        vkDestroyDevice(m_dev, nullptr);
        vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
        vkDestroyInstance(m_instance, nullptr);
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    void cleanupSwapchain()
    {
        for (auto framebuffer : m_swapchain_framebuffers)
            vkDestroyFramebuffer(m_dev, framebuffer, nullptr);
        vkFreeCommandBuffers(m_dev, m_cmd_pool, static_cast<std::uint32_t>(m_cmd_buffers.size()), m_cmd_buffers.data());
        vkDestroyPipeline(m_dev, m_pipeline, nullptr);
        vkDestroyPipelineLayout(m_dev, m_pipeline_layout, nullptr);
        vkDestroyRenderPass(m_dev, m_render_pass, nullptr);
        for (auto img_view : m_swapchain_img_views)
            vkDestroyImageView(m_dev, img_view, nullptr);
        vkDestroySwapchainKHR(m_dev, m_swapchain, nullptr);
    }

    void createCommandBuffers()
    {
        m_cmd_buffers.resize(m_swapchain_framebuffers.size());
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = m_cmd_pool;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = static_cast<std::uint32_t>(m_cmd_buffers.size());

        if (vkAllocateCommandBuffers(m_dev, &alloc_info, m_cmd_buffers.data()) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate command buffers!");

        for (std::size_t i = 0; i < m_cmd_buffers.size(); i++) {
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

            if (vkBeginCommandBuffer(m_cmd_buffers[i], &begin_info) != VK_SUCCESS)
                throw std::runtime_error("failed to begin recording command buffer!");

            VkRenderPassBeginInfo render_pass_info{};
            render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            render_pass_info.renderPass = m_render_pass;
            render_pass_info.framebuffer = m_swapchain_framebuffers[i];
            render_pass_info.renderArea.offset = {0, 0};
            render_pass_info.renderArea.extent = m_swapchain_extent;

            VkClearValue clear_color = {0.0f, 0.0f, 0.0f, 1.0f};
            render_pass_info.clearValueCount = 1;
            render_pass_info.pClearValues = &clear_color;

            vkCmdBeginRenderPass(m_cmd_buffers[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(m_cmd_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

            VkBuffer vert_buffers[] = {m_vert_buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(m_cmd_buffers[i], 0, 1, vert_buffers, offsets);

            vkCmdDraw(m_cmd_buffers[i], static_cast<std::uint32_t>(VERTICES.size()), 1, 0, 0);
            vkCmdEndRenderPass(m_cmd_buffers[i]);
            if (vkEndCommandBuffer(m_cmd_buffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to record command buffer!");
        }
    }

    void createCommandPool()
    {
        QueueFamilyIndices qf_indices = findQueueFamilies(m_phy_dev);

        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = qf_indices.graphicsFamily.value();
        if (vkCreateCommandPool(m_dev, &pool_info, nullptr, &m_cmd_pool) != VK_SUCCESS)
            throw std::runtime_error("Failed to create command pool!");
    }

    void createFramebuffers()
    {
        m_swapchain_framebuffers.resize(m_swapchain_img_views.size());
        for (std::size_t i = 0; i < m_swapchain_img_views.size(); i++) {
            VkImageView attachments[] = { m_swapchain_img_views[i] };

            VkFramebufferCreateInfo framebuffer_info{};
            framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebuffer_info.renderPass = m_render_pass;
            framebuffer_info.attachmentCount = 1;
            framebuffer_info.pAttachments = attachments;
            framebuffer_info.width = m_swapchain_extent.width;
            framebuffer_info.height = m_swapchain_extent.height;
            framebuffer_info.layers = 1;

            if (vkCreateFramebuffer(m_dev, &framebuffer_info, nullptr, &m_swapchain_framebuffers[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create framebuffer!");
        }
    }

    // @todo: RAII for shader modules;
    void createGraphicsPipeline()
    {
        auto vert = load_shader("vert.spv");
        auto frag = load_shader("frag.spv");
        VkShaderModule vert_mod = createShaderModule(vert);
        VkShaderModule frag_mod = createShaderModule(frag);

        VkPipelineShaderStageCreateInfo vert_mod_info{};
        vert_mod_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vert_mod_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vert_mod_info.module = vert_mod;
        vert_mod_info.pName = "main";

        VkPipelineShaderStageCreateInfo frag_mod_info{};
        frag_mod_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        frag_mod_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        frag_mod_info.module = frag_mod;
        frag_mod_info.pName = "main";

        VkPipelineShaderStageCreateInfo stages[] = {vert_mod_info, frag_mod_info};

        auto binding_descr = Vertex::get_binding_description();
        auto attr_descr = Vertex::get_attr_descriptions();

        VkPipelineVertexInputStateCreateInfo vert_input_info{};
        vert_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vert_input_info.vertexBindingDescriptionCount = 1;
        vert_input_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(attr_descr.size());
        vert_input_info.pVertexBindingDescriptions = &binding_descr;
        vert_input_info.pVertexAttributeDescriptions = attr_descr.data();

        VkPipelineInputAssemblyStateCreateInfo input_assembly{};
        input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        input_assembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(m_swapchain_extent.width);
        viewport.height = static_cast<float>(m_swapchain_extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = m_swapchain_extent;

        VkPipelineViewportStateCreateInfo viewport_state{};
        viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewport_state.viewportCount = 1;
        viewport_state.pViewports = &viewport;
        viewport_state.scissorCount = 1;
        viewport_state.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState color_blend_attachment{};
        color_blend_attachment.colorWriteMask =
              VK_COLOR_COMPONENT_R_BIT
            | VK_COLOR_COMPONENT_G_BIT
            | VK_COLOR_COMPONENT_B_BIT
            | VK_COLOR_COMPONENT_A_BIT;
        color_blend_attachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo color_blending{};
        color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = VK_FALSE;
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &color_blend_attachment;

        VkPipelineLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        if (vkCreatePipelineLayout(m_dev, &layout_info, nullptr, &m_pipeline_layout) != VK_SUCCESS)
            throw std::runtime_error("failed to create pipeline layout!");

        VkGraphicsPipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipeline_info.stageCount = 2;
        pipeline_info.pStages = stages;
        pipeline_info.pVertexInputState = &vert_input_info;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        pipeline_info.pColorBlendState = &color_blending;
        pipeline_info.layout = m_pipeline_layout;
        pipeline_info.renderPass = m_render_pass;
        pipeline_info.subpass = 0;

        if (vkCreateGraphicsPipelines(m_dev, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &m_pipeline) != VK_SUCCESS)
            throw std::runtime_error("Failed to create graphics pipeline!");

        vkDestroyShaderModule(m_dev, frag_mod, nullptr);
        vkDestroyShaderModule(m_dev, vert_mod, nullptr);
    }

    void createImageViews()
    {
        m_swapchain_img_views.resize(m_swapchain_imgs.size());
        for (std::size_t i = 0; i < m_swapchain_imgs.size(); ++i)
        {
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = m_swapchain_imgs[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = m_swapchain_img_format;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;
            if (vkCreateImageView(m_dev, &createInfo, nullptr, &m_swapchain_img_views[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create image views!");
            }
        }
    }

    void createLogicalDevice()
    {
        QueueFamilyIndices indices = findQueueFamilies(m_phy_dev);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<std::uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        float queuePriority = 1.0f;

        for (std::uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkPhysicalDeviceFeatures deviceFeatures {};
        VkDeviceCreateInfo createInfo {};

        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<std::uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();
        createInfo.queueCreateInfoCount = 1;
        createInfo.pEnabledFeatures = &deviceFeatures;
        createInfo.enabledExtensionCount = static_cast<std::uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<std::uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
            createInfo.enabledLayerCount = 0;

        if (vkCreateDevice(m_phy_dev, &createInfo, nullptr, &m_dev) != VK_SUCCESS)
            throw std::runtime_error("Failed to create logical device!");

        vkGetDeviceQueue(m_dev, indices.graphicsFamily.value(), 0, &m_graphics_queue);
        vkGetDeviceQueue(m_dev, indices.presentFamily.value(), 0, &m_present_queue);
    }

    void createRenderPass()
    {
        VkAttachmentDescription color_attachment{};
        color_attachment.format = m_swapchain_img_format;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference color_attachment_ref{};
        color_attachment_ref.attachment = 0;
        color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment_ref;

        VkRenderPassCreateInfo render_pass_info{};
        render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        render_pass_info.attachmentCount = 1;
        render_pass_info.pAttachments = &color_attachment;
        render_pass_info.subpassCount = 1;
        render_pass_info.pSubpasses = &subpass;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        render_pass_info.dependencyCount = 1;
        render_pass_info.pDependencies = &dependency;

        if (vkCreateRenderPass(m_dev, &render_pass_info, nullptr, &m_render_pass) != VK_SUCCESS)
            throw std::runtime_error("failed to create render pass!");
    }

    void createSyncPrimitives()
    {
        m_img_available_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_render_finished_semaphores.resize(MAX_FRAMES_IN_FLIGHT);
        m_in_flight_fences.resize(MAX_FRAMES_IN_FLIGHT);
        m_imgs_in_flight.resize(m_swapchain_imgs.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo sem_info{};
        sem_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i)
            if (vkCreateSemaphore(m_dev, &sem_info, nullptr, &m_img_available_semaphores[i]) != VK_SUCCESS
             || vkCreateSemaphore(m_dev, &sem_info, nullptr, &m_render_finished_semaphores[i]) != VK_SUCCESS
             || vkCreateFence(m_dev, &fence_info, nullptr, &m_in_flight_fences[i]) != VK_SUCCESS)
                throw std::runtime_error("Failed to create sync primitives for frame!");
    }

    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        // Default std::vector allocator satisfies alignment requirements.
        createInfo.pCode = reinterpret_cast<const std::uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(m_dev, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
            throw std::runtime_error("failed to create shader module!");

        return shaderModule;
    }

    void createSurface()
    {
        if (glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface) != VK_SUCCESS)
            throw std::runtime_error("Failed to create window surface!");
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(m_phy_dev);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        std::uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
            imageCount = swapChainSupport.capabilities.maxImageCount;

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = m_surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(m_phy_dev);
        std::uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;

        if (vkCreateSwapchainKHR(m_dev, &createInfo, nullptr, &m_swapchain) != VK_SUCCESS)
            throw std::runtime_error("failed to create swap chain!");

        vkGetSwapchainImagesKHR(m_dev, m_swapchain, &imageCount, nullptr);
        m_swapchain_imgs.resize(imageCount);
        vkGetSwapchainImagesKHR(m_dev, m_swapchain, &imageCount, m_swapchain_imgs.data());

        m_swapchain_extent = extent;
        m_swapchain_img_format = surfaceFormat.format;
    }

    void createVertexBuffer()
    {
        VkBufferCreateInfo buff_info{};
        buff_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buff_info.size = sizeof(VERTICES[0]) * VERTICES.size();
        buff_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        buff_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(m_dev, &buff_info, nullptr, &m_vert_buffer) != VK_SUCCESS)
            throw std::runtime_error("Failed to create vertex buffer!");

        VkMemoryRequirements mem_requirements;
        vkGetBufferMemoryRequirements(m_dev, m_vert_buffer, &mem_requirements);
        VkMemoryAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc_info.allocationSize = mem_requirements.size;
        alloc_info.memoryTypeIndex = findMemoryType(
            mem_requirements.memoryTypeBits
          , VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        );

        if (vkAllocateMemory(m_dev, &alloc_info, nullptr, &m_vert_buffer_memory) != VK_SUCCESS)
            throw std::runtime_error("Failed to allocate vertex buffer memory!");

        vkBindBufferMemory(m_dev, m_vert_buffer, m_vert_buffer_memory, 0);
        void* data;
        vkMapMemory(m_dev, m_vert_buffer_memory, 0, buff_info.size, 0, &data);
        memcpy(data, VERTICES.data(), static_cast<std::uint32_t>(buff_info.size));
        vkUnmapMemory(m_dev, m_vert_buffer_memory);
    }

    std::uint32_t findMemoryType(std::uint32_t type_filter, VkMemoryPropertyFlags props)
    {
        VkPhysicalDeviceMemoryProperties mem_props;
        vkGetPhysicalDeviceMemoryProperties(m_phy_dev, &mem_props);
        for (std::uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
            if ((type_filter & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & props) == props)
                return i;

        throw std::runtime_error("Failed to find a suitable memory type!");
    }

    void pickPhysicalDevice()
    {
        std::uint32_t devCnt = 0;
        vkEnumeratePhysicalDevices(m_instance, &devCnt, nullptr);

        if (devCnt == 0)
            throw std::runtime_error("Failed to find any GPUs with Vulkan support!");

        std::vector<VkPhysicalDevice> devices(devCnt);
        vkEnumeratePhysicalDevices(m_instance, &devCnt, devices.data());

        for (const auto& dev : devices)
        {
            if (isDeviceSuitable(dev))
            {
                m_phy_dev = dev;
                break;
            }
        }

        if (m_phy_dev == VK_NULL_HANDLE)
            throw std::runtime_error("Failed to find suitable GPU!");
    }

    // For now any device will do.
    bool isDeviceSuitable(VkPhysicalDevice dev)
    {
        bool extensionsSupported = checkDeviceExtensionsSupport(dev);
        bool swapChainAdequate = false;
        if (extensionsSupported)
        {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(dev);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
        return findQueueFamilies(dev).isComplete() && extensionsSupported && swapChainAdequate;
    }

    bool checkDeviceExtensionsSupport(VkPhysicalDevice dev)
    {
        std::uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions)
            requiredExtensions.erase(extension.extensionName);

        return requiredExtensions.empty();
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != UINT32_MAX)
            return capabilities.currentExtent;

        int width, height;
        glfwGetFramebufferSize(m_window, &width, &height);

        VkExtent2D actualExtent = { static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height) };
        actualExtent.width = std::max(
            capabilities.minImageExtent.width
          , std::min(capabilities.maxImageExtent.width , actualExtent.width)
        );
        actualExtent.height = std::max(
            capabilities.minImageExtent.height
          , std::min(capabilities.maxImageExtent.height, actualExtent.height)
        );

        return actualExtent;
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
    {

        for (const auto& availablePresentMode : availablePresentModes)
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
                return availablePresentMode;

        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
    {
        for (const auto& availableFormat : availableFormats)
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return availableFormat;

        return availableFormats[0];
    }

    // TODO: Remove reference to global vars.
    bool checkValidationLayerSupport()
    {
        std::uint32_t layerCnt;
        vkEnumerateInstanceLayerProperties(&layerCnt, nullptr);
        std::vector<VkLayerProperties> availableLayers(layerCnt);
        vkEnumerateInstanceLayerProperties(&layerCnt, availableLayers.data());

        for (const char* layerName: validationLayers)
        {
            bool layerFound = false;
            for (const auto& layerProperties : availableLayers)
            {
                if (strcmp(layerName, layerProperties.layerName) == 0)
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound)
                return false;
        }

        return true;
    }

    // TODO: Remove reference to global vars.
    void createInstance()
    {
        if (enableValidationLayers && !checkValidationLayerSupport())
            throw std::runtime_error("Validation layers requested, but not available!");

        VkApplicationInfo appInfo;
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo;
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        std::uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        if (enableValidationLayers)
        {
            createInfo.enabledLayerCount = static_cast<std::uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else
            createInfo.enabledLayerCount = 0;

        std::uint32_t extCnt = 0;
        std::vector<VkExtensionProperties> extensions(extCnt);
        vkEnumerateInstanceExtensionProperties(nullptr, &extCnt, extensions.data());

        std::cout << "Available Vulkan extensions:\n";
        for (const auto& ext : extensions)
            std::cout << "\t" << ext.extensionName << "\n";

        if (vkCreateInstance(&createInfo, nullptr, &m_instance) != VK_SUCCESS)
            throw std::runtime_error("Failed to create Vulkan instance!");
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev)
    {
        QueueFamilyIndices indices;

        std::uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());

        int i = 0;
        for (const auto& queueFamily : queueFamilies)
        {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT)
                indices.graphicsFamily = i;

            // Lookup for queue family which is able to present to window surface.
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, m_surface, &presentSupport);
            if (presentSupport)
                indices.presentFamily = i;

            // Why do we even need this? (See QueueFamilies section.)
            if (indices.isComplete())
                break;

            i++;
        }

        return indices;
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice dev)
    {
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev, m_surface, &details.capabilities);
        std::uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(dev, m_surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(dev, m_surface, &formatCount, details.formats.data());
        }

        std::uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(dev, m_surface, &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(dev, m_surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    void recreateSwapChain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(m_window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(m_window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(m_dev);
        cleanupSwapchain();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandBuffers();
    }
};
} // anonymous namespace


int main()
{
    TriangleApp app(WIDTH, HEIGHT);
    try
    {
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
