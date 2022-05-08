#include <string>
#include <fstream>
#include <iostream>
#include <functional>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define TINYOBJLOADER_IMPLEMENTATION
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <tiny_obj_loader.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

constexpr int WIDTH = 1024;
constexpr int HEIGHT = 1024;

VKAPI_ATTR VkBool32 VKAPI_CALL
debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                            VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData, void* pUserData)
{
    std::cerr << pCallbackData->pMessage << "\n\n";
    return VK_FALSE;
}

void setImageLayout(vk::CommandBuffer cmdBuf, vk::Image image,
                    vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
    vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eAllCommands;
    vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eAllCommands;

    vk::ImageMemoryBarrier barrier{};
    barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
    barrier.setImage(image);
    barrier.setOldLayout(oldLayout);
    barrier.setNewLayout(newLayout);
    barrier.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

    using vkAF = vk::AccessFlagBits;
    switch (oldLayout) {
    case vk::ImageLayout::eTransferSrcOptimal:
        barrier.srcAccessMask = vkAF::eTransferRead;
        break;
    case vk::ImageLayout::eTransferDstOptimal:
        barrier.srcAccessMask = vkAF::eTransferWrite;
        break;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        barrier.srcAccessMask = vkAF::eShaderRead;
        break;
    default:
        break;
    }

    switch (newLayout) {
    case vk::ImageLayout::eTransferDstOptimal:
        barrier.dstAccessMask = vkAF::eTransferWrite;
        break;
    case vk::ImageLayout::eTransferSrcOptimal:
        barrier.dstAccessMask = vkAF::eTransferRead;
        break;
    case vk::ImageLayout::eShaderReadOnlyOptimal:
        if (barrier.srcAccessMask == vk::AccessFlags{}) {
            barrier.srcAccessMask = vkAF::eHostWrite | vkAF::eTransferWrite;
        }
        barrier.dstAccessMask = vkAF::eShaderRead;
        break;
    default:
        break;
    }
    cmdBuf.pipelineBarrier(srcStageMask, dstStageMask, {}, {}, {}, barrier);
}

struct Vertex
{
    float pos[3];
};

struct Face
{
    float diffuse[3];
    float emission[3];
};

void loadFromFile(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
                  std::vector<Face>& faces)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          "../assets/CornellBox-Original.obj", "../assets")) {
        throw std::runtime_error(warn + err);
    }

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};
            vertex.pos[0] = attrib.vertices[3 * index.vertex_index + 0];
            vertex.pos[1] = -attrib.vertices[3 * index.vertex_index + 1];
            vertex.pos[2] = attrib.vertices[3 * index.vertex_index + 2];
            vertices.push_back(vertex);
            indices.push_back(static_cast<uint32_t>(indices.size()));
        }
        for (const auto& matIndex : shape.mesh.material_ids) {
            Face face;
            face.diffuse[0] = materials[matIndex].diffuse[0];
            face.diffuse[1] = materials[matIndex].diffuse[1];
            face.diffuse[2] = materials[matIndex].diffuse[2];
            face.emission[0] = materials[matIndex].emission[0];
            face.emission[1] = materials[matIndex].emission[1];
            face.emission[2] = materials[matIndex].emission[2];
            faces.push_back(face);
        }
    }
}

std::vector<char> readFile(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

struct Context
{
    static void init()
    {
        // Create window
        {
            glfwInit();
            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
            window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Pathtracing", nullptr, nullptr);
        }

        // Create instance
        {
            // Gather extensions
            uint32_t glfwExtensionCount = 0;
            const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
            std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

            // Gather layers
            std::vector<const char*> layers{ "VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor" };

            // Setup DynamicLoader (see https://github.com/KhronosGroup/Vulkan-Hpp)
            static vk::DynamicLoader dl;
            auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
            VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

            vk::ApplicationInfo appInfo;
            appInfo.setPApplicationName("VulkanPathtracing");
            appInfo.setApiVersion(VK_API_VERSION_1_2);
            instance = vk::createInstanceUnique({ {}, &appInfo, layers, extensions });
            VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
            physicalDevice = instance->enumeratePhysicalDevices().front();
        }

        // Create debug messenger
        {
            vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
            messengerInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
            messengerInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
            messengerInfo.setPfnUserCallback(&debugUtilsMessengerCallback);
            messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo);
        }

        // Create surface
        {
            VkSurfaceKHR _surface;
            VkResult res = glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface);
            if (res != VK_SUCCESS) {
                throw std::runtime_error("failed to create window surface!");
            }
            surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), { *instance });
        }

        // Find queue family
        {
            auto queueFamilies = physicalDevice.getQueueFamilyProperties();
            for (int i = 0; i < queueFamilies.size(); i++) {
                auto supportCompute = queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute;
                auto supportPresent = physicalDevice.getSurfaceSupportKHR(i, *surface);
                if (supportCompute && supportPresent) {
                    queueFamily = i;
                }
            }
        }

        // Create device
        {
            float queuePriority = 1.0f;
            vk::DeviceQueueCreateInfo queueCreateInfo{ {}, queueFamily, 1, &queuePriority };

            const std::vector<const char*> deviceExtensions{
                VK_KHR_SWAPCHAIN_EXTENSION_NAME,
                VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
                VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
                VK_KHR_MAINTENANCE3_EXTENSION_NAME,
                VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
                VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
                VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
                VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
            };

            vk::PhysicalDeviceFeatures deviceFeatures;
            vk::DeviceCreateInfo createInfo{ {}, queueCreateInfo, {}, deviceExtensions, &deviceFeatures };
            vk::StructureChain<vk::DeviceCreateInfo,
                vk::PhysicalDeviceBufferDeviceAddressFeatures,
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR>
                createInfoChain{ createInfo, {true}, {true}, {true} };

            device = physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());
            VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
            queue = device->getQueue(queueFamily, 0);
        }

        // Create swapchain
        {
            vk::SwapchainCreateInfoKHR swapchainInfo{};
            swapchainInfo.setSurface(*surface);
            swapchainInfo.setMinImageCount(3);
            swapchainInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
            swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
            swapchainInfo.setImageExtent({ WIDTH, HEIGHT });
            swapchainInfo.setImageArrayLayers(1);
            swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst);
            swapchainInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
            swapchainInfo.setPresentMode(vk::PresentModeKHR::eFifo);
            swapchainInfo.setClipped(true);
            swapchain = device->createSwapchainKHRUnique(swapchainInfo);
            swapchainImages = device->getSwapchainImagesKHR(*swapchain);
        }

        // Create command pool
        {
            vk::CommandPoolCreateInfo poolInfo;
            poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
            poolInfo.setQueueFamilyIndex(queueFamily);
            commandPool = device->createCommandPoolUnique(poolInfo);
        }

        // Create descriptor pool
        {
            std::vector<vk::DescriptorPoolSize> poolSizes{
                {vk::DescriptorType::eAccelerationStructureKHR, 1},
                {vk::DescriptorType::eStorageImage, 1},
                {vk::DescriptorType::eStorageBuffer, 3} };

            vk::DescriptorPoolCreateInfo poolInfo;
            poolInfo.setPoolSizes(poolSizes);
            poolInfo.setMaxSets(1);
            poolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
            descPool = device->createDescriptorPoolUnique(poolInfo);
        }
    }

    static void shutdown()
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    static bool windowShouldClose()
    {
        return glfwWindowShouldClose(window);
    }

    static void pollEvents()
    {
        glfwPollEvents();
    }

    static uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type");
    }

    static std::vector<vk::UniqueCommandBuffer> allocateCommandBuffers(size_t count)
    {
        vk::CommandBufferAllocateInfo allocInfo;
        allocInfo.setCommandPool(*commandPool);
        allocInfo.setCommandBufferCount(static_cast<uint32_t>(count));
        return device->allocateCommandBuffersUnique(allocInfo);
    }

    static void oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func)
    {
        vk::UniqueCommandBuffer cmdBuf = std::move(allocateCommandBuffers(1).front());
        cmdBuf->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        func(*cmdBuf);
        cmdBuf->end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(*cmdBuf);
        queue.submit(submitInfo);
        queue.waitIdle();
    }

    static vk::UniqueDescriptorSet allocateDescSet(vk::DescriptorSetLayout descSetLayout)
    {
        return std::move(device->allocateDescriptorSetsUnique({ *descPool, descSetLayout }).front());
    }

    static uint32_t acquireNextImage(vk::Semaphore semaphore)
    {
        auto res = Context::device->acquireNextImageKHR(*swapchain, UINT64_MAX, semaphore);
        if (res.result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to acquire next image!");
        }
        return res.value;
    }

    static inline GLFWwindow* window;
    static inline vk::UniqueInstance instance;
    static inline vk::UniqueDebugUtilsMessengerEXT messenger;
    static inline vk::UniqueSurfaceKHR surface;
    static inline vk::UniqueDevice device;
    static inline vk::PhysicalDevice physicalDevice;
    static inline uint32_t queueFamily;
    static inline vk::Queue queue;
    static inline vk::UniqueCommandPool commandPool;
    static inline vk::UniqueSwapchainKHR swapchain;
    static inline std::vector<vk::Image> swapchainImages;
    static inline vk::UniqueDescriptorPool descPool;
};

struct Buffer
{
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorBufferInfo bufferInfo;
    uint64_t deviceAddress = 0;

    void create(vk::BufferUsageFlags usage, vk::DeviceSize size)
    {
        create(usage, vk::MemoryPropertyFlagBits::eDeviceLocal, size);
    }

    void create(vk::BufferUsageFlags usage, vk::DeviceSize size, void* data)
    {
        create(usage, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, size);

        // Copy
        void* mapped = Context::device->mapMemory(*memory, 0, size);
        memcpy(mapped, data, static_cast<size_t>(size));
        Context::device->unmapMemory(*memory);
    }

    template <typename T>
    void create(vk::BufferUsageFlags usage, std::vector<T> data)
    {
        create(usage, sizeof(T) * data.size(), data.data());
    }

private:
    void create(vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProps, vk::DeviceSize size)
    {
        buffer = Context::device->createBufferUnique({ {}, size, usage });

        // Allocate memory
        vk::MemoryRequirements requirements = Context::device->getBufferMemoryRequirements(*buffer);
        uint32_t memoryTypeIndex = Context::findMemoryType(requirements.memoryTypeBits, memoryProps);

        vk::MemoryAllocateFlagsInfo flagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };
        vk::MemoryAllocateInfo allocInfo{ requirements.size, memoryTypeIndex };
        allocInfo.pNext = &flagsInfo;

        memory = Context::device->allocateMemoryUnique(allocInfo);
        Context::device->bindBufferMemory(*buffer, *memory, 0);

        // Get device address
        vk::BufferDeviceAddressInfoKHR bufferDeviceAI{ *buffer };
        deviceAddress = Context::device->getBufferAddressKHR(&bufferDeviceAI);

        bufferInfo = vk::DescriptorBufferInfo{ *buffer, 0, size };
    }
};

struct Image
{
    vk::UniqueImage image;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorImageInfo imageInfo;

    void create(vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage)
    {
        // Create image
        vk::ImageCreateInfo createInfo;
        createInfo.setImageType(vk::ImageType::e2D);
        createInfo.setExtent({ extent.width, extent.height, 1 });
        createInfo.setMipLevels(1);
        createInfo.setArrayLayers(1);
        createInfo.setFormat(format);
        createInfo.setUsage(usage);
        image = Context::device->createImageUnique(createInfo);

        // Allocate memory
        vk::MemoryRequirements requirements = Context::device->getImageMemoryRequirements(*image);
        uint32_t memoryTypeIndex = Context::findMemoryType(requirements.memoryTypeBits,
                                                           vk::MemoryPropertyFlagBits::eDeviceLocal);
        memory = Context::device->allocateMemoryUnique({ requirements.size, memoryTypeIndex });
        Context::device->bindImageMemory(*image, *memory, 0);

        // Create image view
        vk::ImageViewCreateInfo viewInfo;
        viewInfo.setImage(*image);
        viewInfo.setViewType(vk::ImageViewType::e2D);
        viewInfo.setFormat(format);
        viewInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        view = Context::device->createImageViewUnique(viewInfo);

        // Set image layout
        imageInfo = { {}, *view, vk::ImageLayout::eGeneral };
        Context::oneTimeSubmit(
            [&](vk::CommandBuffer cmdBuf) {
                setImageLayout(cmdBuf, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
            });
    }
};

struct Accel
{
    vk::UniqueAccelerationStructureKHR accel;
    Buffer buffer;
    vk::WriteDescriptorSetAccelerationStructureKHR accelInfo;
    uint64_t deviceAddress = 0;
    vk::DeviceSize size = 0;

    void createAsBottom(vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount)
    {
        create(geometry, primitiveCount, vk::AccelerationStructureTypeKHR::eBottomLevel);
    }

    void createAsTop(vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount)
    {
        create(geometry, primitiveCount, vk::AccelerationStructureTypeKHR::eTopLevel);
    }

private:
    void create(vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount,
                vk::AccelerationStructureTypeKHR type)
    {
        vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo;
        buildGeometryInfo.setType(type);
        buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        buildGeometryInfo.setGeometries(geometry);

        // Create buffer
        vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = Context::device->getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
        size = buildSizesInfo.accelerationStructureSize;
        buffer.create(vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);

        // Create accel
        vk::AccelerationStructureCreateInfoKHR createInfo;
        createInfo.setBuffer(*buffer.buffer);
        createInfo.setSize(size);
        createInfo.setType(type);
        accel = Context::device->createAccelerationStructureKHRUnique(createInfo);

        // Build
        Buffer scratchBuffer;
        scratchBuffer.create(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, size);
        buildGeometryInfo.setScratchData(scratchBuffer.deviceAddress);
        buildGeometryInfo.setDstAccelerationStructure(*accel);

        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{ primitiveCount , 0, 0, 0 };
        Context::oneTimeSubmit(
            [&](vk::CommandBuffer commandBuffer) {
                commandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
            });

        accelInfo = vk::WriteDescriptorSetAccelerationStructureKHR{ *accel };
    }
};

struct PushConstants
{
    int frame = 0;
};

class Application
{
public:
    ~Application()
    {
        for (size_t i = 0; i < maxFramesInFlight; i++) {
            Context::device->destroyFence(inFlightFences[i]);
        }
    }

    void run()
    {
        initVulkan();
        mainLoop();
    }

private:
    Image outputImage;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer faceBuffer;
    PushConstants pushConstants;

    Accel bottomAccel;
    Accel topAccel;

    std::vector<vk::UniqueShaderModule> shaderModules;
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;

    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueDescriptorSetLayout descSetLayout;
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    vk::UniqueDescriptorSet descSet;

    Buffer raygenSBT;
    Buffer missSBT;
    Buffer hitSBT;

    vk::StridedDeviceAddressRegionKHR raygenRegion;
    vk::StridedDeviceAddressRegionKHR missRegion;
    vk::StridedDeviceAddressRegionKHR hitRegion;

    std::vector<vk::UniqueCommandBuffer> drawCommandBuffers;
    std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    size_t currentFrame = 0;
    int maxFramesInFlight = 2;

    void initVulkan()
    {
        outputImage.create({ WIDTH, HEIGHT }, vk::Format::eB8G8R8A8Unorm, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
        loadFromFile(vertices, indices, faces);
        createMeshBuffers();
        createBottomLevelAS();
        createTopLevelAS();
        loadShaders();
        createRayTracingPipeLine();
        createShaderBindingTable();
        createDescriptorSets();
        drawCommandBuffers = Context::allocateCommandBuffers(Context::swapchainImages.size());
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!Context::windowShouldClose()) {
            Context::pollEvents();
            drawFrame();
        }
        Context::device->waitIdle();
    }

    void createMeshBuffers()
    {
        vk::BufferUsageFlags usage{ vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress };
        vertexBuffer.create(usage, vertices);
        indexBuffer.create(usage, indices);
        faceBuffer.create(vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress, faces);
    }

    void createBottomLevelAS()
    {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangleData;
        triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
        triangleData.setVertexData(vertexBuffer.deviceAddress);
        triangleData.setVertexStride(sizeof(Vertex));
        triangleData.setMaxVertex(static_cast<uint32_t>(vertices.size()));
        triangleData.setIndexType(vk::IndexType::eUint32);
        triangleData.setIndexData(indexBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry;
        geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
        geometry.setGeometry({ triangleData });
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        uint32_t primitiveCount = static_cast<uint32_t>(indices.size() / 3);
        bottomAccel.createAsBottom(geometry, primitiveCount);
    }

    void createTopLevelAS()
    {
        vk::TransformMatrixKHR transformMatrix = std::array{
            std::array{1.0f, 0.0f, 0.0f, 0.0f },
            std::array{0.0f, 1.0f, 0.0f, 0.0f },
            std::array{0.0f, 0.0f, 1.0f, 0.0f } };

        vk::AccelerationStructureInstanceKHR asInstance;
        asInstance.setTransform(transformMatrix);
        asInstance.setMask(0xFF);
        asInstance.setAccelerationStructureReference(bottomAccel.buffer.deviceAddress);
        asInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

        Buffer instancesBuffer;
        instancesBuffer.create(vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR |
                               vk::BufferUsageFlagBits::eShaderDeviceAddress,
                               sizeof(vk::AccelerationStructureInstanceKHR), &asInstance);

        vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
        instancesData.setArrayOfPointers(false);
        instancesData.setData(instancesBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry;
        geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
        geometry.setGeometry({ instancesData });
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        uint32_t primitiveCount = 1;
        topAccel.createAsTop(geometry, primitiveCount);
    }

    void loadShaders()
    {
        const uint32_t raygenIndex = 0;
        const uint32_t missIndex = 1;
        const uint32_t chitIndex = 2;

        shaderModules.push_back(createShaderModule("../shaders/raygen.rgen.spv"));
        shaderStages.push_back({ {}, vk::ShaderStageFlagBits::eRaygenKHR, *shaderModules.back(), "main" });
        shaderGroups.push_back({ vk::RayTracingShaderGroupTypeKHR::eGeneral,
                               raygenIndex, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR });

        shaderModules.push_back(createShaderModule("../shaders/miss.rmiss.spv"));
        shaderStages.push_back({ {}, vk::ShaderStageFlagBits::eMissKHR, *shaderModules.back(), "main" });
        shaderGroups.push_back({ vk::RayTracingShaderGroupTypeKHR::eGeneral,
                               missIndex, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR });

        shaderModules.push_back(createShaderModule("../shaders/closesthit.rchit.spv"));
        shaderStages.push_back({ {}, vk::ShaderStageFlagBits::eClosestHitKHR, *shaderModules.back(), "main" });
        shaderGroups.push_back({ vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                               VK_SHADER_UNUSED_KHR, chitIndex, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR });
    }

    vk::UniqueShaderModule createShaderModule(const std::string& filename)
    {
        const std::vector<char> code = readFile(filename);
        return Context::device->createShaderModuleUnique({ {}, code.size(), reinterpret_cast<const uint32_t*>(code.data()) });
    }

    void createRayTracingPipeLine()
    {
        using vkDT = vk::DescriptorType;
        using vkSS = vk::ShaderStageFlagBits;
        bindings.push_back({ 0, vkDT::eAccelerationStructureKHR, 1, vkSS::eRaygenKHR }); // Binding = 0 : TLAS
        bindings.push_back({ 1, vkDT::eStorageImage, 1, vkSS::eRaygenKHR });             // Binding = 1 : Storage image
        bindings.push_back({ 2, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 2 : Vertices
        bindings.push_back({ 3, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 3 : Indices
        bindings.push_back({ 4, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 4 : Faces

        descSetLayout = Context::device->createDescriptorSetLayoutUnique({ {}, bindings });

        vk::PushConstantRange pushRange;
        pushRange.setOffset(0);
        pushRange.setSize(sizeof(PushConstants));
        pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);
        pipelineLayout = Context::device->createPipelineLayoutUnique({ {}, *descSetLayout, pushRange });

        // Create pipeline
        vk::RayTracingPipelineCreateInfoKHR createInfo;
        createInfo.setStages(shaderStages);
        createInfo.setGroups(shaderGroups);
        createInfo.setMaxPipelineRayRecursionDepth(4);
        createInfo.setLayout(*pipelineLayout);
        auto res = Context::device->createRayTracingPipelineKHRUnique(nullptr, nullptr, createInfo);
        if (res.result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to create ray tracing pipeline.");
        }
        pipeline = std::move(res.value);
    }

    void createShaderBindingTable()
    {
        // Get Ray Tracing Properties
        using vkRTP = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR;
        vkRTP rtProperties = Context::physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vkRTP>().get<vkRTP>();

        // Calculate SBT size
        uint32_t handleSize = rtProperties.shaderGroupHandleSize;
        uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
        uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
        uint32_t sbtSize = static_cast<uint32_t>(groupCount * handleSizeAligned);

        // Get shader group handles
        std::vector<uint8_t> handleStorage(sbtSize);
        if (Context::device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess) {
            throw std::runtime_error("failed to get ray tracing shader group handles.");
        }

        vk::BufferUsageFlags usage =
            vk::BufferUsageFlagBits::eShaderBindingTableKHR |
            vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress;
        raygenSBT.create(usage, handleSize, handleStorage.data() + 0 * handleSizeAligned);
        missSBT.create(usage, handleSize, handleStorage.data() + 1 * handleSizeAligned);
        hitSBT.create(usage, handleSize, handleStorage.data() + 2 * handleSizeAligned);

        uint32_t stride = rtProperties.shaderGroupHandleAlignment;
        uint32_t size = rtProperties.shaderGroupHandleAlignment;
        raygenRegion = vk::StridedDeviceAddressRegionKHR{ raygenSBT.deviceAddress, stride, size };
        missRegion = vk::StridedDeviceAddressRegionKHR{ missSBT.deviceAddress, stride, size };
        hitRegion = vk::StridedDeviceAddressRegionKHR{ hitSBT.deviceAddress, stride, size };
    }

    void createDescriptorSets()
    {
        descSet = Context::allocateDescSet(*descSetLayout);

        std::vector<vk::WriteDescriptorSet> writes(bindings.size());
        for (int i = 0; i < bindings.size(); i++) {
            writes[i].setDstSet(*descSet);
            writes[i].setDescriptorType(bindings[i].descriptorType);
            writes[i].setDescriptorCount(bindings[i].descriptorCount);
            writes[i].setDstBinding(bindings[i].binding);
        }
        writes[0].setPNext(&topAccel.accelInfo);
        writes[1].setImageInfo(outputImage.imageInfo);
        writes[2].setBufferInfo(vertexBuffer.bufferInfo);
        writes[3].setBufferInfo(indexBuffer.bufferInfo);
        writes[4].setBufferInfo(faceBuffer.bufferInfo);
        Context::device->updateDescriptorSets(writes, nullptr);
    }

    void recordCommandBuffers(vk::CommandBuffer commandBuffer, vk::Image swapchainImage)
    {
        commandBuffer.begin(vk::CommandBufferBeginInfo{});
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0, *descSet, nullptr);
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, sizeof(PushConstants), &pushConstants);
        commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);

        using vkIL = vk::ImageLayout;
        setImageLayout(commandBuffer, *outputImage.image, vkIL::eUndefined, vkIL::eTransferSrcOptimal);
        setImageLayout(commandBuffer, swapchainImage, vkIL::eUndefined, vkIL::eTransferDstOptimal);

        copyImage(commandBuffer, *outputImage.image, swapchainImage);

        setImageLayout(commandBuffer, *outputImage.image, vkIL::eTransferSrcOptimal, vkIL::eGeneral);
        setImageLayout(commandBuffer, swapchainImage, vkIL::eTransferDstOptimal, vkIL::ePresentSrcKHR);
        commandBuffer.end();
    }

    void copyImage(vk::CommandBuffer& cmdBuf, vk::Image& srcImage, vk::Image& dstImage)
    {
        vk::ImageCopy copyRegion{};
        copyRegion.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setExtent({ WIDTH, HEIGHT, 1 });
        cmdBuf.copyImage(srcImage, vk::ImageLayout::eTransferSrcOptimal,
                         dstImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(maxFramesInFlight);
        renderFinishedSemaphores.resize(maxFramesInFlight);
        inFlightFences.resize(maxFramesInFlight);

        for (size_t i = 0; i < maxFramesInFlight; i++) {
            imageAvailableSemaphores[i] = Context::device->createSemaphoreUnique({});
            renderFinishedSemaphores[i] = Context::device->createSemaphoreUnique({});
            inFlightFences[i] = Context::device->createFence({ vk::FenceCreateFlagBits::eSignaled });
        }
    }

    void drawFrame()
    {
        if (Context::device->waitForFences(inFlightFences[currentFrame], true, UINT64_MAX) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to wait for fences");
        }
        Context::device->resetFences(inFlightFences[currentFrame]);

        uint32_t imageIndex = Context::acquireNextImage(*imageAvailableSemaphores[currentFrame]);
        recordCommandBuffers(*drawCommandBuffers[imageIndex], Context::swapchainImages[imageIndex]);

        // Submit draw command
        vk::PipelineStageFlags waitStage{ vk::PipelineStageFlagBits::eRayTracingShaderKHR };
        vk::SubmitInfo submitInfo;
        submitInfo.setWaitSemaphores(*imageAvailableSemaphores[currentFrame]);
        submitInfo.setWaitDstStageMask(waitStage);
        submitInfo.setCommandBuffers(*drawCommandBuffers[imageIndex]);
        submitInfo.setSignalSemaphores(*renderFinishedSemaphores[currentFrame]);
        Context::queue.submit(submitInfo, inFlightFences[currentFrame]);

        // Present image
        vk::PresentInfoKHR presentInfo;
        presentInfo.setWaitSemaphores(*renderFinishedSemaphores[currentFrame]);
        presentInfo.setSwapchains(*Context::swapchain);
        presentInfo.setImageIndices(imageIndex);
        if (Context::queue.presentKHR(presentInfo) != vk::Result::eSuccess) {
            throw std::runtime_error("Failed to present");
        }

        currentFrame = (currentFrame + 1) % maxFramesInFlight;
        pushConstants.frame++;
    }
};

int main()
{
    try {
        Context::init();
        {
            Application app;
            app.run();
        }
        Context::shutdown();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
