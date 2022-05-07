
#include <set>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <functional>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

using vkBU = vk::BufferUsageFlagBits;
using vkIU = vk::ImageUsageFlagBits;
using vkMP = vk::MemoryPropertyFlagBits;
using vkDT = vk::DescriptorType;
using vkIL = vk::ImageLayout;

// ----------------------------------------------------------------------------------------------------------
// Globals
// ----------------------------------------------------------------------------------------------------------
constexpr int WIDTH = 1024;
constexpr int HEIGHT = 1024;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
const std::string ASSET_PATH = "../assets/CornellBox.obj";

// ----------------------------------------------------------------------------------------------------------
// Functions
// ----------------------------------------------------------------------------------------------------------
VKAPI_ATTR VkBool32 VKAPI_CALL
debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                            VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData, void* /*pUserData*/)
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

std::vector<std::string> split(std::string& str, char separator)
{
    std::vector<std::string> list;
    size_t offset = 0;
    while (1) {
        auto pos = str.find(separator, offset);
        if (pos == std::string::npos) {
            list.push_back(str.substr(offset));
            break;
        }
        list.push_back(str.substr(offset, pos - offset));
        offset = pos + 1;
    }
    return list;
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
            std::vector<const char*> layers{ "VK_LAYER_KHRONOS_validation" };

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
            swapchainInfo.setImageUsage(vkIU::eTransferDst);
            swapchainInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
            swapchainInfo.setPresentMode(vk::PresentModeKHR::eFifo);
            swapchainInfo.setClipped(true);
            swapChain = device->createSwapchainKHRUnique(swapchainInfo);
            swapchainImages = device->getSwapchainImagesKHR(*swapChain);
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
                {vkDT::eAccelerationStructureKHR, 1},
                {vkDT::eStorageImage, 2},
                {vkDT::eStorageBuffer, 3},
                {vkDT::eUniformBuffer, 1} };

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

    static std::vector<vk::UniqueCommandBuffer> allocateCommandBuffers(uint32_t count)
    {
        vk::CommandBufferAllocateInfo allocInfo;
        allocInfo.setCommandPool(*commandPool);
        allocInfo.setCommandBufferCount(count);
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

    static uint32_t getImageCount()
    {
        return swapchainImages.size();
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
    static inline vk::UniqueSwapchainKHR swapChain;
    static inline std::vector<vk::Image> swapchainImages;
    static inline vk::UniqueDescriptorPool descPool;
};

// ----------------------------------------------------------------------------------------------------------
// Structs
// ----------------------------------------------------------------------------------------------------------
struct Buffer
{
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DeviceSize size;
    uint64_t deviceAddress;
    void* mapped = nullptr;
    vk::DescriptorBufferInfo bufferInfo;

    void create(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags memoryProps)
    {
        this->size = size;
        buffer = Context::device->createBufferUnique({ {}, size, usage });

        // Allocate memory
        vk::MemoryRequirements requirements = Context::device->getBufferMemoryRequirements(*buffer);
        uint32_t memoryTypeIndex = Context::findMemoryType(requirements.memoryTypeBits, memoryProps);
        vk::MemoryAllocateInfo allocInfo{ requirements.size, memoryTypeIndex };

        vk::MemoryAllocateFlagsInfo flagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };
        allocInfo.pNext = &flagsInfo;

        memory = Context::device->allocateMemoryUnique(allocInfo);
        Context::device->bindBufferMemory(*buffer, *memory, 0);

        vk::BufferDeviceAddressInfoKHR bufferDeviceAI{ *buffer };
        deviceAddress = Context::device->getBufferAddressKHR(&bufferDeviceAI);
    }

    void copy(void* data)
    {
        if (!mapped) {
            mapped = Context::device->mapMemory(*memory, 0, size);
        }
        memcpy(mapped, data, static_cast<size_t>(size));
    }

    vk::WriteDescriptorSet createWrite(vk::DescriptorSet& descSet, vk::DescriptorType type, uint32_t binding)
    {
        bufferInfo = vk::DescriptorBufferInfo{ *buffer, 0, size };
        vk::WriteDescriptorSet bufferWrite;
        bufferWrite.setDstSet(descSet);
        bufferWrite.setDescriptorType(type);
        bufferWrite.setDescriptorCount(1);
        bufferWrite.setDstBinding(binding);
        bufferWrite.setBufferInfo(bufferInfo);
        return bufferWrite;
    }
};

struct Image
{
    vk::UniqueImage image;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory memory;
    vk::Extent2D extent;
    vk::Format format;
    vk::ImageLayout imageLayout;
    vk::DescriptorImageInfo imageInfo;

    void create(vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage)
    {
        this->extent = extent;
        this->format = format;

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
    }

    vk::WriteDescriptorSet createWrite(vk::DescriptorSet& descSet, vk::DescriptorType type, uint32_t binding)
    {
        imageInfo = vk::DescriptorImageInfo{ {}, *view, imageLayout };
        vk::WriteDescriptorSet imageWrite;
        imageWrite.setDstSet(descSet);
        imageWrite.setDescriptorType(type);
        imageWrite.setDescriptorCount(1);
        imageWrite.setDstBinding(binding);
        imageWrite.setImageInfo(imageInfo);
        return imageWrite;
    }
};

enum class Material : int
{
    White, Red, Green, Light
};

struct Vertex
{
    float pos[3];
};

struct Accel
{
    vk::UniqueAccelerationStructureKHR handle;
    Buffer buffer;
    vk::DeviceSize size;
    uint64_t deviceAddress;

    void create(vk::AccelerationStructureGeometryKHR geometry,
                vk::AccelerationStructureTypeKHR type, uint32_t primitiveCount)
    {
        vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo;
        buildGeometryInfo.setType(type);
        buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        buildGeometryInfo.setGeometries(geometry);

        // Create buffer
        vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = Context::device->getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
        size = buildSizesInfo.accelerationStructureSize;
        buffer.create(size, vkBU::eAccelerationStructureStorageKHR | vkBU::eShaderDeviceAddress, vkMP::eDeviceLocal);

        // Create accel
        vk::AccelerationStructureCreateInfoKHR createInfo;
        createInfo.setBuffer(*buffer.buffer);
        createInfo.setSize(size);
        createInfo.setType(type);
        handle = Context::device->createAccelerationStructureKHRUnique(createInfo);

        // Build
        Buffer scratchBuffer;
        scratchBuffer.create(size, vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress, vkMP::eDeviceLocal);
        buildGeometryInfo.setScratchData(scratchBuffer.deviceAddress);
        buildGeometryInfo.setDstAccelerationStructure(*handle);

        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{ primitiveCount , 0, 0, 0 };
        Context::oneTimeSubmit(
            [&](vk::CommandBuffer commandBuffer) {
                commandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
            });
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
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            Context::device->destroyFence(inFlightFences[i]);
        }
    }

    void run()
    {
        initVulkan();
        mainLoop();
    }

private:
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties;

    std::vector<vk::UniqueCommandBuffer> drawCommandBuffers;

    Image inputImage;
    Image outputImage;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    Buffer vertexBuffer;
    Buffer indexBuffer;

    std::vector<Material> primitiveMaterials;
    Buffer primitiveBuffer;

    Accel bottomAccel;
    Accel topAccel;

    std::vector<vk::UniqueShaderModule> shaderModules;
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;

    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueDescriptorSetLayout descSetLayout;

    Buffer raygenSBT;
    Buffer missSBT;
    Buffer hitSBT;

    vk::UniqueDescriptorSet descSet;

    std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    size_t currentFrame = 0;

    PushConstants pushConstants;

    void initVulkan()
    {
        createImage(inputImage);
        createImage(outputImage);
        loadMesh();
        createMeshBuffers();
        createBottomLevelAS();
        createTopLevelAS();
        loadShaders();
        createRayTracingPipeLine();
        createShaderBindingTable();
        createDescriptorSets();
        drawCommandBuffers = Context::allocateCommandBuffers(Context::getImageCount());
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!Context::windowShouldClose()) {
            Context::pollEvents();
            drawFrame();
            if (pushConstants.frame % 10 == 0) {
                std::cout << "frame: " << pushConstants.frame << std::endl;
            }
        }
        Context::device->waitIdle();
    }

    void createImage(Image& image)
    {
        image.create({ WIDTH, HEIGHT }, vk::Format::eB8G8R8A8Unorm, vkIU::eStorage | vkIU::eTransferSrc | vkIU::eTransferDst);

        // Set image layout
        image.imageLayout = vk::ImageLayout::eGeneral;
        Context::oneTimeSubmit(
            [&](vk::CommandBuffer cmdBuf) {
                setImageLayout(cmdBuf, *image.image, vk::ImageLayout::eUndefined, image.imageLayout);
            });
    }

    void loadMesh()
    {
        std::ifstream file(ASSET_PATH);
        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        std::string line;
        Material currentMaterial = Material::White;
        while (std::getline(file, line)) {
            std::vector<std::string> list = split(line, ' ');
            if (list[0] == "v") {
                vertices.push_back(Vertex{ stof(list[1]), -stof(list[2]), stof(list[3]) });
            }
            if (list[0] == "usemtl") {
                if (list[1] == "White") currentMaterial = Material::White;
                if (list[1] == "Red")   currentMaterial = Material::Red;
                if (list[1] == "Green") currentMaterial = Material::Green;
                if (list[1] == "Light") currentMaterial = Material::Light;
            }
            if (list[0] == "f") {
                for (int i = 1; i <= 3; i++) {
                    std::vector<std::string> vertAttrs = split(list[i], '/');
                    int vertIndex = stoi(vertAttrs[0]) - 1;
                    indices.push_back(static_cast<uint32_t>(vertIndex));
                }
                primitiveMaterials.push_back(currentMaterial);
            }
        }
    }

    void createMeshBuffers()
    {
        vk::BufferUsageFlags usage{
            vkBU::eAccelerationStructureBuildInputReadOnlyKHR
            | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress };
        vk::MemoryPropertyFlags properties{ vkMP::eHostVisible | vkMP::eHostCoherent };

        vertexBuffer.create(vertices.size() * sizeof(Vertex), usage, properties);
        vertexBuffer.copy(vertices.data());

        indexBuffer.create(indices.size() * sizeof(uint32_t), usage, properties);
        indexBuffer.copy(indices.data());

        primitiveBuffer.create(primitiveMaterials.size() * sizeof(int), vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress, properties);
        primitiveBuffer.copy(primitiveMaterials.data());
    }

    void createBottomLevelAS()
    {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangleData;
        triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
        triangleData.setVertexData(vertexBuffer.deviceAddress);
        triangleData.setVertexStride(sizeof(Vertex));
        triangleData.setMaxVertex(vertices.size());
        triangleData.setIndexType(vk::IndexType::eUint32);
        triangleData.setIndexData(indexBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry;
        geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
        geometry.setGeometry({ triangleData });
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        uint32_t primitiveCount = indices.size() / 3;
        bottomAccel.create(geometry, vk::AccelerationStructureTypeKHR::eBottomLevel, primitiveCount);
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
        instancesBuffer.create(sizeof(vk::AccelerationStructureInstanceKHR),
                               vkBU::eAccelerationStructureBuildInputReadOnlyKHR | vkBU::eShaderDeviceAddress,
                               vkMP::eHostVisible | vkMP::eHostCoherent);
        instancesBuffer.copy(&asInstance);

        vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
        instancesData.setArrayOfPointers(false);
        instancesData.setData(instancesBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry;
        geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
        geometry.setGeometry({ instancesData });
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        uint32_t primitiveCount = 1;
        topAccel.create(geometry, vk::AccelerationStructureTypeKHR::eTopLevel, primitiveCount);
    }

    void loadShaders()
    {
        const uint32_t raygenIndex = 0;
        const uint32_t missIndex = 1;
        const uint32_t closestHitIndex = 2;

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
                               VK_SHADER_UNUSED_KHR, closestHitIndex, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR });
    }

    vk::UniqueShaderModule createShaderModule(const std::string& filename)
    {
        const std::vector<char> code = readFile(filename);
        return Context::device->createShaderModuleUnique({ {}, code.size(), reinterpret_cast<const uint32_t*>(code.data()) });
    }

    void createRayTracingPipeLine()
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        using vkSS = vk::ShaderStageFlagBits;
        bindings.push_back({ 0, vkDT::eAccelerationStructureKHR, 1, vkSS::eRaygenKHR }); // Binding = 0 : TLAS
        bindings.push_back({ 1, vkDT::eStorageImage, 1, vkSS::eRaygenKHR });             // Binding = 1 : Storage image
        bindings.push_back({ 2, vkDT::eStorageImage, 1, vkSS::eRaygenKHR });             // Binding = 1 : Storage image
        bindings.push_back({ 3, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 2 : Vertices
        bindings.push_back({ 4, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 3 : Indices
        bindings.push_back({ 5, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 4 : Materials
        bindings.push_back({ 6, vkDT::eUniformBuffer, 1, vkSS::eRaygenKHR });            // Binding = 5 : Uniform data

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
        rtProperties = Context::physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vkRTP>().get<vkRTP>();

        // Calculate SBT size
        uint32_t handleSize = rtProperties.shaderGroupHandleSize;
        size_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
        size_t groupCount = shaderGroups.size();
        size_t sbtSize = groupCount * handleSizeAligned;

        // Get shader group handles
        std::vector<uint8_t> shaderHandleStorage(sbtSize);
        vk::Result res = Context::device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize,
                                                                             shaderHandleStorage.data());
        if (res != vk::Result::eSuccess) {
            throw std::runtime_error("failed to get ray tracing shader group handles.");
        }

        vk::BufferUsageFlags usage = vkBU::eShaderBindingTableKHR | vkBU::eTransferSrc | vkBU::eShaderDeviceAddress;
        vk::MemoryPropertyFlags properties = vkMP::eHostVisible | vkMP::eHostCoherent;

        raygenSBT.create(handleSize, usage, properties);
        raygenSBT.copy(shaderHandleStorage.data() + 0 * handleSizeAligned);

        missSBT.create(handleSize, usage, properties);
        missSBT.copy(shaderHandleStorage.data() + 1 * handleSizeAligned);

        hitSBT.create(handleSize, usage, properties);
        hitSBT.copy(shaderHandleStorage.data() + 2 * handleSizeAligned);
    }

    void createDescriptorSets()
    {
        descSet = Context::allocateDescSet(*descSetLayout);
        updateDescSet();
    }

    void updateDescSet()
    {
        vk::WriteDescriptorSetAccelerationStructureKHR asInfo{ *topAccel.handle };
        vk::WriteDescriptorSet asWrite{};
        asWrite.setDstSet(*descSet);
        asWrite.setDescriptorType(vkDT::eAccelerationStructureKHR);
        asWrite.setDescriptorCount(1);
        asWrite.setDstBinding(0);
        asWrite.setPNext(&asInfo);

        std::vector<vk::WriteDescriptorSet> writeDescSets;
        writeDescSets.push_back(asWrite);
        writeDescSets.push_back(inputImage.createWrite(*descSet, vkDT::eStorageImage, 1));
        writeDescSets.push_back(outputImage.createWrite(*descSet, vkDT::eStorageImage, 2));
        writeDescSets.push_back(vertexBuffer.createWrite(*descSet, vkDT::eStorageBuffer, 3));
        writeDescSets.push_back(indexBuffer.createWrite(*descSet, vkDT::eStorageBuffer, 4));
        writeDescSets.push_back(primitiveBuffer.createWrite(*descSet, vkDT::eStorageBuffer, 5));
        Context::device->updateDescriptorSets(writeDescSets, nullptr);
    }

    vk::WriteDescriptorSet createImageWrite(vk::DescriptorImageInfo imageInfo, vk::DescriptorType type, uint32_t binding)
    {
        vk::WriteDescriptorSet imageWrite{};
        imageWrite.setDstSet(*descSet);
        imageWrite.setDescriptorType(type);
        imageWrite.setDescriptorCount(1);
        imageWrite.setDstBinding(binding);
        imageWrite.setImageInfo(imageInfo);
        return imageWrite;
    }

    vk::WriteDescriptorSet createBufferWrite(vk::DescriptorBufferInfo bufferInfo, vk::DescriptorType type, uint32_t binding)
    {
        vk::WriteDescriptorSet bufferWrite{};
        bufferWrite.setDstSet(*descSet);
        bufferWrite.setDescriptorType(type);
        bufferWrite.setDescriptorCount(1);
        bufferWrite.setDstBinding(binding);
        bufferWrite.setBufferInfo(bufferInfo);
        return bufferWrite;
    }

    void recordCommandBuffers(vk::CommandBuffer commandBuffer, vk::Image swapchainImage)
    {
        commandBuffer.begin(vk::CommandBufferBeginInfo{});
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0, *descSet, nullptr);
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, sizeof(PushConstants), &pushConstants);

        vk::StridedDeviceAddressRegionKHR raygenRegion = createAddressRegion(raygenSBT.deviceAddress);
        vk::StridedDeviceAddressRegionKHR missRegion = createAddressRegion(missSBT.deviceAddress);
        vk::StridedDeviceAddressRegionKHR hitRegion = createAddressRegion(hitSBT.deviceAddress);
        commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);

        setImageLayout(commandBuffer, *outputImage.image, vkIL::eUndefined, vkIL::eTransferSrcOptimal);
        setImageLayout(commandBuffer, *inputImage.image, vkIL::eUndefined, vkIL::eTransferDstOptimal);
        setImageLayout(commandBuffer, swapchainImage, vkIL::eUndefined, vkIL::eTransferDstOptimal);

        copyImage(commandBuffer, *outputImage.image, *inputImage.image);
        copyImage(commandBuffer, *outputImage.image, swapchainImage);

        setImageLayout(commandBuffer, *outputImage.image, vkIL::eTransferSrcOptimal, vkIL::eGeneral);
        setImageLayout(commandBuffer, *inputImage.image, vkIL::eTransferDstOptimal, vkIL::eGeneral);
        setImageLayout(commandBuffer, swapchainImage, vkIL::eTransferDstOptimal, vkIL::ePresentSrcKHR);

        commandBuffer.end();
    }

    vk::StridedDeviceAddressRegionKHR createAddressRegion(vk::DeviceAddress deviceAddress)
    {
        vk::StridedDeviceAddressRegionKHR region{};
        region.setDeviceAddress(deviceAddress);
        region.setStride(rtProperties.shaderGroupHandleAlignment);
        region.setSize(rtProperties.shaderGroupHandleAlignment);
        return region;
    }

    void copyImage(vk::CommandBuffer& cmdBuf, vk::Image& srcImage, vk::Image& dstImage)
    {
        vk::ImageCopy copyRegion{};
        copyRegion.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setExtent({ WIDTH, HEIGHT, 1 });
        cmdBuf.copyImage(srcImage, vkIL::eTransferSrcOptimal,
                         dstImage, vkIL::eTransferDstOptimal, copyRegion);
    }


    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            imageAvailableSemaphores[i] = Context::device->createSemaphoreUnique({});
            renderFinishedSemaphores[i] = Context::device->createSemaphoreUnique({});
            inFlightFences[i] = Context::device->createFence({ vk::FenceCreateFlagBits::eSignaled });
        }
    }

    void present(uint32_t imageIndex)
    {
        vk::PresentInfoKHR presentInfo;
        presentInfo.setWaitSemaphores(*renderFinishedSemaphores[currentFrame]);
        presentInfo.setSwapchains(*Context::swapChain);
        presentInfo.setImageIndices(imageIndex);
        Context::queue.presentKHR(presentInfo);
    }

    void drawFrame()
    {
        Context::device->waitForFences(inFlightFences[currentFrame], true, UINT64_MAX);
        Context::device->resetFences(inFlightFences[currentFrame]);

        uint32_t imageIndex = acquireNextImage();

        recordCommandBuffers(*drawCommandBuffers[imageIndex], Context::swapchainImages[imageIndex]);

        // Submit draw command
        vk::PipelineStageFlags waitStage{ vk::PipelineStageFlagBits::eRayTracingShaderKHR };
        vk::SubmitInfo submitInfo;
        submitInfo.setWaitSemaphores(*imageAvailableSemaphores[currentFrame]);
        submitInfo.setWaitDstStageMask(waitStage);
        submitInfo.setCommandBuffers(*drawCommandBuffers[imageIndex]);
        submitInfo.setSignalSemaphores(*renderFinishedSemaphores[currentFrame]);
        Context::queue.submit(submitInfo, inFlightFences[currentFrame]);
        present(imageIndex);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        pushConstants.frame++;
    }

    uint32_t acquireNextImage()
    {
        auto res = Context::device->acquireNextImageKHR(*Context::swapChain, UINT64_MAX, *imageAvailableSemaphores[currentFrame]);
        if (res.result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to acquire next image!");
        }
        return res.value;
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
