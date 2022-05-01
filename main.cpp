
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

uint32_t findMemoryType(const vk::PhysicalDevice physicalDevice, const uint32_t typeFilter,
                        const vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type");
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

// ----------------------------------------------------------------------------------------------------------
// Structs
// ----------------------------------------------------------------------------------------------------------
struct Buffer
{
    vk::Device device;
    vk::PhysicalDevice physicalDevice;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DeviceSize size;
    vk::BufferUsageFlags usage;
    uint64_t deviceAddress;
    void* mapped = nullptr;
    vk::DescriptorBufferInfo bufferInfo;

    void create(vk::Device device, vk::DeviceSize size, vk::BufferUsageFlags usage)
    {
        this->device = device;
        this->size = size;
        this->usage = usage;
        buffer = device.createBufferUnique({ {}, size, usage });
    }

    void bindMemory(vk::PhysicalDevice physicalDevice, vk::MemoryPropertyFlags properties)
    {
        vk::MemoryRequirements requirements = device.getBufferMemoryRequirements(*buffer);
        uint32_t memoryTypeIndex = findMemoryType(physicalDevice, requirements.memoryTypeBits, properties);
        vk::MemoryAllocateInfo allocInfo{ requirements.size, memoryTypeIndex };

        if (usage & vk::BufferUsageFlagBits::eShaderDeviceAddress) {
            vk::MemoryAllocateFlagsInfo flagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };
            allocInfo.pNext = &flagsInfo;

            memory = device.allocateMemoryUnique(allocInfo);
            device.bindBufferMemory(*buffer, *memory, 0);

            vk::BufferDeviceAddressInfoKHR bufferDeviceAI{ *buffer };
            deviceAddress = device.getBufferAddressKHR(&bufferDeviceAI);
        } else {
            memory = device.allocateMemoryUnique(allocInfo);
            device.bindBufferMemory(*buffer, *memory, 0);
        }
    }

    void copy(void* data)
    {
        if (!mapped) {
            mapped = device.mapMemory(*memory, 0, size);
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
    vk::Device device;
    vk::UniqueImage image;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory memory;
    vk::Extent2D extent;
    vk::Format format;
    vk::ImageLayout imageLayout;
    vk::DescriptorImageInfo imageInfo;

    void create(vk::Device device, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage)
    {
        this->device = device;
        this->extent = extent;
        this->format = format;

        vk::ImageCreateInfo createInfo;
        createInfo.setImageType(vk::ImageType::e2D);
        createInfo.setExtent({ extent.width, extent.height, 1 });
        createInfo.setMipLevels(1);
        createInfo.setArrayLayers(1);
        createInfo.setFormat(format);
        createInfo.setUsage(usage);
        image = device.createImageUnique(createInfo);
    }

    void bindMemory(vk::PhysicalDevice physicalDevice)
    {
        vk::MemoryRequirements requirements = device.getImageMemoryRequirements(*image);
        uint32_t memoryTypeIndex = findMemoryType(physicalDevice, requirements.memoryTypeBits,
                                                  vk::MemoryPropertyFlagBits::eDeviceLocal);
        memory = device.allocateMemoryUnique({ requirements.size, memoryTypeIndex });
        device.bindImageMemory(*image, *memory, 0);
    }

    void createImageView()
    {
        vk::ImageViewCreateInfo createInfo;
        createInfo.setImage(*image);
        createInfo.setViewType(vk::ImageViewType::e2D);
        createInfo.setFormat(format);
        createInfo.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });
        view = device.createImageViewUnique(createInfo);
    }

    vk::WriteDescriptorSet createWrite(vk::DescriptorSet& descSet, vk::DescriptorType type,
                                       uint32_t binding)
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

struct AccelerationStructure
{
    vk::UniqueAccelerationStructureKHR handle;
    Buffer buffer;

    vk::Device device;
    vk::PhysicalDevice physicalDevice;
    vk::AccelerationStructureTypeKHR type;
    uint32_t primitiveCount;
    vk::DeviceSize size;
    vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo;
    uint64_t deviceAddress;

    void createBuffer(vk::Device device, vk::PhysicalDevice physicalDevice,
                      vk::AccelerationStructureGeometryKHR geometry,
                      vk::AccelerationStructureTypeKHR type, uint32_t primitiveCount)
    {
        this->device = device;
        this->physicalDevice = physicalDevice;
        this->type = type;
        this->primitiveCount = primitiveCount;

        buildGeometryInfo.setType(type);
        buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        buildGeometryInfo.setGeometries(geometry);

        vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
        size = buildSizesInfo.accelerationStructureSize;
        buffer.create(device, size, vkBU::eAccelerationStructureStorageKHR | vkBU::eShaderDeviceAddress);
        buffer.bindMemory(physicalDevice, vkMP::eDeviceLocal);
    }

    void create()
    {
        vk::AccelerationStructureCreateInfoKHR createInfo;
        createInfo.setBuffer(*buffer.buffer);
        createInfo.setSize(size);
        createInfo.setType(type);
        handle = device.createAccelerationStructureKHRUnique(createInfo);
    }

    void build(vk::CommandBuffer cmdBuf)
    {
        Buffer scratchBuffer;
        scratchBuffer.create(device, size, vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
        scratchBuffer.bindMemory(physicalDevice, vkMP::eDeviceLocal);
        buildGeometryInfo.setScratchData(scratchBuffer.deviceAddress);
        buildGeometryInfo.setDstAccelerationStructure(*handle);

        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{ primitiveCount , 0, 0, 0 };
        cmdBuf.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
    }
};

struct UniformData
{
    int frame = 0;
};

class Application
{
public:
    ~Application()
    {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            device->destroyFence(inFlightFences[i]);
        }
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
    }

private:
    GLFWwindow* window;
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::UniqueSurfaceKHR surface;

    vk::UniqueDevice device;
    vk::PhysicalDevice physicalDevice;
    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties;

    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> drawCommandBuffers;

    uint32_t queueFamily;
    vk::Queue queue;

    vk::UniqueSwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;

    Image inputImage;
    Image outputImage;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    Buffer vertexBuffer;
    Buffer indexBuffer;

    std::vector<Material> primitiveMaterials;
    Buffer primitiveBuffer;

    AccelerationStructure bottomLevelAS;
    AccelerationStructure topLevelAS;

    std::vector<vk::UniqueShaderModule> shaderModules;
    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages;
    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups;

    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniqueDescriptorSetLayout descSetLayout;

    Buffer raygenSBT;
    Buffer missSBT;
    Buffer hitSBT;

    vk::UniqueDescriptorPool descPool;
    vk::UniqueDescriptorSet descSet;

    std::vector<vk::UniqueSemaphore> imageAvailableSemaphores;
    std::vector<vk::UniqueSemaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    size_t currentFrame = 0;

    UniformData uniformData;
    Buffer uniformBuffer;

    const std::vector<const char*> requiredExtensions{
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

    void initWindow()
    {
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "VulkanPathtracing", nullptr, nullptr);
    }

    void initVulkan()
    {
        createInstance();
        createSurface();
        createDevice();
        createSwapChain();
        createImage(inputImage);
        createImage(outputImage);
        loadMesh();
        createMeshBuffers();
        createUniformBuffer();
        createBottomLevelAS();
        createTopLevelAS();
        loadShaders();
        createRayTracingPipeLine();
        createShaderBindingTable();
        createDescriptorPool();
        createDescriptorSets();
        buildCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
            if (uniformData.frame % 10 == 0) {
                std::cout << "frame: " << uniformData.frame << std::endl;
            }
        }
        device->waitIdle();
    }

    void createInstance()
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

        // Create instance
        vk::ApplicationInfo appInfo;
        appInfo.setPApplicationName("VulkanPathtracing");
        appInfo.setApiVersion(VK_API_VERSION_1_2);
        instance = vk::createInstanceUnique({ {}, &appInfo, layers, extensions });
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

        // Pick first gpu
        physicalDevice = instance->enumeratePhysicalDevices().front();

        // Create debug messenger
        using vkDUMS = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        using vkDUMT = vk::DebugUtilsMessageTypeFlagBitsEXT;
        vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
        messengerInfo.setMessageSeverity(vkDUMS::eWarning | vkDUMS::eError);
        messengerInfo.setMessageType(vkDUMT::eGeneral | vkDUMT::ePerformance | vkDUMT::eValidation);
        messengerInfo.setPfnUserCallback(&debugUtilsMessengerCallback);
        messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo);
    }

    void createSurface()
    {
        VkSurfaceKHR _surface;
        VkResult res = glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), { *instance });
    }

    void createDevice()
    {
        findQueueFamilies();

        float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo{ {}, queueFamily, 1, &queuePriority };

        // Set physical device features
        vk::PhysicalDeviceFeatures deviceFeatures;
        vk::DeviceCreateInfo createInfo{ {}, queueCreateInfo, {}, requiredExtensions, &deviceFeatures };
        vk::StructureChain<vk::DeviceCreateInfo,
            vk::PhysicalDeviceBufferDeviceAddressFeatures,
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR>
            createInfoChain{ createInfo, {true}, {true}, {true} };

        device = physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

        queue = device->getQueue(queueFamily, 0);

        vk::CommandPoolCreateInfo poolInfo;
        poolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        poolInfo.setQueueFamilyIndex(queueFamily);
        commandPool = device->createCommandPoolUnique(poolInfo);
    }

    void findQueueFamilies()
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

    void createSwapChain()
    {
        vk::SwapchainCreateInfoKHR createInfo{};
        createInfo.setSurface(*surface);
        createInfo.setMinImageCount(3);
        createInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
        createInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
        createInfo.setImageExtent({ WIDTH, HEIGHT });
        createInfo.setImageArrayLayers(1);
        createInfo.setImageUsage(vkIU::eTransferDst);
        createInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
        createInfo.setPresentMode(vk::PresentModeKHR::eFifo);
        createInfo.setClipped(true);
        swapChain = device->createSwapchainKHRUnique(createInfo);
        swapChainImages = device->getSwapchainImagesKHR(*swapChain);
    }

    void createImage(Image& image)
    {
        image.create(*device, { WIDTH, HEIGHT }, vk::Format::eB8G8R8A8Unorm, vkIU::eStorage | vkIU::eTransferSrc | vkIU::eTransferDst);
        image.bindMemory(physicalDevice);
        image.createImageView();

        // Set image layout
        image.imageLayout = vk::ImageLayout::eGeneral;
        oneTimeSubmit(
            [&](vk::CommandBuffer cmdBuf) {
                setImageLayout(cmdBuf, *image.image, vk::ImageLayout::eUndefined, image.imageLayout);
            });
    }

    void oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func)
    {
        vk::CommandBufferAllocateInfo allocInfo{ *commandPool, vk::CommandBufferLevel::ePrimary, 1 };
        vk::UniqueCommandBuffer cmdBuf = std::move(device->allocateCommandBuffersUnique(allocInfo).front());

        cmdBuf->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        func(*cmdBuf);
        cmdBuf->end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(*cmdBuf);
        queue.submit(submitInfo);
        queue.waitIdle();
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

        vertexBuffer.create(*device, vertices.size() * sizeof(Vertex), usage);
        vertexBuffer.bindMemory(physicalDevice, properties);
        vertexBuffer.copy(vertices.data());

        indexBuffer.create(*device, indices.size() * sizeof(uint32_t), usage);
        indexBuffer.bindMemory(physicalDevice, properties);
        indexBuffer.copy(indices.data());

        primitiveBuffer.create(*device, primitiveMaterials.size() * sizeof(int), vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
        primitiveBuffer.bindMemory(physicalDevice, properties);
        primitiveBuffer.copy(primitiveMaterials.data());
    }

    void createUniformBuffer()
    {
        uniformBuffer.create(*device, sizeof(UniformData), vkBU::eUniformBuffer);
        uniformBuffer.bindMemory(physicalDevice, vkMP::eHostVisible | vkMP::eHostCoherent);
        uniformBuffer.copy(&uniformData);
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
        bottomLevelAS.createBuffer(*device, physicalDevice, geometry,
                                   vk::AccelerationStructureTypeKHR::eBottomLevel, primitiveCount);
        bottomLevelAS.create();
        oneTimeSubmit([&](vk::CommandBuffer cmdBuf) { bottomLevelAS.build(cmdBuf); });
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
        asInstance.setAccelerationStructureReference(bottomLevelAS.buffer.deviceAddress);
        asInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

        Buffer instancesBuffer;
        instancesBuffer.create(*device, sizeof(vk::AccelerationStructureInstanceKHR),
                               vkBU::eAccelerationStructureBuildInputReadOnlyKHR | vkBU::eShaderDeviceAddress);
        instancesBuffer.bindMemory(physicalDevice, vkMP::eHostVisible | vkMP::eHostCoherent);
        instancesBuffer.copy(&asInstance);

        vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
        instancesData.setArrayOfPointers(false);
        instancesData.setData(instancesBuffer.deviceAddress);

        vk::AccelerationStructureGeometryKHR geometry;
        geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
        geometry.setGeometry({ instancesData });
        geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

        uint32_t primitiveCount = 1;
        topLevelAS.createBuffer(*device, physicalDevice, geometry,
                                vk::AccelerationStructureTypeKHR::eTopLevel, primitiveCount);
        topLevelAS.create();
        oneTimeSubmit([&](vk::CommandBuffer cmdBuf) { topLevelAS.build(cmdBuf); });
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
        return device->createShaderModuleUnique({ {}, code.size(), reinterpret_cast<const uint32_t*>(code.data()) });
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

        descSetLayout = device->createDescriptorSetLayoutUnique({ {}, bindings });
        pipelineLayout = device->createPipelineLayoutUnique({ {}, *descSetLayout });

        // Create pipeline
        vk::RayTracingPipelineCreateInfoKHR createInfo;
        createInfo.setStages(shaderStages);
        createInfo.setGroups(shaderGroups);
        createInfo.setMaxPipelineRayRecursionDepth(4);
        createInfo.setLayout(*pipelineLayout);
        auto res = device->createRayTracingPipelineKHRUnique(nullptr, nullptr, createInfo);
        if (res.result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to create ray tracing pipeline.");
        }
        pipeline = std::move(res.value);
    }

    void createShaderBindingTable()
    {
        // Get Ray Tracing Properties
        using vkRTP = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR;
        rtProperties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vkRTP>().get<vkRTP>();

        // Calculate SBT size
        uint32_t handleSize = rtProperties.shaderGroupHandleSize;
        size_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
        size_t groupCount = shaderGroups.size();
        size_t sbtSize = groupCount * handleSizeAligned;

        // Get shader group handles
        std::vector<uint8_t> shaderHandleStorage(sbtSize);
        vk::Result res = device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize,
                                                                    shaderHandleStorage.data());
        if (res != vk::Result::eSuccess) {
            throw std::runtime_error("failed to get ray tracing shader group handles.");
        }

        vk::BufferUsageFlags usage = vkBU::eShaderBindingTableKHR | vkBU::eTransferSrc | vkBU::eShaderDeviceAddress;
        vk::MemoryPropertyFlags properties = vkMP::eHostVisible | vkMP::eHostCoherent;

        raygenSBT.create(*device, handleSize, usage);
        raygenSBT.bindMemory(physicalDevice, properties);
        raygenSBT.copy(shaderHandleStorage.data() + 0 * handleSizeAligned);

        missSBT.create(*device, handleSize, usage);
        missSBT.bindMemory(physicalDevice, properties);
        missSBT.copy(shaderHandleStorage.data() + 1 * handleSizeAligned);

        hitSBT.create(*device, handleSize, usage);
        hitSBT.bindMemory(physicalDevice, properties);
        hitSBT.copy(shaderHandleStorage.data() + 2 * handleSizeAligned);
    }

    void createDescriptorPool()
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

    void createDescriptorSets()
    {
        descSet = std::move(device->allocateDescriptorSetsUnique({ *descPool, *descSetLayout }).front());
        updateDescSet();
    }

    void updateDescSet()
    {
        vk::WriteDescriptorSetAccelerationStructureKHR asInfo{ *topLevelAS.handle };
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
        writeDescSets.push_back(uniformBuffer.createWrite(*descSet, vkDT::eUniformBuffer, 6));
        device->updateDescriptorSets(writeDescSets, nullptr);
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

    void buildCommandBuffers()
    {
        allocateDrawCommandBuffers();
        for (int32_t i = 0; i < drawCommandBuffers.size(); ++i) {
            drawCommandBuffers[i]->begin(vk::CommandBufferBeginInfo{});
            drawCommandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
            drawCommandBuffers[i]->bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR,
                                                      *pipelineLayout, 0, *descSet, nullptr);
            traceRays(*drawCommandBuffers[i]);

            setImageLayout(*drawCommandBuffers[i], *outputImage.image, vkIL::eUndefined, vkIL::eTransferSrcOptimal);
            setImageLayout(*drawCommandBuffers[i], *inputImage.image, vkIL::eUndefined, vkIL::eTransferDstOptimal);
            setImageLayout(*drawCommandBuffers[i], swapChainImages[i], vkIL::eUndefined, vkIL::eTransferDstOptimal);

            copyImage(*drawCommandBuffers[i], *outputImage.image, *inputImage.image);
            copyImage(*drawCommandBuffers[i], *outputImage.image, swapChainImages[i]);

            setImageLayout(*drawCommandBuffers[i], *outputImage.image, vkIL::eTransferSrcOptimal, vkIL::eGeneral);
            setImageLayout(*drawCommandBuffers[i], *inputImage.image, vkIL::eTransferDstOptimal, vkIL::eGeneral);
            setImageLayout(*drawCommandBuffers[i], swapChainImages[i], vkIL::eTransferDstOptimal, vkIL::ePresentSrcKHR);

            drawCommandBuffers[i]->end();
        }
    }

    void traceRays(vk::CommandBuffer& cmdBuf)
    {
        vk::StridedDeviceAddressRegionKHR raygenRegion = createAddressRegion(raygenSBT.deviceAddress);
        vk::StridedDeviceAddressRegionKHR missRegion = createAddressRegion(missSBT.deviceAddress);
        vk::StridedDeviceAddressRegionKHR hitRegion = createAddressRegion(hitSBT.deviceAddress);
        cmdBuf.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);
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

    void allocateDrawCommandBuffers()
    {
        vk::CommandBufferAllocateInfo allocInfo;
        allocInfo.setCommandPool(*commandPool);
        allocInfo.setCommandBufferCount(swapChainImages.size());
        drawCommandBuffers = device->allocateCommandBuffersUnique(allocInfo);
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            imageAvailableSemaphores[i] = device->createSemaphoreUnique({});
            renderFinishedSemaphores[i] = device->createSemaphoreUnique({});
            inFlightFences[i] = device->createFence({ vk::FenceCreateFlagBits::eSignaled });
        }
    }

    void present(uint32_t imageIndex)
    {
        vk::PresentInfoKHR presentInfo;
        presentInfo.setWaitSemaphores(*renderFinishedSemaphores[currentFrame]);
        presentInfo.setSwapchains(*swapChain);
        presentInfo.setImageIndices(imageIndex);
        queue.presentKHR(presentInfo);
    }

    void drawFrame()
    {
        device->waitForFences(inFlightFences[currentFrame], true, UINT64_MAX);
        device->resetFences(inFlightFences[currentFrame]);

        uint32_t imageIndex = acquireNextImage();

        // Submit draw command
        vk::PipelineStageFlags waitStage{ vk::PipelineStageFlagBits::eRayTracingShaderKHR };
        vk::SubmitInfo submitInfo;
        submitInfo.setWaitSemaphores(*imageAvailableSemaphores[currentFrame]);
        submitInfo.setWaitDstStageMask(waitStage);
        submitInfo.setCommandBuffers(*drawCommandBuffers[imageIndex]);
        submitInfo.setSignalSemaphores(*renderFinishedSemaphores[currentFrame]);
        queue.submit(submitInfo, inFlightFences[currentFrame]);
        present(imageIndex);

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
        uniformData.frame++;
        uniformBuffer.copy(&uniformData);
    }

    uint32_t acquireNextImage()
    {
        auto res = device->acquireNextImageKHR(*swapChain, UINT64_MAX, *imageAvailableSemaphores[currentFrame]);
        if (res.result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to acquire next image!");
        }
        return res.value;
    }
};

int main()
{
    Application app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
