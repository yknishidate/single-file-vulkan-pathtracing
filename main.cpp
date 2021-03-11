
#include <set>
#include <fstream>
#include <iostream>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <GLFW/glfw3.h> // include after vulkan.hpp

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

using vkBU = vk::BufferUsageFlagBits;
using vkMP = vk::MemoryPropertyFlagBits;

// ----------------------------------------------------------------------------------------------------------
// Globals
// ----------------------------------------------------------------------------------------------------------
constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;
#ifdef _DEBUG
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = false;
#endif
std::vector<const char*> validationLayers;

// ----------------------------------------------------------------------------------------------------------
// Functuins
// ----------------------------------------------------------------------------------------------------------
VKAPI_ATTR VkBool32 VKAPI_CALL
debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                            VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                            void* /*pUserData*/)
{
    std::cerr << "messageIndexName   = " << pCallbackData->pMessageIdName << "\n";
    for (uint8_t i = 0; i < pCallbackData->objectCount; i++) {
        std::cerr << "objectType      = " << vk::to_string(static_cast<vk::ObjectType>(
            pCallbackData->pObjects[i].objectType)) << "\n";
    }
    std::cerr << pCallbackData->pMessage << "\n\n";
    return VK_FALSE;
}

void transitionImageLayout(vk::CommandBuffer cmdBuf, vk::Image image,
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

    // Source layouts (old)
    using vkAF = vk::AccessFlagBits;
    switch (oldLayout) {
        case vk::ImageLayout::eUndefined:
            barrier.srcAccessMask = {};
            break;
        case vk::ImageLayout::ePreinitialized:
            barrier.srcAccessMask = vkAF::eHostWrite;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            barrier.srcAccessMask = vkAF::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            barrier.srcAccessMask = vkAF::eDepthStencilAttachmentWrite;
            break;
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

    // Target layouts (new)
    switch (newLayout) {
        case vk::ImageLayout::eTransferDstOptimal:
            barrier.dstAccessMask = vkAF::eTransferWrite;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            barrier.dstAccessMask = vkAF::eTransferRead;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            barrier.dstAccessMask = vkAF::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            barrier.dstAccessMask = barrier.dstAccessMask | vkAF::eDepthStencilAttachmentWrite;
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

uint32_t findMemoryType(const vk::PhysicalDevice physicalDevice,
                        const uint32_t typeFilter,
                        const vk::MemoryPropertyFlags properties)
{
    vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i) {
        if ((typeFilter & (1 << i))
            && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type");
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats)
{
    if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
        return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
    }
    for (const auto& format : formats) {
        if (format.format == vk::Format::eB8G8R8A8Unorm
            && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }
    throw std::runtime_error("found no suitable surface format");
}

vk::PresentModeKHR chooseSwapPresentMode(
    const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eFifoRelaxed) {
            return availablePresentMode;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }

    vk::Extent2D actualExtent{ WIDTH, HEIGHT };
    actualExtent.width = std::clamp(actualExtent.width,
                                    capabilities.minImageExtent.width,
                                    capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height,
                                     capabilities.minImageExtent.height,
                                     capabilities.maxImageExtent.height);
    return actualExtent;
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

    void create(vk::Device device, vk::DeviceSize size, vk::BufferUsageFlags usage)
    {
        this->device = device;
        this->size = size;
        this->usage = usage;

        buffer = device.createBufferUnique({ {}, size, usage });
    }

    void bindMemory(vk::PhysicalDevice physicalDevice, vk::MemoryPropertyFlags properties)
    {
        auto requirements = device.getBufferMemoryRequirements(*buffer);
        auto memoryTypeIndex = findMemoryType(physicalDevice, requirements.memoryTypeBits,
                                              properties);
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

    void fillData(void* data)
    {
        mapped = device.mapMemory(*memory, 0, size);
        memcpy(mapped, data, static_cast<size_t>(size));
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

    void create(vk::Device device, vk::Extent2D extent,
                vk::Format format, vk::ImageUsageFlags usage)
    {
        this->device = device;
        this->extent = extent;
        this->format = format;

        image = device.createImageUnique(
            vk::ImageCreateInfo{}
            .setImageType(vk::ImageType::e2D)
            .setExtent({ extent.width, extent.height, 1 })
            .setMipLevels(1)
            .setArrayLayers(1)
            .setFormat(format)
            .setTiling(vk::ImageTiling::eOptimal)
            .setUsage(usage));
    }

    void bindMemory(vk::PhysicalDevice physicalDevice)
    {
        auto requirements = device.getImageMemoryRequirements(*image);
        auto memoryTypeIndex = findMemoryType(physicalDevice, requirements.memoryTypeBits,
                                              vk::MemoryPropertyFlagBits::eDeviceLocal);
        memory = device.allocateMemoryUnique({ requirements.size, memoryTypeIndex });
        device.bindImageMemory(*image, *memory, 0);
    }

    void createImageView()
    {
        view = device.createImageViewUnique(
            vk::ImageViewCreateInfo{}
            .setImage(*image)
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(format)
            .setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }));
    }
};

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec4 color;
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

        auto buildSizesInfo = device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
        size = buildSizesInfo.accelerationStructureSize;
        buffer.create(device, size,
                      vkBU::eAccelerationStructureStorageKHR | vkBU::eShaderDeviceAddress);
        buffer.bindMemory(physicalDevice, vkMP::eDeviceLocal);
    }

    void create()
    {
        handle = device.createAccelerationStructureKHRUnique(
            vk::AccelerationStructureCreateInfoKHR{}
            .setBuffer(*buffer.buffer)
            .setSize(size)
            .setType(type));
    }

    void build(vk::CommandBuffer commandBuffer)
    {
        Buffer scratchBuffer;
        scratchBuffer.create(device, size, vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress); // ? shaderDevice
        scratchBuffer.bindMemory(physicalDevice, vkMP::eDeviceLocal);
        buildGeometryInfo.setScratchData(scratchBuffer.deviceAddress);
        buildGeometryInfo.setDstAccelerationStructure(*handle);

        vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{ primitiveCount , 0, 0, 0 };
        commandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
        // ? get device address
    }
};

// ----------------------------------------------------------------------------------------------------------
// Application
// ----------------------------------------------------------------------------------------------------------
class Application
{
public:

    ~Application()
    {
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

    uint32_t graphicsFamily;
    uint32_t presentFamily;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;

    vk::UniqueSwapchainKHR swapChain;
    vk::PresentModeKHR presentMode;
    vk::Format format;
    vk::Extent2D extent;
    std::vector<vk::Image> swapChainImages;

    Image storageImage;

    std::vector<Vertex> vertices{ { {1.0f, 1.0f, 0.0f} },
                                  { {-1.0f, 1.0f, 0.0f} },
                                  { {0.0f, -1.0f, 0.0f} } };
    std::vector<uint32_t> indices{ 0, 1, 2 };
    Buffer vertexBuffer;
    Buffer indexBuffer;

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
        createStorageImage();
        createMeshBuffers();
        createBottomLevelAS();
        createTopLevelAS();
        loadShaders();
        createRayTracingPipeLine();
        createShaderBindingTable();
        createDescriptorSets();
        buildCommandBuffers();

        //swapChain->initDrawCommandBuffers(*pipeline, *descSets, *shaderManager, *storageImage);
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void createInstance()
    {
        // Get GLFW extensions
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // Setup DynamicLoader (see https://github.com/KhronosGroup/Vulkan-Hpp)
        static vk::DynamicLoader dl;
        auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>(
            "vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        // Add validation layer, extension
        if (enableValidationLayers) {
            validationLayers.push_back("VK_LAYER_KHRONOS_validation");
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        // Create instance
        vk::ApplicationInfo appInfo;
        appInfo.setPApplicationName("VulkanPathtracing");
        appInfo.setApiVersion(VK_API_VERSION_1_2);
        instance = vk::createInstanceUnique({ {}, &appInfo, validationLayers, extensions });
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

        // Get first physical device
        physicalDevice = instance->enumeratePhysicalDevices().front();
        if (enableValidationLayers) {
            createDebugMessenger();
        }
    }

    void createDebugMessenger()
    {
        using vkDUMS = vk::DebugUtilsMessageSeverityFlagBitsEXT;
        using vkDUMT = vk::DebugUtilsMessageTypeFlagBitsEXT;
        messenger = instance->createDebugUtilsMessengerEXTUnique(
            vk::DebugUtilsMessengerCreateInfoEXT{}
            .setMessageSeverity(vkDUMS::eWarning | vkDUMS::eError)
            .setMessageType(vkDUMT::eGeneral | vkDUMT::ePerformance | vkDUMT::eValidation)
            .setPfnUserCallback(&debugUtilsMessengerCallback));
    }

    void createSurface()
    {
        VkSurfaceKHR _surface;
        auto res = glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        vk::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE> _deleter(*instance);
        surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), _deleter);
    }

    void createDevice()
    {
        findQueueFamilies();
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { graphicsFamily, presentFamily };

        // Create queues
        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo{ {}, queueFamily, 1, &queuePriority };
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Set physical device features
        vk::PhysicalDeviceFeatures deviceFeatures;
        deviceFeatures.fillModeNonSolid = true;
        deviceFeatures.samplerAnisotropy = true;

        vk::DeviceCreateInfo createInfo{ {}, queueCreateInfos, validationLayers,
            requiredExtensions, &deviceFeatures };

        // Create structure chain
        // TODO: minimaize
        vk::PhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures;
        indexingFeatures.runtimeDescriptorArray = true;
        vk::StructureChain<vk::DeviceCreateInfo,
            vk::PhysicalDeviceDescriptorIndexingFeaturesEXT,
            vk::PhysicalDeviceBufferDeviceAddressFeatures,
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
            vk::PhysicalDeviceShaderClockFeaturesKHR>
            createInfoChain{ createInfo, indexingFeatures, {true}, {true}, {true}, {true, true} };

        device = physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

        graphicsQueue = device->getQueue(graphicsFamily, 0);
        presentQueue = device->getQueue(presentFamily, 0);

        commandPool = device->createCommandPoolUnique(
            vk::CommandPoolCreateInfo{}
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
            .setQueueFamilyIndex(graphicsFamily));

        std::cout << "created device\n";
    }

    void findQueueFamilies()
    {
        int i = 0;
        for (const auto& queueFamily : physicalDevice.getQueueFamilyProperties()) {
            if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                graphicsFamily = i;
            }
            vk::Bool32 presentSupport = physicalDevice.getSurfaceSupportKHR(i, *surface);
            if (presentSupport) {
                presentFamily = i;
            }
            if (graphicsFamily != -1 && presentFamily != -1) {
                break;
            }
            i++;
        }
    }

    void createSwapChain()
    {
        // Query swapchain support
        auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        auto formats = physicalDevice.getSurfaceFormatsKHR(*surface);
        auto presentModes = physicalDevice.getSurfacePresentModesKHR(*surface);

        // Choose swapchain settings
        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(formats);
        format = surfaceFormat.format;
        presentMode = chooseSwapPresentMode(presentModes);
        extent = chooseSwapExtent(capabilities);
        uint32_t imageCount = capabilities.minImageCount + 1;

        // Create swap chain
        using vkIUF = vk::ImageUsageFlagBits;
        vk::SwapchainCreateInfoKHR createInfo;
        createInfo.surface = *surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = vkIUF::eColorAttachment | vkIUF::eTransferDst;
        createInfo.preTransform = capabilities.currentTransform;
        createInfo.presentMode = presentMode;
        createInfo.clipped = true;
        if (graphicsFamily != presentFamily) {
            uint32_t queueFamilyIndices[] = { graphicsFamily, presentFamily };
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        swapChain = device->createSwapchainKHRUnique(createInfo);
        swapChainImages = device->getSwapchainImagesKHR(*swapChain);

        std::cout << "created swapchain\n";
    }

    void createStorageImage()
    {
        using vkIUF = vk::ImageUsageFlagBits;
        storageImage.create(*device, extent, format, vkIUF::eStorage | vkIUF::eTransferSrc);
        storageImage.bindMemory(physicalDevice);
        storageImage.createImageView();

        // Set image layout
        storageImage.imageLayout = vk::ImageLayout::eGeneral;
        auto commandBuffer = createCommandBuffer();
        transitionImageLayout(*commandBuffer, *storageImage.image,
                              vk::ImageLayout::eUndefined, storageImage.imageLayout);
        submitCommandBuffer(*commandBuffer);

        std::cout << "created storage image\n";
    }

    vk::UniqueCommandBuffer createCommandBuffer()
    {
        vk::CommandBufferAllocateInfo allocInfo{ *commandPool,
            vk::CommandBufferLevel::ePrimary, 1 };
        vk::UniqueCommandBuffer commandBuffer = std::move(
            device->allocateCommandBuffersUnique(allocInfo).front());
        commandBuffer->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        return commandBuffer;
    }

    void submitCommandBuffer(vk::CommandBuffer& cmdBuf)
    {
        cmdBuf.end();

        vk::UniqueFence fence = device->createFenceUnique({});
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(cmdBuf);
        graphicsQueue.submit(submitInfo, *fence);
        auto res = device->waitForFences(*fence, true, UINT64_MAX);
        assert(res == vk::Result::eSuccess);
    }

    void createMeshBuffers()
    {
        vk::BufferUsageFlags usage{ vkBU::eAccelerationStructureBuildInputReadOnlyKHR
            | vkBU::eStorageBuffer | vkBU::eTransferDst | vkBU::eShaderDeviceAddress };
        vk::MemoryPropertyFlags properties{ vkMP::eHostVisible | vkMP::eHostCoherent };

        uint64_t vertexBufferSize = vertices.size() * sizeof(Vertex);
        vertexBuffer.create(*device, vertexBufferSize, usage);
        vertexBuffer.bindMemory(physicalDevice, properties);
        vertexBuffer.fillData(vertices.data());

        uint64_t indexBufferSize = indices.size() * sizeof(uint32_t);
        indexBuffer.create(*device, indexBufferSize, usage);
        indexBuffer.bindMemory(physicalDevice, properties);
        indexBuffer.fillData(indices.data());

        std::cout << "created mesh buffers\n";
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
        auto commandBuffer = createCommandBuffer();
        bottomLevelAS.build(*commandBuffer);
        submitCommandBuffer(*commandBuffer);

        std::cout << "created bottom level as\n";
    }

    void createTopLevelAS()
    {
        VkTransformMatrixKHR transformMatrix = { 1.0f, 0.0f, 0.0f, 0.0f,
                                                 0.0f, 1.0f, 0.0f, 0.0f,
                                                 0.0f, 0.0f, 1.0f, 0.0f };
        vk::AccelerationStructureInstanceKHR asInstance;
        asInstance.setTransform(transformMatrix);
        asInstance.setMask(0xFF);
        asInstance.setAccelerationStructureReference(bottomLevelAS.buffer.deviceAddress);
        asInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

        Buffer instancesBuffer;
        instancesBuffer.create(*device, sizeof(vk::AccelerationStructureInstanceKHR),
                               vkBU::eAccelerationStructureBuildInputReadOnlyKHR
                               | vkBU::eShaderDeviceAddress); // ? shaderDevice
        instancesBuffer.bindMemory(physicalDevice, vkMP::eHostVisible | vkMP::eHostCoherent);
        instancesBuffer.fillData(&asInstance);

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
        auto commandBuffer = createCommandBuffer();
        topLevelAS.build(*commandBuffer);
        submitCommandBuffer(*commandBuffer);

        std::cout << "created top level as\n";
    }

    void loadShaders()
    {
        const uint32_t shaderIndexRaygen = 0;
        const uint32_t shaderIndexMiss = 1;
        const uint32_t shaderIndexClosestHit = 2;

        shaderModules.push_back(createShaderModule("shaders/raygen.rgen.spv"));
        shaderStages.push_back({ {}, vk::ShaderStageFlagBits::eRaygenKHR,
                               *shaderModules.back(), "main" });
        shaderGroups.push_back({ vk::RayTracingShaderGroupTypeKHR::eGeneral,
                               shaderIndexRaygen, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR });

        shaderModules.push_back(createShaderModule("shaders/miss.rmiss.spv"));
        shaderStages.push_back({ {}, vk::ShaderStageFlagBits::eMissKHR,
                               *shaderModules.back(), "main" });
        shaderGroups.push_back({ vk::RayTracingShaderGroupTypeKHR::eGeneral,
                               shaderIndexMiss, VK_SHADER_UNUSED_KHR,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR });

        shaderModules.push_back(createShaderModule("shaders/closesthit.rchit.spv"));
        shaderStages.push_back({ {}, vk::ShaderStageFlagBits::eClosestHitKHR,
                               *shaderModules.back(), "main" });
        shaderGroups.push_back({ vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                               VK_SHADER_UNUSED_KHR, shaderIndexClosestHit,
                               VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR });

        std::cout << "loaded shaders\n";
    }

    vk::UniqueShaderModule createShaderModule(const std::string& filename)
    {
        const std::vector<char> code = readFile(filename);
        return device->createShaderModuleUnique({ {}, code.size(),
                                                reinterpret_cast<const uint32_t*>(code.data()) });
    }

    void createRayTracingPipeLine()
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        // Binding = 0 : TLAS
        bindings.push_back({ 0, vk::DescriptorType::eAccelerationStructureKHR,
                           1, vk::ShaderStageFlagBits::eRaygenKHR });
        // Binding = 1 : Storage image
        bindings.push_back({ 1, vk::DescriptorType::eStorageImage,
                           1, vk::ShaderStageFlagBits::eRaygenKHR });

        // Create layouts
        descSetLayout = device->createDescriptorSetLayoutUnique({ {}, bindings });
        pipelineLayout = device->createPipelineLayoutUnique({ {}, *descSetLayout });

        // Create pipeline
        vk::RayTracingPipelineCreateInfoKHR createInfo;
        createInfo.setStages(shaderStages);
        createInfo.setGroups(shaderGroups);
        createInfo.setMaxPipelineRayRecursionDepth(4);
        createInfo.setLayout(*pipelineLayout);
        auto res = device->createRayTracingPipelineKHRUnique(nullptr, nullptr, createInfo);
        if (res.result == vk::Result::eSuccess) {
            pipeline = std::move(res.value);
        } else {
            throw std::runtime_error("failed to create ray tracing pipeline.");
        }
        std::cout << "created raytracing pipeline\n";
    }

    void createShaderBindingTable()
    {
        // Get Ray Tracing Properties
        rtProperties = physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>()
            .get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

        // Calculate SBT size
        uint32_t handleSize = rtProperties.shaderGroupHandleSize;
        size_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
        size_t groupCount = shaderGroups.size();
        size_t sbtSize = groupCount * handleSizeAligned;

        // Get shader group handles
        std::vector<uint8_t> shaderHandleStorage(sbtSize);
        auto res = device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize,
                                                              shaderHandleStorage.data());
        if (res != vk::Result::eSuccess) {
            throw std::runtime_error("failed to get ray tracing shader group handles.");
        }

        vk::BufferUsageFlags usage = vkBU::eShaderBindingTableKHR
            | vkBU::eTransferSrc | vkBU::eShaderDeviceAddress;
        vk::MemoryPropertyFlags properties = vkMP::eHostVisible | vkMP::eHostCoherent;

        raygenSBT.create(*device, handleSize, usage);
        raygenSBT.bindMemory(physicalDevice, properties);
        raygenSBT.fillData(shaderHandleStorage.data() + 0 * handleSizeAligned);

        missSBT.create(*device, handleSize, usage);
        missSBT.bindMemory(physicalDevice, properties);
        missSBT.fillData(shaderHandleStorage.data() + 1 * handleSizeAligned);

        hitSBT.create(*device, handleSize, usage);
        hitSBT.bindMemory(physicalDevice, properties);
        hitSBT.fillData(shaderHandleStorage.data() + 2 * handleSizeAligned);

        std::cout << "created shader binding table\n";
    }

    void createDescriptorSets()
    {
        createDescPool();

        auto descriptorSets = device->allocateDescriptorSetsUnique(
            vk::DescriptorSetAllocateInfo{}
            .setDescriptorPool(*descPool)
            .setSetLayouts(*descSetLayout));
        descSet = std::move(descriptorSets.front());

        updateDescSet();

        std::cout << "created desc set\n";
    }

    void createDescPool()
    {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            {vk::DescriptorType::eAccelerationStructureKHR, 1},
            {vk::DescriptorType::eStorageImage, 1} };

        descPool = device->createDescriptorPoolUnique(
            vk::DescriptorPoolCreateInfo{}
            .setPoolSizes(poolSizes)
            .setMaxSets(1)
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet));
    }

    void updateDescSet()
    {
        vk::WriteDescriptorSetAccelerationStructureKHR asDesc{ *topLevelAS.handle };
        vk::WriteDescriptorSet asWrite{};
        asWrite.setDstSet(*descSet);
        asWrite.setDstBinding(0);
        asWrite.setDescriptorCount(1);
        asWrite.setDescriptorType(vk::DescriptorType::eAccelerationStructureKHR);
        asWrite.setPNext(&asDesc);

        vk::DescriptorImageInfo imageDesc{};
        imageDesc.setImageView(*storageImage.view);
        imageDesc.setImageLayout(vk::ImageLayout::eGeneral);

        vk::WriteDescriptorSet imageWrite{};
        imageWrite.setDstSet(*descSet);
        imageWrite.setDescriptorType(vk::DescriptorType::eStorageImage);
        imageWrite.setDstBinding(1);
        imageWrite.setImageInfo(imageDesc);

        device->updateDescriptorSets({ asWrite, imageWrite }, nullptr);
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
            copyStorageImage(*drawCommandBuffers[i], swapChainImages[i]);
            drawCommandBuffers[i]->end();
        }

        std::cout << "built command buffers\n";
    }

    void traceRays(vk::CommandBuffer& cmdBuf)
    {
        size_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;

        vk::StridedDeviceAddressRegionKHR raygenRegion{};
        raygenRegion.setDeviceAddress(raygenSBT.deviceAddress);
        raygenRegion.setStride(handleSizeAligned);
        raygenRegion.setSize(handleSizeAligned);

        vk::StridedDeviceAddressRegionKHR missRegion{};
        missRegion.setDeviceAddress(missSBT.deviceAddress);
        missRegion.setStride(handleSizeAligned);
        missRegion.setSize(handleSizeAligned);

        vk::StridedDeviceAddressRegionKHR hitRegion{};
        hitRegion.setDeviceAddress(hitSBT.deviceAddress);
        hitRegion.setStride(handleSizeAligned);
        hitRegion.setSize(handleSizeAligned);

        cmdBuf.traceRaysKHR(raygenRegion, missRegion, hitRegion, {},
                            storageImage.extent.width, storageImage.extent.height, 1);
    }

    void copyStorageImage(vk::CommandBuffer& cmdBuf, vk::Image& swapChainImage)
    {
        transitionImageLayout(cmdBuf, *storageImage.image, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eTransferSrcOptimal);
        transitionImageLayout(cmdBuf, swapChainImage, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eTransferDstOptimal);

        vk::ImageCopy copyRegion{};
        copyRegion.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setSrcOffset({ 0, 0, 0 });
        copyRegion.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setDstOffset({ 0, 0, 0 });
        copyRegion.setExtent({ storageImage.extent.width, storageImage.extent.height, 1 });
        cmdBuf.copyImage(*storageImage.image, vk::ImageLayout::eTransferSrcOptimal,
                         swapChainImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);

        transitionImageLayout(cmdBuf, *storageImage.image, vk::ImageLayout::eTransferSrcOptimal,
                              vk::ImageLayout::eGeneral);
        transitionImageLayout(cmdBuf, swapChainImage, vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::ePresentSrcKHR);
    }

    void allocateDrawCommandBuffers()
    {
        drawCommandBuffers = device->allocateCommandBuffersUnique(
            vk::CommandBufferAllocateInfo{}
            .setCommandPool(*commandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(swapChainImages.size()));
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
