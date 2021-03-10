
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

    vk::ImageMemoryBarrier imageMemoryBarrier{};
    imageMemoryBarrier
        .setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
        .setImage(image)
        .setOldLayout(oldLayout)
        .setNewLayout(newLayout)
        .setSubresourceRange({ vk::ImageAspectFlagBits::eColor , 0, 1, 0, 1 });

    // Source layouts (old)
    using vkAF = vk::AccessFlagBits;
    switch (oldLayout) {
        case vk::ImageLayout::eUndefined:
            imageMemoryBarrier.srcAccessMask = {};
            break;
        case vk::ImageLayout::ePreinitialized:
            imageMemoryBarrier.srcAccessMask = vkAF::eHostWrite;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            imageMemoryBarrier.srcAccessMask = vkAF::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            imageMemoryBarrier.srcAccessMask = vkAF::eDepthStencilAttachmentWrite;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            imageMemoryBarrier.srcAccessMask = vkAF::eTransferRead;
            break;
        case vk::ImageLayout::eTransferDstOptimal:
            imageMemoryBarrier.srcAccessMask = vkAF::eTransferWrite;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            imageMemoryBarrier.srcAccessMask = vkAF::eShaderRead;
            break;
        default:
            break;
    }

    // Target layouts (new)
    switch (newLayout) {
        case vk::ImageLayout::eTransferDstOptimal:
            imageMemoryBarrier.dstAccessMask = vkAF::eTransferWrite;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            imageMemoryBarrier.dstAccessMask = vkAF::eTransferRead;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            imageMemoryBarrier.dstAccessMask = vkAF::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask
                | vkAF::eDepthStencilAttachmentWrite;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            if (imageMemoryBarrier.srcAccessMask == vk::AccessFlags{}) {
                imageMemoryBarrier.srcAccessMask = vkAF::eHostWrite | vkAF::eTransferWrite;
            }
            imageMemoryBarrier.dstAccessMask = vkAF::eShaderRead;
            break;
        default:
            break;
    }

    cmdBuf.pipelineBarrier(srcStageMask, dstStageMask, {}, {}, {}, imageMemoryBarrier);
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
    actualExtent.width = std::min(capabilities.maxImageExtent.width,
                                  std::max(actualExtent.width,
                                           capabilities.minImageExtent.width));
    actualExtent.height = std::min(capabilities.maxImageExtent.height,
                                   std::max(actualExtent.height,
                                            capabilities.minImageExtent.height));
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

        vk::ImageCreateInfo imageInfo;
        imageInfo.imageType = vk::ImageType::e2D;
        imageInfo.extent.width = extent.width;
        imageInfo.extent.height = extent.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = vk::ImageTiling::eOptimal;
        imageInfo.usage = usage;
        image = device.createImageUnique(imageInfo);
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
        vk::ImageViewCreateInfo viewInfo{ {}, *image, vk::ImageViewType::e2D, format };
        viewInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        view = device.createImageViewUnique(viewInfo);
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
    using vkBU = vk::BufferUsageFlagBits;
    using vkMP = vk::MemoryPropertyFlagBits;

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
        auto usage = vkBU::eAccelerationStructureStorageKHR | vkBU::eShaderDeviceAddress;
        buffer.create(device, size, usage);
        buffer.bindMemory(physicalDevice, vkMP::eDeviceLocal);
    }

    void create()
    {
        vk::AccelerationStructureCreateInfoKHR createInfo;
        createInfo.buffer = *buffer.buffer;
        createInfo.size = size;
        createInfo.type = type;
        handle = device.createAccelerationStructureKHRUnique(createInfo);
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
    vk::UniqueCommandPool commandPool;

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

        //createDescSets();

        //pipeline = descSets->createRayTracingPipeline(*shaderManager, 1);
        //shaderManager->initShaderBindingTable(*pipeline, 1, 1, 1);
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
            { {}, { vkDUMS::eWarning | vkDUMS::eError },
            { vkDUMT::eGeneral | vkDUMT::ePerformance | vkDUMT::eValidation },
            &debugUtilsMessengerCallback });
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
            { vk::CommandPoolCreateFlagBits::eResetCommandBuffer, graphicsFamily });

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
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
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
        vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary;
        vk::UniqueCommandBuffer commandBuffer = std::move(
            device->allocateCommandBuffersUnique({ *commandPool, level, 1 }).front());
        commandBuffer->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        return commandBuffer;
    }

    void submitCommandBuffer(vk::CommandBuffer& commandBuffer)
    {
        commandBuffer.end();

        vk::UniqueFence fence = device->createFenceUnique({});
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(commandBuffer);
        graphicsQueue.submit(submitInfo, fence.get());
        auto res = device->waitForFences(fence.get(), true, UINT64_MAX);
        assert(res == vk::Result::eSuccess);
    }

    void createMeshBuffers()
    {
        using vkBU = vk::BufferUsageFlagBits;
        using vkMP = vk::MemoryPropertyFlagBits;
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
    }

    void createBottomLevelAS()
    {
        vk::AccelerationStructureGeometryTrianglesDataKHR triangleData;
        triangleData.vertexFormat = vk::Format::eR32G32B32Sfloat;
        triangleData.vertexData = vertexBuffer.deviceAddress;
        triangleData.vertexStride = sizeof(Vertex);
        triangleData.maxVertex = vertices.size();
        triangleData.indexType = vk::IndexType::eUint32;
        triangleData.indexData = indexBuffer.deviceAddress;

        vk::AccelerationStructureGeometryKHR geometry;
        geometry.geometryType = vk::GeometryTypeKHR::eTriangles;
        geometry.geometry = { triangleData };
        geometry.flags = vk::GeometryFlagBitsKHR::eOpaque;

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
        asInstance.transform = transformMatrix;
        asInstance.mask = 0xFF;
        asInstance.accelerationStructureReference = bottomLevelAS.buffer.deviceAddress;
        asInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

        using vkBU = vk::BufferUsageFlagBits;
        using vkMP = vk::MemoryPropertyFlagBits;
        Buffer instancesBuffer;
        instancesBuffer.create(*device, sizeof(vk::AccelerationStructureInstanceKHR),
                               vkBU::eAccelerationStructureBuildInputReadOnlyKHR
                               | vkBU::eShaderDeviceAddress); // ? shaderDevice
        instancesBuffer.bindMemory(physicalDevice, vkMP::eHostVisible | vkMP::eHostCoherent);
        instancesBuffer.fillData(&asInstance);

        vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
        instancesData.arrayOfPointers = false;
        instancesData.data = instancesBuffer.deviceAddress;

        vk::AccelerationStructureGeometryKHR geometry;
        geometry.geometryType = vk::GeometryTypeKHR::eInstances;
        geometry.geometry = { instancesData };
        geometry.flags = vk::GeometryFlagBitsKHR::eOpaque;

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
