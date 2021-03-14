
#include <set>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

#include <GLFW/glfw3.h> // include after vulkan.hpp
#include <glm/glm.hpp>

using vkBU = vk::BufferUsageFlagBits;
using vkMP = vk::MemoryPropertyFlagBits;

// ----------------------------------------------------------------------------------------------------------
// Globals
// ----------------------------------------------------------------------------------------------------------
constexpr int WIDTH = 1024;
constexpr int HEIGHT = 1024;
constexpr int MAX_FRAMES_IN_FLIGHT = 2;
#ifdef _DEBUG
constexpr bool enableValidationLayers = true;
#else
constexpr bool enableValidationLayers = false;
#endif
std::vector<const char*> validationLayers;
const std::string ASSET_PATH = "assets/CornellBox.obj";

// ----------------------------------------------------------------------------------------------------------
// Functuins
// ----------------------------------------------------------------------------------------------------------
VKAPI_ATTR VkBool32 VKAPI_CALL
debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                            VkDebugUtilsMessageTypeFlagsEXT messageTypes,
                            VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData, void* /*pUserData*/)
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

uint32_t findMemoryType(const vk::PhysicalDevice physicalDevice, const uint32_t typeFilter,
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
        if (format.format == vk::Format::eB8G8R8A8Unorm && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }
    throw std::runtime_error("found no suitable surface format");
}

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
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
                                    capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    actualExtent.height = std::clamp(actualExtent.height,
                                     capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
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

    void fillData(void* data)
    {
        if (!mapped) {
            mapped = device.mapMemory(*memory, 0, size);
        }
        memcpy(mapped, data, static_cast<size_t>(size));
    }

    vk::DescriptorBufferInfo createDescInfo()
    {
        return vk::DescriptorBufferInfo{ *buffer, 0, size };
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

    void create(vk::Device device, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage)
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
        vk::MemoryRequirements requirements = device.getImageMemoryRequirements(*image);
        uint32_t memoryTypeIndex = findMemoryType(physicalDevice, requirements.memoryTypeBits,
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

    vk::DescriptorImageInfo createDescInfo()
    {
        return vk::DescriptorImageInfo{ {}, *view, imageLayout };
    }
};

enum class Material : int
{
    White, Red, Green, Light
};

struct Vertex
{
    glm::vec3 pos;
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
        handle = device.createAccelerationStructureKHRUnique(
            vk::AccelerationStructureCreateInfoKHR{}
            .setBuffer(*buffer.buffer)
            .setSize(size)
            .setType(type));
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

// ----------------------------------------------------------------------------------------------------------
// Application
// ----------------------------------------------------------------------------------------------------------
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
    std::vector<vk::Fence> imagesInFlight;
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
        createStorageImage();
        loadMesh();
        createMeshBuffers();
        createUniformBuffer();
        createBottomLevelAS();
        createTopLevelAS();
        loadShaders();
        createRayTracingPipeLine();
        createShaderBindingTable();
        createDescriptorSets();
        buildCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
            updateUniformBuffer();
            if (uniformData.frame % 10 == 0) {
                std::cout << "frame: " << uniformData.frame << std::endl;
            }
        }
        device->waitIdle();
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
        auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
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
        VkResult res = glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface);
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
        vk::DeviceCreateInfo createInfo{ {}, queueCreateInfos, validationLayers, requiredExtensions, &deviceFeatures };

        // Create structure chain
        vk::StructureChain<vk::DeviceCreateInfo,
            vk::PhysicalDeviceBufferDeviceAddressFeatures,
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR,
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR>
            createInfoChain{ createInfo, {true}, {true}, {true} };

        device = physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

        graphicsQueue = device->getQueue(graphicsFamily, 0);
        presentQueue = device->getQueue(presentFamily, 0);

        commandPool = device->createCommandPoolUnique(
            vk::CommandPoolCreateInfo{}
            .setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
            .setQueueFamilyIndex(graphicsFamily));
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
        vk::SwapchainCreateInfoKHR createInfo{};
        createInfo.setSurface(*surface);
        createInfo.setMinImageCount(imageCount);
        createInfo.setImageFormat(format);
        createInfo.setImageColorSpace(surfaceFormat.colorSpace);
        createInfo.setImageExtent(extent);
        createInfo.setImageArrayLayers(1);
        createInfo.setImageUsage(vkIUF::eColorAttachment | vkIUF::eTransferDst);
        createInfo.setPreTransform(capabilities.currentTransform);
        createInfo.setPresentMode(presentMode);
        createInfo.setClipped(true);
        if (graphicsFamily != presentFamily) {
            uint32_t queueFamilyIndices[] = { graphicsFamily, presentFamily };
            createInfo.setImageSharingMode(vk::SharingMode::eConcurrent);
            createInfo.setQueueFamilyIndexCount(2);
            createInfo.setPQueueFamilyIndices(queueFamilyIndices);
        }
        swapChain = device->createSwapchainKHRUnique(createInfo);
        swapChainImages = device->getSwapchainImagesKHR(*swapChain);
    }

    void createStorageImage()
    {
        using vkIUF = vk::ImageUsageFlagBits;
        storageImage.create(*device, extent, format, vkIUF::eStorage | vkIUF::eTransferSrc);
        storageImage.bindMemory(physicalDevice);
        storageImage.createImageView();

        // Set image layout
        storageImage.imageLayout = vk::ImageLayout::eGeneral;
        vk::UniqueCommandBuffer cmdBuf = createCommandBuffer();
        transitionImageLayout(*cmdBuf, *storageImage.image,
                              vk::ImageLayout::eUndefined, storageImage.imageLayout);
        submitCommandBuffer(*cmdBuf);
    }

    vk::UniqueCommandBuffer createCommandBuffer()
    {
        vk::CommandBufferAllocateInfo allocInfo{ *commandPool, vk::CommandBufferLevel::ePrimary, 1 };
        vk::UniqueCommandBuffer cmdBuf = std::move(device->allocateCommandBuffersUnique(allocInfo).front());
        cmdBuf->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        return cmdBuf;
    }

    void submitCommandBuffer(vk::CommandBuffer& cmdBuf)
    {
        cmdBuf.end();

        vk::UniqueFence fence = device->createFenceUnique({});
        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(cmdBuf);
        graphicsQueue.submit(submitInfo, *fence);

        vk::Result res = device->waitForFences(*fence, true, UINT64_MAX);
        assert(res == vk::Result::eSuccess);
    }

    void loadMesh()
    {
        std::ifstream file(ASSET_PATH);
        std::string line;
        Material currentMaterial = Material::White;
        while (std::getline(file, line)) {
            std::vector<std::string> list = split(line, ' ');
            if (list[0] == "v") {
                vertices.push_back(Vertex{ glm::vec3{ stof(list[1]), -stof(list[2]), stof(list[3]) } });
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
        vk::BufferUsageFlags usage{ vkBU::eAccelerationStructureBuildInputReadOnlyKHR
            | vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress };
        vk::MemoryPropertyFlags properties{ vkMP::eHostVisible | vkMP::eHostCoherent };

        uint64_t vertexBufferSize = vertices.size() * sizeof(Vertex);
        vertexBuffer.create(*device, vertexBufferSize, usage);
        vertexBuffer.bindMemory(physicalDevice, properties);
        vertexBuffer.fillData(vertices.data());

        uint64_t indexBufferSize = indices.size() * sizeof(uint32_t);
        indexBuffer.create(*device, indexBufferSize, usage);
        indexBuffer.bindMemory(physicalDevice, properties);
        indexBuffer.fillData(indices.data());

        uint64_t primitiveBufferSize = primitiveMaterials.size() * sizeof(int);
        primitiveBuffer.create(*device, primitiveBufferSize, vkBU::eStorageBuffer | vkBU::eShaderDeviceAddress);
        primitiveBuffer.bindMemory(physicalDevice, properties);
        primitiveBuffer.fillData(primitiveMaterials.data());
    }

    void createUniformBuffer()
    {
        uniformBuffer.create(*device, sizeof(UniformData), vkBU::eUniformBuffer);
        uniformBuffer.bindMemory(physicalDevice, vkMP::eHostVisible | vkMP::eHostCoherent);
        updateUniformBuffer();
    }

    void updateUniformBuffer()
    {
        uniformData.frame += 1;
        uniformBuffer.fillData(&uniformData);
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
        vk::UniqueCommandBuffer cmdBuf = createCommandBuffer();
        bottomLevelAS.build(*cmdBuf);
        submitCommandBuffer(*cmdBuf);
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
                               vkBU::eAccelerationStructureBuildInputReadOnlyKHR | vkBU::eShaderDeviceAddress);
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
        vk::UniqueCommandBuffer cmdBuf = createCommandBuffer();
        topLevelAS.build(*cmdBuf);
        submitCommandBuffer(*cmdBuf);
    }

    void loadShaders()
    {
        const uint32_t raygenIndex = 0;
        const uint32_t missIndex = 1;
        const uint32_t ClosestHitIndex = 2;

        shaderModules.push_back(createShaderModule("shaders/raygen.rgen.spv"));
        shaderStages.push_back({ {}, vk::ShaderStageFlagBits::eRaygenKHR, *shaderModules.back(), "main" });
        shaderGroups.push_back({ vk::RayTracingShaderGroupTypeKHR::eGeneral,
                               raygenIndex, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR });

        shaderModules.push_back(createShaderModule("shaders/miss.rmiss.spv"));
        shaderStages.push_back({ {}, vk::ShaderStageFlagBits::eMissKHR, *shaderModules.back(), "main" });
        shaderGroups.push_back({ vk::RayTracingShaderGroupTypeKHR::eGeneral,
                               missIndex, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR });

        shaderModules.push_back(createShaderModule("shaders/closesthit.rchit.spv"));
        shaderStages.push_back({ {}, vk::ShaderStageFlagBits::eClosestHitKHR, *shaderModules.back(), "main" });
        shaderGroups.push_back({ vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup,
                               VK_SHADER_UNUSED_KHR, ClosestHitIndex, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR });
    }

    vk::UniqueShaderModule createShaderModule(const std::string& filename)
    {
        const std::vector<char> code = readFile(filename);
        return device->createShaderModuleUnique({ {}, code.size(), reinterpret_cast<const uint32_t*>(code.data()) });
    }

    void createRayTracingPipeLine()
    {
        std::vector<vk::DescriptorSetLayoutBinding> bindings;
        using vkDT = vk::DescriptorType;
        using vkSS = vk::ShaderStageFlagBits;
        bindings.push_back({ 0, vkDT::eAccelerationStructureKHR, 1, vkSS::eRaygenKHR }); // Binding = 0 : TLAS
        bindings.push_back({ 1, vkDT::eStorageImage, 1, vkSS::eRaygenKHR });             // Binding = 1 : Storage image
        bindings.push_back({ 2, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 2 : Vertices
        bindings.push_back({ 3, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 3 : Indices
        bindings.push_back({ 4, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 4 : Materials
        bindings.push_back({ 5, vkDT::eUniformBuffer, 1, vkSS::eRaygenKHR });            // Binding = 5 : Uniform data

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
        raygenSBT.fillData(shaderHandleStorage.data() + 0 * handleSizeAligned);

        missSBT.create(*device, handleSize, usage);
        missSBT.bindMemory(physicalDevice, properties);
        missSBT.fillData(shaderHandleStorage.data() + 1 * handleSizeAligned);

        hitSBT.create(*device, handleSize, usage);
        hitSBT.bindMemory(physicalDevice, properties);
        hitSBT.fillData(shaderHandleStorage.data() + 2 * handleSizeAligned);
    }

    void createDescriptorSets()
    {
        createDescPool();
        descSet = std::move(device->allocateDescriptorSetsUnique({ *descPool, *descSetLayout }).front());
        updateDescSet();
    }

    void createDescPool()
    {
        std::vector<vk::DescriptorPoolSize> poolSizes{
            {vk::DescriptorType::eAccelerationStructureKHR, 1},
            {vk::DescriptorType::eStorageImage, 1},
            {vk::DescriptorType::eStorageBuffer, 3},
            {vk::DescriptorType::eUniformBuffer, 1} };

        descPool = device->createDescriptorPoolUnique(
            vk::DescriptorPoolCreateInfo{}
            .setPoolSizes(poolSizes)
            .setMaxSets(1)
            .setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet));
    }

    void updateDescSet()
    {
        using vkDT = vk::DescriptorType;
        std::vector<vk::WriteDescriptorSet> writeDescSets;

        vk::WriteDescriptorSetAccelerationStructureKHR asInfo{ *topLevelAS.handle };
        vk::WriteDescriptorSet asWrite{};
        asWrite.setDstSet(*descSet);
        asWrite.setDescriptorType(vkDT::eAccelerationStructureKHR);
        asWrite.setDescriptorCount(1);
        asWrite.setDstBinding(0);
        asWrite.setPNext(&asInfo);
        writeDescSets.push_back(asWrite);

        writeDescSets.push_back(createImageWrite(storageImage.createDescInfo(), vkDT::eStorageImage, 1));
        writeDescSets.push_back(createBufferWrite(vertexBuffer.createDescInfo(), vkDT::eStorageBuffer, 2));
        writeDescSets.push_back(createBufferWrite(indexBuffer.createDescInfo(), vkDT::eStorageBuffer, 3));
        writeDescSets.push_back(createBufferWrite(primitiveBuffer.createDescInfo(), vkDT::eStorageBuffer, 4));
        writeDescSets.push_back(createBufferWrite(uniformBuffer.createDescInfo(), vkDT::eUniformBuffer, 5));
        device->updateDescriptorSets(writeDescSets, nullptr);
    }

    vk::WriteDescriptorSet createImageWrite(vk::DescriptorImageInfo imageInfo, vk::DescriptorType type,
                                            uint32_t binding)
    {
        vk::WriteDescriptorSet imageWrite{};
        imageWrite.setDstSet(*descSet);
        imageWrite.setDescriptorType(type);
        imageWrite.setDescriptorCount(1);
        imageWrite.setDstBinding(binding);
        imageWrite.setImageInfo(imageInfo);
        return imageWrite;
    }

    vk::WriteDescriptorSet createBufferWrite(vk::DescriptorBufferInfo bufferInfo, vk::DescriptorType type,
                                             uint32_t binding)
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
            copyStorageImage(*drawCommandBuffers[i], swapChainImages[i]);
            drawCommandBuffers[i]->end();
        }
    }

    void traceRays(vk::CommandBuffer& cmdBuf)
    {
        vk::StridedDeviceAddressRegionKHR raygenRegion = createAddressRegion(raygenSBT.deviceAddress);
        vk::StridedDeviceAddressRegionKHR missRegion = createAddressRegion(missSBT.deviceAddress);
        vk::StridedDeviceAddressRegionKHR hitRegion = createAddressRegion(hitSBT.deviceAddress);
        cmdBuf.traceRaysKHR(raygenRegion, missRegion, hitRegion, {},
                            storageImage.extent.width, storageImage.extent.height, 1);
    }

    vk::StridedDeviceAddressRegionKHR createAddressRegion(vk::DeviceAddress deviceAddress)
    {
        vk::StridedDeviceAddressRegionKHR region{};
        region.setDeviceAddress(deviceAddress);
        region.setStride(rtProperties.shaderGroupHandleAlignment);
        region.setSize(rtProperties.shaderGroupHandleAlignment);
        return region;
    }

    void copyStorageImage(vk::CommandBuffer& cmdBuf, vk::Image& swapChainImage)
    {
        using vkIL = vk::ImageLayout;
        transitionImageLayout(cmdBuf, *storageImage.image, vkIL::eGeneral, vkIL::eTransferSrcOptimal);
        transitionImageLayout(cmdBuf, swapChainImage, vkIL::eUndefined, vkIL::eTransferDstOptimal);

        vk::ImageCopy copyRegion{};
        copyRegion.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 });
        copyRegion.setExtent({ storageImage.extent.width, storageImage.extent.height, 1 });
        cmdBuf.copyImage(*storageImage.image, vkIL::eTransferSrcOptimal,
                         swapChainImage, vkIL::eTransferDstOptimal, copyRegion);

        transitionImageLayout(cmdBuf, *storageImage.image, vkIL::eTransferSrcOptimal, vkIL::eGeneral);
        transitionImageLayout(cmdBuf, swapChainImage, vkIL::eTransferDstOptimal, vkIL::ePresentSrcKHR);
    }

    void allocateDrawCommandBuffers()
    {
        drawCommandBuffers = device->allocateCommandBuffersUnique(
            vk::CommandBufferAllocateInfo{}
            .setCommandPool(*commandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(swapChainImages.size()));
    }

    void createSyncObjects()
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size());

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            imageAvailableSemaphores[i] = device->createSemaphoreUnique({});
            renderFinishedSemaphores[i] = device->createSemaphoreUnique({});
            inFlightFences[i] = device->createFence({ vk::FenceCreateFlagBits::eSignaled });
        }
    }

    void drawFrame()
    {
        device->waitForFences(inFlightFences[currentFrame], true, UINT64_MAX);

        uint32_t imageIndex = acquireNextImageIndex();

        // Wait for fence
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) {
            device->waitForFences(imagesInFlight[imageIndex], true, UINT64_MAX);
        }
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];
        device->resetFences(inFlightFences[currentFrame]);

        // Submit draw command
        vk::PipelineStageFlags waitStage{ vk::PipelineStageFlagBits::eRayTracingShaderKHR };
        graphicsQueue.submit(
            vk::SubmitInfo{}
            .setWaitSemaphores(*imageAvailableSemaphores[currentFrame])
            .setWaitDstStageMask(waitStage)
            .setCommandBuffers(*drawCommandBuffers[imageIndex])
            .setSignalSemaphores(*renderFinishedSemaphores[currentFrame]),
            inFlightFences[currentFrame]);

        // Present image
        graphicsQueue.presentKHR(
            vk::PresentInfoKHR{}
            .setWaitSemaphores(*renderFinishedSemaphores[currentFrame])
            .setSwapchains(*swapChain)
            .setImageIndices(imageIndex));

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    uint32_t acquireNextImageIndex()
    {
        auto res = device->acquireNextImageKHR(*swapChain, UINT64_MAX, *imageAvailableSemaphores[currentFrame]);
        if (res.result == vk::Result::eSuccess) {
            return res.value;
        }
        throw std::runtime_error("failed to acquire next image!");
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
