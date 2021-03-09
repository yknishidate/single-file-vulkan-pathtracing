
#include <set>
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


// Debug callback
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
    switch (oldLayout) {
        case vk::ImageLayout::eUndefined:
            imageMemoryBarrier.srcAccessMask = {};
            break;
        case vk::ImageLayout::ePreinitialized:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
            break;
        case vk::ImageLayout::eTransferDstOptimal:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eShaderRead;
            break;
        default:
            break;
    }

    // Target layouts (new)
    switch (newLayout) {
        case vk::ImageLayout::eTransferDstOptimal:
            imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask
                | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            if (imageMemoryBarrier.srcAccessMask == vk::AccessFlags{}) {
                imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite
                    | vk::AccessFlagBits::eTransferWrite;
            }
            imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
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


struct QueueFamilyIndices
{
    uint32_t graphicsFamily = -1;
    uint32_t presentFamily = -1;

    bool isComplete()
    {
        return graphicsFamily != -1 && presentFamily != -1;
    }
};

struct Buffer
{
    vk::Device device;
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DeviceSize size;
    vk::BufferUsageFlags usage;
    uint64_t deviceAddress;
    void* mapped = nullptr;

    void createBuffer(vk::Device device, vk::DeviceSize size, vk::BufferUsageFlags usage)
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
                                              vk::MemoryPropertyFlagBits::eDeviceLocal);
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

    void createImage(vk::Device device, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage)
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

// Globals
constexpr int WIDTH = 800;
constexpr int HEIGHT = 600;
#ifdef _DEBUG
const bool enableValidationLayers = true;
#else
const bool enableValidationLayers = false;
#endif
std::vector<const char*> validationLayers;


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

    Buffer vertexBuffer;
    Buffer indexBuffer;


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
        createBuffers();
        buildAccelStruct();

        //loadShaders();

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
        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags{
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError };

        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags{
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
            | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance
            | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation };

        messenger = instance->createDebugUtilsMessengerEXTUnique(
            { {}, severityFlags, messageTypeFlags, &debugUtilsMessengerCallback });
    }

    void createSurface()
    {
        VkSurfaceKHR _surface;
        if (glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface) != VK_SUCCESS) {
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
            vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamily, 1, &queuePriority);
            queueCreateInfos.push_back(queueCreateInfo);
        }

        // Set physical device features
        vk::PhysicalDeviceFeatures deviceFeatures;
        deviceFeatures.fillModeNonSolid = true;
        deviceFeatures.samplerAnisotropy = true;

        vk::PhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures;
        indexingFeatures.runtimeDescriptorArray = true;

        vk::DeviceCreateInfo createInfo{
            {}, queueCreateInfos, validationLayers, requiredExtensions, &deviceFeatures };

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
        vk::SurfaceCapabilitiesKHR capabilities = physicalDevice.getSurfaceCapabilitiesKHR(*surface);
        std::vector<vk::SurfaceFormatKHR> formats = physicalDevice.getSurfaceFormatsKHR(*surface);
        std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(*surface);

        // Choose swapchain settings
        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(formats);
        format = surfaceFormat.format;
        presentMode = chooseSwapPresentMode(presentModes);
        extent = chooseSwapExtent(capabilities);
        uint32_t imageCount = capabilities.minImageCount + 1;

        // Create swap chain
        vk::SwapchainCreateInfoKHR createInfo;
        createInfo.surface = *surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
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

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
        const std::vector<vk::SurfaceFormatKHR>& formats)
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

        vk::Extent2D actualExtent = extent;
        actualExtent.width = std::min(std::max(actualExtent.width, capabilities.minImageExtent.width),
                                      capabilities.maxImageExtent.width);
        actualExtent.height = std::min(std::max(actualExtent.height, capabilities.minImageExtent.height),
                                       capabilities.maxImageExtent.height);
        return actualExtent;
    }

    void createStorageImage()
    {
        storageImage.createImage(*device, extent, format,
                                 vk::ImageUsageFlagBits::eStorage
                                 | vk::ImageUsageFlagBits::eTransferSrc);
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
        vk::UniqueCommandBuffer commandBuffer = std::move(
            device->allocateCommandBuffersUnique(
                { *commandPool, vk::CommandBufferLevel::ePrimary, 1 }).front());
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

    void createBuffers()
    {
        std::vector<Vertex> vertices{ { {1.0f, 1.0f, 0.0f} },
                                      { {-1.0f, 1.0f, 0.0f} },
                                      { {0.0f, -1.0f, 0.0f} } };
        std::vector<uint32_t> indices{ 0, 1, 2 };

        using vkbu = vk::BufferUsageFlagBits;
        using vkmp = vk::MemoryPropertyFlagBits;
        vk::BufferUsageFlags usage{ vkbu::eAccelerationStructureBuildInputReadOnlyKHR
            | vkbu::eStorageBuffer | vkbu::eTransferDst };
        //| vkbu::eStorageBuffer | vkbu::eShaderDeviceAddress | vkbu::eTransferDst };
        vk::MemoryPropertyFlags properties{ vkmp::eDeviceLocal };

        uint64_t vertexBufferSize = vertices.size() * sizeof(Vertex);
        vertexBuffer.createBuffer(*device, vertexBufferSize, usage);
        vertexBuffer.bindMemory(physicalDevice, properties);

        uint64_t indexBufferSize = indices.size() * sizeof(uint32_t);
        indexBuffer.createBuffer(*device, indexBufferSize, usage);
        indexBuffer.bindMemory(physicalDevice, properties);
    }

    void buildAccelStruct()
    {



        std::cout << "builded accel struct\n";
    }

};


int main()
{
    Application app;
    app.run();
}
