
#include <string>
#include <fstream>
#include <iostream>
#include <functional>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

static constexpr int WIDTH = 1024;
static constexpr int HEIGHT = 1024;

struct Vertex {
    float position[3];
};

struct Face {
    float diffuse[3];
    float emission[3];
};

void loadFromFile(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<Face>& faces) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "../assets/CornellBox-Original.obj", "../assets")) {
        throw std::runtime_error(warn + err);
    }

    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};
            vertex.position[0] = attrib.vertices[3 * index.vertex_index + 0];
            vertex.position[1] = -attrib.vertices[3 * index.vertex_index + 1];
            vertex.position[2] = attrib.vertices[3 * index.vertex_index + 2];
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

std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

struct Context {
    Context() {
        // Create window
        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Pathtracing", nullptr, nullptr);

        // Prepase extensions and layers
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

        std::vector layers{"VK_LAYER_KHRONOS_validation"};

        auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        // Create instance
        vk::ApplicationInfo appInfo;
        appInfo.setApiVersion(VK_API_VERSION_1_4);

        vk::InstanceCreateInfo instanceInfo;
        instanceInfo.setPApplicationInfo(&appInfo);
        instanceInfo.setPEnabledLayerNames(layers);
        instanceInfo.setPEnabledExtensionNames(extensions);
        instance = vk::createInstanceUnique(instanceInfo);
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

        // Pick first gpu
        physicalDevice = instance->enumeratePhysicalDevices().front();

        // Create debug messenger
        vk::DebugUtilsMessengerCreateInfoEXT messengerInfo;
        messengerInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        messengerInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        messengerInfo.setPfnUserCallback(&debugUtilsMessengerCallback);
        messenger = instance->createDebugUtilsMessengerEXTUnique(messengerInfo);

        // Create surface
        VkSurfaceKHR _surface;
        VkResult res = glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), {*instance});

        // Find queue family
        std::vector queueFamilies = physicalDevice.getQueueFamilyProperties();
        for (int i = 0; i < queueFamilies.size(); i++) {
            auto supportCompute = queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute;
            auto supportPresent = physicalDevice.getSurfaceSupportKHR(i, *surface);
            if (supportCompute && supportPresent) {
                queueFamilyIndex = i;
            }
        }

        // Create device
        const float queuePriority = 1.0f;
        vk::DeviceQueueCreateInfo queueCreateInfo;
        queueCreateInfo.setQueueFamilyIndex(queueFamilyIndex);
        queueCreateInfo.setQueuePriorities(queuePriority);

        const std::vector deviceExtensions{
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

        if (!checkDeviceExtensionSupport(deviceExtensions)) {
            throw std::runtime_error("Some required extensions are not supported");
        }

        vk::DeviceCreateInfo deviceInfo;
        deviceInfo.setQueueCreateInfos(queueCreateInfo);
        deviceInfo.setPEnabledExtensionNames(deviceExtensions);

        vk::PhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{true};
        vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{true};
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructureFeatures{true};
        vk::StructureChain createInfoChain{
            deviceInfo,
            bufferDeviceAddressFeatures,
            rayTracingPipelineFeatures,
            accelerationStructureFeatures,
        };

        device = physicalDevice.createDeviceUnique(createInfoChain.get<vk::DeviceCreateInfo>());
        VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);

        queue = device->getQueue(queueFamilyIndex, 0);

        // Create command pool
        vk::CommandPoolCreateInfo commandPoolInfo;
        commandPoolInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
        commandPoolInfo.setQueueFamilyIndex(queueFamilyIndex);
        commandPool = device->createCommandPoolUnique(commandPoolInfo);

        // Create descriptor pool
        std::vector<vk::DescriptorPoolSize> poolSizes{
            {vk::DescriptorType::eAccelerationStructureKHR, 1},
            {vk::DescriptorType::eStorageImage, 1},
            {vk::DescriptorType::eStorageBuffer, 3},
        };

        vk::DescriptorPoolCreateInfo descPoolInfo;
        descPoolInfo.setPoolSizes(poolSizes);
        descPoolInfo.setMaxSets(1);
        descPoolInfo.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);
        descPool = device->createDescriptorPoolUnique(descPoolInfo);
    }

    bool checkDeviceExtensionSupport(const std::vector<const char*>& requiredExtensions) const {
        std::vector<vk::ExtensionProperties> availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
        std::vector<std::string> requiredExtensionNames(requiredExtensions.begin(), requiredExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensionNames.erase(std::remove(requiredExtensionNames.begin(), requiredExtensionNames.end(), extension.extensionName),
                                         requiredExtensionNames.end());
        }

        if (requiredExtensionNames.empty()) {
            std::cout << "All required extensions are supported by the device." << std::endl;
            return true;
        } else {
            std::cout << "The following required extensions are not supported by the device:" << std::endl;
            for (const auto& name : requiredExtensionNames) {
                std::cout << "\t" << name << std::endl;
            }
            return false;
        }
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("failed to find suitable memory type");
    }

    void oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func) const {
        vk::CommandBufferAllocateInfo commandBufferInfo;
        commandBufferInfo.setCommandPool(*commandPool);
        commandBufferInfo.setCommandBufferCount(1);

        vk::UniqueCommandBuffer commandBuffer = std::move(device->allocateCommandBuffersUnique(commandBufferInfo).front());
        commandBuffer->begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        func(*commandBuffer);
        commandBuffer->end();

        vk::SubmitInfo submitInfo;
        submitInfo.setCommandBuffers(*commandBuffer);
        queue.submit(submitInfo);
        queue.waitIdle();
    }

    vk::UniqueDescriptorSet allocateDescSet(vk::DescriptorSetLayout descSetLayout) {
        vk::DescriptorSetAllocateInfo descSetInfo;
        descSetInfo.setDescriptorPool(*descPool);
        descSetInfo.setSetLayouts(descSetLayout);
        return std::move(device->allocateDescriptorSetsUnique(descSetInfo).front());
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL 
        debugUtilsMessengerCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                    vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
                                    vk::DebugUtilsMessengerCallbackDataEXT const* pCallbackData,
                                    void* pUserData) {
        std::cerr << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    GLFWwindow* window;
    vk::detail::DynamicLoader dl;
    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT messenger;
    vk::UniqueSurfaceKHR surface;
    vk::UniqueDevice device;
    vk::PhysicalDevice physicalDevice;
    uint32_t queueFamilyIndex;
    vk::Queue queue;
    vk::UniqueCommandPool commandPool;
    vk::UniqueDescriptorPool descPool;
};

struct Buffer {
    enum class Type {
        Scratch,
        AccelInput,
        AccelStorage,
        ShaderBindingTable,
    };

    Buffer() = default;
    Buffer(const Context& context, Type type, vk::DeviceSize size, const void* data = nullptr) {
        vk::BufferUsageFlags usage;
        vk::MemoryPropertyFlags memoryProps;
        using Usage = vk::BufferUsageFlagBits;
        using Memory = vk::MemoryPropertyFlagBits;
        if (type == Type::AccelInput) {
            usage = Usage::eAccelerationStructureBuildInputReadOnlyKHR | Usage::eStorageBuffer | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eHostVisible | Memory::eHostCoherent;
        } else if (type == Type::Scratch) {
            usage = Usage::eStorageBuffer | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eDeviceLocal;
        } else if (type == Type::AccelStorage) {
            usage = Usage::eAccelerationStructureStorageKHR | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eDeviceLocal;
        } else if (type == Type::ShaderBindingTable) {
            usage = Usage::eShaderBindingTableKHR | Usage::eShaderDeviceAddress;
            memoryProps = Memory::eHostVisible | Memory::eHostCoherent;
        }

        buffer = context.device->createBufferUnique({{}, size, usage});

        // Allocate memory
        vk::MemoryRequirements requirements = context.device->getBufferMemoryRequirements(*buffer);
        uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, memoryProps);

        vk::MemoryAllocateFlagsInfo flagsInfo{vk::MemoryAllocateFlagBits::eDeviceAddress};

        vk::MemoryAllocateInfo memoryInfo;
        memoryInfo.setAllocationSize(requirements.size);
        memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
        memoryInfo.setPNext(&flagsInfo);
        memory = context.device->allocateMemoryUnique(memoryInfo);

        context.device->bindBufferMemory(*buffer, *memory, 0);

        // Get device address
        vk::BufferDeviceAddressInfoKHR bufferDeviceAI{*buffer};
        deviceAddress = context.device->getBufferAddressKHR(&bufferDeviceAI);

        descBufferInfo.setBuffer(*buffer);
        descBufferInfo.setOffset(0);
        descBufferInfo.setRange(size);

        if (data) {
            void* mapped = context.device->mapMemory(*memory, 0, size);
            memcpy(mapped, data, size);
            context.device->unmapMemory(*memory);
        }
    }

    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorBufferInfo descBufferInfo;
    uint64_t deviceAddress = 0;
};

struct Image {
    Image() = default;
    Image(const Context& context, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage) {
        // Create image
        vk::ImageCreateInfo imageInfo;
        imageInfo.setImageType(vk::ImageType::e2D);
        imageInfo.setExtent({extent.width, extent.height, 1});
        imageInfo.setMipLevels(1);
        imageInfo.setArrayLayers(1);
        imageInfo.setFormat(format);
        imageInfo.setUsage(usage);
        image = context.device->createImageUnique(imageInfo);

        // Allocate memory
        vk::MemoryRequirements requirements = context.device->getImageMemoryRequirements(*image);
        uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        vk::MemoryAllocateInfo memoryInfo;
        memoryInfo.setAllocationSize(requirements.size);
        memoryInfo.setMemoryTypeIndex(memoryTypeIndex);
        memory = context.device->allocateMemoryUnique(memoryInfo);

        // Bind memory and image
        context.device->bindImageMemory(*image, *memory, 0);

        // Create image view
        vk::ImageViewCreateInfo imageViewInfo;
        imageViewInfo.setImage(*image);
        imageViewInfo.setViewType(vk::ImageViewType::e2D);
        imageViewInfo.setFormat(format);
        imageViewInfo.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        view = context.device->createImageViewUnique(imageViewInfo);

        // Set image info
        descImageInfo.setImageView(*view);
        descImageInfo.setImageLayout(vk::ImageLayout::eGeneral);
        context.oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {  //
            setImageLayout(commandBuffer, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
        });
    }

    static vk::AccessFlags toAccessFlags(vk::ImageLayout layout) {
        switch (layout) {
            case vk::ImageLayout::eTransferSrcOptimal:
                return vk::AccessFlagBits::eTransferRead;
            case vk::ImageLayout::eTransferDstOptimal:
                return vk::AccessFlagBits::eTransferWrite;
            default:
                return {};
        }
    }

    static void setImageLayout(vk::CommandBuffer commandBuffer, vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        vk::ImageMemoryBarrier barrier;
        barrier.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
        barrier.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED);
        barrier.setImage(image);
        barrier.setOldLayout(oldLayout);
        barrier.setNewLayout(newLayout);
        barrier.setSubresourceRange({vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1});
        barrier.setSrcAccessMask(toAccessFlags(oldLayout));
        barrier.setDstAccessMask(toAccessFlags(newLayout));
        commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,  //
                                      vk::PipelineStageFlagBits::eAllCommands,  //
                                      {}, {}, {}, barrier);
    }

    static void copyImage(vk::CommandBuffer commandBuffer, vk::Image srcImage, vk::Image dstImage) {
        vk::ImageCopy copyRegion;
        copyRegion.setSrcSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
        copyRegion.setDstSubresource({vk::ImageAspectFlagBits::eColor, 0, 0, 1});
        copyRegion.setExtent({WIDTH, HEIGHT, 1});
        commandBuffer.copyImage(srcImage, vk::ImageLayout::eTransferSrcOptimal, dstImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);
    }

    vk::UniqueImage image;
    vk::UniqueImageView view;
    vk::UniqueDeviceMemory memory;
    vk::DescriptorImageInfo descImageInfo;
};

struct Accel {
    Accel() = default;
    Accel(const Context& context, vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type) {
        vk::AccelerationStructureBuildGeometryInfoKHR buildGeometryInfo;
        buildGeometryInfo.setType(type);
        buildGeometryInfo.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace);
        buildGeometryInfo.setGeometries(geometry);

        // Create buffer
        vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = context.device->getAccelerationStructureBuildSizesKHR(  //
            vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
        vk::DeviceSize size = buildSizesInfo.accelerationStructureSize;
        buffer = Buffer{context, Buffer::Type::AccelStorage, size};

        // Create accel
        vk::AccelerationStructureCreateInfoKHR accelInfo;
        accelInfo.setBuffer(*buffer.buffer);
        accelInfo.setSize(size);
        accelInfo.setType(type);
        accel = context.device->createAccelerationStructureKHRUnique(accelInfo);

        // Build
        Buffer scratchBuffer{context, Buffer::Type::Scratch, buildSizesInfo.buildScratchSize};
        buildGeometryInfo.setScratchData(scratchBuffer.deviceAddress);
        buildGeometryInfo.setDstAccelerationStructure(*accel);

        context.oneTimeSubmit([&](vk::CommandBuffer commandBuffer) {  //
            vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo;
            buildRangeInfo.setPrimitiveCount(primitiveCount);
            buildRangeInfo.setFirstVertex(0);
            buildRangeInfo.setPrimitiveOffset(0);
            buildRangeInfo.setTransformOffset(0);
            commandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
        });

        descAccelInfo.setAccelerationStructures(*accel);
    }

    Buffer buffer;
    vk::UniqueAccelerationStructureKHR accel;
    vk::WriteDescriptorSetAccelerationStructureKHR descAccelInfo;
};

int main() {
    Context context;

    vk::SwapchainCreateInfoKHR swapchainInfo;
    swapchainInfo.setSurface(*context.surface);
    swapchainInfo.setMinImageCount(3);
    swapchainInfo.setImageFormat(vk::Format::eB8G8R8A8Unorm);
    swapchainInfo.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear);
    swapchainInfo.setImageExtent({WIDTH, HEIGHT});
    swapchainInfo.setImageArrayLayers(1);
    swapchainInfo.setImageUsage(vk::ImageUsageFlagBits::eTransferDst);
    swapchainInfo.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
    swapchainInfo.setPresentMode(vk::PresentModeKHR::eFifo);
    swapchainInfo.setClipped(true);
    swapchainInfo.setQueueFamilyIndices(context.queueFamilyIndex);
    vk::UniqueSwapchainKHR swapchain = context.device->createSwapchainKHRUnique(swapchainInfo);

    std::vector<vk::Image> swapchainImages = context.device->getSwapchainImagesKHR(*swapchain);

    vk::CommandBufferAllocateInfo commandBufferInfo;
    commandBufferInfo.setCommandPool(*context.commandPool);
    commandBufferInfo.setCommandBufferCount(static_cast<uint32_t>(swapchainImages.size()));
    std::vector<vk::UniqueCommandBuffer> commandBuffers = context.device->allocateCommandBuffersUnique(commandBufferInfo);

    Image outputImage{context,
                      {WIDTH, HEIGHT},
                      vk::Format::eB8G8R8A8Unorm,
                      vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst};

    // Load mesh
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Face> faces;
    loadFromFile(vertices, indices, faces);

    Buffer vertexBuffer{context, Buffer::Type::AccelInput, sizeof(Vertex) * vertices.size(), vertices.data()};
    Buffer indexBuffer{context, Buffer::Type::AccelInput, sizeof(uint32_t) * indices.size(), indices.data()};
    Buffer faceBuffer{context, Buffer::Type::AccelInput, sizeof(Face) * faces.size(), faces.data()};

    // Create bottom level accel struct
    vk::AccelerationStructureGeometryTrianglesDataKHR triangleData;
    triangleData.setVertexFormat(vk::Format::eR32G32B32Sfloat);
    triangleData.setVertexData(vertexBuffer.deviceAddress);
    triangleData.setVertexStride(sizeof(Vertex));
    triangleData.setMaxVertex(static_cast<uint32_t>(vertices.size()));
    triangleData.setIndexType(vk::IndexType::eUint32);
    triangleData.setIndexData(indexBuffer.deviceAddress);

    vk::AccelerationStructureGeometryKHR triangleGeometry;
    triangleGeometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
    triangleGeometry.setGeometry({triangleData});
    triangleGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    const auto primitiveCount = static_cast<uint32_t>(indices.size() / 3);

    Accel bottomAccel{context, triangleGeometry, primitiveCount, vk::AccelerationStructureTypeKHR::eBottomLevel};

    // Create top level accel struct
    vk::TransformMatrixKHR transformMatrix = std::array{
        std::array{1.0f, 0.0f, 0.0f, 0.0f},
        std::array{0.0f, 1.0f, 0.0f, 0.0f},
        std::array{0.0f, 0.0f, 1.0f, 0.0f},
    };

    vk::AccelerationStructureInstanceKHR accelInstance;
    accelInstance.setTransform(transformMatrix);
    accelInstance.setMask(0xFF);
    accelInstance.setAccelerationStructureReference(bottomAccel.buffer.deviceAddress);
    accelInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

    Buffer instancesBuffer{context, Buffer::Type::AccelInput, sizeof(vk::AccelerationStructureInstanceKHR), &accelInstance};

    vk::AccelerationStructureGeometryInstancesDataKHR instancesData;
    instancesData.setArrayOfPointers(false);
    instancesData.setData(instancesBuffer.deviceAddress);

    vk::AccelerationStructureGeometryKHR instanceGeometry;
    instanceGeometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
    instanceGeometry.setGeometry({instancesData});
    instanceGeometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

    Accel topAccel{context, instanceGeometry, 1, vk::AccelerationStructureTypeKHR::eTopLevel};

    // Load shaders
    const std::vector<char> raygenCode = readFile("../shaders/raygen.rgen.spv");
    const std::vector<char> missCode = readFile("../shaders/miss.rmiss.spv");
    const std::vector<char> chitCode = readFile("../shaders/closesthit.rchit.spv");

    std::vector<vk::UniqueShaderModule> shaderModules(3);
    shaderModules[0] = context.device->createShaderModuleUnique({{}, raygenCode.size(), reinterpret_cast<const uint32_t*>(raygenCode.data())});
    shaderModules[1] = context.device->createShaderModuleUnique({{}, missCode.size(), reinterpret_cast<const uint32_t*>(missCode.data())});
    shaderModules[2] = context.device->createShaderModuleUnique({{}, chitCode.size(), reinterpret_cast<const uint32_t*>(chitCode.data())});

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages(3);
    shaderStages[0] = {{}, vk::ShaderStageFlagBits::eRaygenKHR, *shaderModules[0], "main"};
    shaderStages[1] = {{}, vk::ShaderStageFlagBits::eMissKHR, *shaderModules[1], "main"};
    shaderStages[2] = {{}, vk::ShaderStageFlagBits::eClosestHitKHR, *shaderModules[2], "main"};

    std::vector<vk::RayTracingShaderGroupCreateInfoKHR> shaderGroups(3);
    shaderGroups[0] = {vk::RayTracingShaderGroupTypeKHR::eGeneral, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    shaderGroups[1] = {vk::RayTracingShaderGroupTypeKHR::eGeneral, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};
    shaderGroups[2] = {vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, 2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR};

    // create ray tracing pipeline
    std::vector<vk::DescriptorSetLayoutBinding> bindings{
        {0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR},  // Binding = 0 : TLAS
        {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR},              // Binding = 1 : Storage image
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 2 : Vertices
        {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 3 : Indices
        {4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 4 : Faces
    };

    // Create desc set layout
    vk::DescriptorSetLayoutCreateInfo descSetLayoutInfo;
    descSetLayoutInfo.setBindings(bindings);
    vk::UniqueDescriptorSetLayout descSetLayout = context.device->createDescriptorSetLayoutUnique(descSetLayoutInfo);

    // Create pipeline layout
    vk::PushConstantRange pushRange;
    pushRange.setOffset(0);
    pushRange.setSize(sizeof(int));
    pushRange.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo;
    pipelineLayoutInfo.setSetLayouts(*descSetLayout);
    pipelineLayoutInfo.setPushConstantRanges(pushRange);
    vk::UniquePipelineLayout pipelineLayout = context.device->createPipelineLayoutUnique(pipelineLayoutInfo);

    // Create pipeline
    vk::RayTracingPipelineCreateInfoKHR rtPipelineInfo;
    rtPipelineInfo.setStages(shaderStages);
    rtPipelineInfo.setGroups(shaderGroups);
    rtPipelineInfo.setMaxPipelineRayRecursionDepth(4);
    rtPipelineInfo.setLayout(*pipelineLayout);

    auto result = context.device->createRayTracingPipelineKHRUnique(nullptr, nullptr, rtPipelineInfo);
    if (result.result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to create ray tracing pipeline.");
    }

    vk::UniquePipeline pipeline = std::move(result.value);

    // Get ray tracing properties
    auto properties = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();
    auto rtProperties = properties.get<vk::PhysicalDeviceRayTracingPipelinePropertiesKHR>();

    // Calculate shader binding table (SBT) size
    uint32_t handleSize = rtProperties.shaderGroupHandleSize;
    uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
    uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
    uint32_t sbtSize = groupCount * handleSizeAligned;

    // Get shader group handles
    std::vector<uint8_t> handleStorage(sbtSize);
    if (context.device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess) {
        throw std::runtime_error("failed to get ray tracing shader group handles.");
    }

    // Create SBT
    Buffer raygenSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 0 * handleSizeAligned};
    Buffer missSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 1 * handleSizeAligned};
    Buffer hitSBT{context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 2 * handleSizeAligned};

    uint32_t stride = rtProperties.shaderGroupHandleAlignment;
    uint32_t size = rtProperties.shaderGroupHandleAlignment;

    vk::StridedDeviceAddressRegionKHR raygenRegion{raygenSBT.deviceAddress, stride, size};
    vk::StridedDeviceAddressRegionKHR missRegion{missSBT.deviceAddress, stride, size};
    vk::StridedDeviceAddressRegionKHR hitRegion{hitSBT.deviceAddress, stride, size};

    // Create desc set
    vk::UniqueDescriptorSet descSet = context.allocateDescSet(*descSetLayout);
    std::vector<vk::WriteDescriptorSet> writes(bindings.size());
    for (int i = 0; i < bindings.size(); i++) {
        writes[i].setDstSet(*descSet);
        writes[i].setDescriptorType(bindings[i].descriptorType);
        writes[i].setDescriptorCount(bindings[i].descriptorCount);
        writes[i].setDstBinding(bindings[i].binding);
    }
    writes[0].setPNext(&topAccel.descAccelInfo);
    writes[1].setImageInfo(outputImage.descImageInfo);
    writes[2].setBufferInfo(vertexBuffer.descBufferInfo);
    writes[3].setBufferInfo(indexBuffer.descBufferInfo);
    writes[4].setBufferInfo(faceBuffer.descBufferInfo);
    context.device->updateDescriptorSets(writes, nullptr);

    // Main loop
    uint32_t imageIndex = 0;
    int frame = 0;
    vk::UniqueSemaphore imageAcquiredSemaphore = context.device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
    while (!glfwWindowShouldClose(context.window)) {
        glfwPollEvents();

        // Acquire next image
        imageIndex = context.device->acquireNextImageKHR(*swapchain, UINT64_MAX, *imageAcquiredSemaphore).value;

        // Record commands
        vk::CommandBuffer commandBuffer = *commandBuffers[imageIndex];
        commandBuffer.begin(vk::CommandBufferBeginInfo());
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0, *descSet, nullptr);
        commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, sizeof(int), &frame);
        commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);

        vk::Image srcImage = *outputImage.image;
        vk::Image dstImage = swapchainImages[imageIndex];
        Image::setImageLayout(commandBuffer, srcImage, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
        Image::setImageLayout(commandBuffer, dstImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        Image::copyImage(commandBuffer, srcImage, dstImage);
        Image::setImageLayout(commandBuffer, srcImage, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
        Image::setImageLayout(commandBuffer, dstImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);

        commandBuffer.end();

        // Submit
        context.queue.submit(vk::SubmitInfo().setCommandBuffers(commandBuffer));

        // Present image
        vk::PresentInfoKHR presentInfo;
        presentInfo.setSwapchains(*swapchain);
        presentInfo.setImageIndices(imageIndex);
        presentInfo.setWaitSemaphores(*imageAcquiredSemaphore);
        auto result = context.queue.presentKHR(presentInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present.");
        }
        context.queue.waitIdle();
        frame++;
    }

    context.device->waitIdle();
    glfwDestroyWindow(context.window);
    glfwTerminate();
}
