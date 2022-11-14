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

constexpr int WIDTH = 1024;
constexpr int HEIGHT = 1024;

vk::AccessFlags toAccessFlags(vk::ImageLayout layout)
{
	switch (layout) {
		case vk::ImageLayout::eTransferSrcOptimal:
			return vk::AccessFlagBits::eTransferRead;
		case vk::ImageLayout::eTransferDstOptimal:
			return vk::AccessFlagBits::eTransferWrite;
		default:
			return {};
	}
}

void setImageLayout(vk::CommandBuffer cmdBuf, vk::Image image,
					vk::ImageLayout oldLayout, vk::ImageLayout newLayout)
{
	auto barrier = vk::ImageMemoryBarrier()
		.setDstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
		.setSrcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
		.setImage(image)
		.setOldLayout(oldLayout)
		.setNewLayout(newLayout)
		.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 })
		.setSrcAccessMask(toAccessFlags(oldLayout))
		.setDstAccessMask(toAccessFlags(newLayout));
	cmdBuf.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
						   vk::PipelineStageFlagBits::eAllCommands,
						   {}, {}, {}, barrier);
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

	size_t fileSize = file.tellg();
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
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
		window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Pathtracing", nullptr, nullptr);

		// Create instance
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

		std::vector layers{ "VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor" };

		auto vkGetInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
		VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

		auto appInfo = vk::ApplicationInfo()
			.setApiVersion(VK_API_VERSION_1_2);

		instance = vk::createInstanceUnique({ {}, &appInfo, layers, extensions });
		VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);

		physicalDevice = instance->enumeratePhysicalDevices().front();

		// Create debug messenger
		messenger = instance->createDebugUtilsMessengerEXTUnique(
			vk::DebugUtilsMessengerCreateInfoEXT()
			.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError)
			.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
			.setPfnUserCallback(&debugUtilsMessengerCallback));

		// Create surface
		VkSurfaceKHR _surface;
		VkResult res = glfwCreateWindowSurface(VkInstance(*instance), window, nullptr, &_surface);
		if (res != VK_SUCCESS) {
			throw std::runtime_error("failed to create window surface!");
		}
		surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(_surface), { *instance });

		// Find queue family
		auto queueFamilies = physicalDevice.getQueueFamilyProperties();
		for (int i = 0; i < queueFamilies.size(); i++) {
			auto supportCompute = queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute;
			auto supportPresent = physicalDevice.getSurfaceSupportKHR(i, *surface);
			if (supportCompute && supportPresent) {
				queueFamily = i;
			}
		}

		// Create device
		float queuePriority = 1.0f;
		vk::DeviceQueueCreateInfo queueCreateInfo{ {}, queueFamily, 1, &queuePriority };

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

		// Create swapchain
		swapchain = device->createSwapchainKHRUnique(
			vk::SwapchainCreateInfoKHR()
			.setSurface(*surface)
			.setMinImageCount(3)
			.setImageFormat(vk::Format::eB8G8R8A8Unorm)
			.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear)
			.setImageExtent({ WIDTH, HEIGHT })
			.setImageArrayLayers(1)
			.setImageUsage(vk::ImageUsageFlagBits::eTransferDst)
			.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity)
			.setPresentMode(vk::PresentModeKHR::eFifo)
			.setClipped(true));

		swapchainImages = device->getSwapchainImagesKHR(*swapchain);

		// Create command pool
		commandPool = device->createCommandPoolUnique(
			vk::CommandPoolCreateInfo()
			.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer)
			.setQueueFamilyIndex(queueFamily));

		// Create descriptor pool
		std::vector<vk::DescriptorPoolSize> poolSizes{
			{vk::DescriptorType::eAccelerationStructureKHR, 1},
			{vk::DescriptorType::eStorageImage, 1},
			{vk::DescriptorType::eStorageBuffer, 3} };

		descPool = device->createDescriptorPoolUnique(
			vk::DescriptorPoolCreateInfo()
			.setPoolSizes(poolSizes)
			.setMaxSets(1)
			.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet));
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
		return device->allocateCommandBuffersUnique(
			vk::CommandBufferAllocateInfo()
			.setCommandPool(*commandPool)
			.setCommandBufferCount(static_cast<uint32_t>(count)));
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
		auto res = device->acquireNextImageKHR(*swapchain, UINT64_MAX, semaphore);
		if (res.result != vk::Result::eSuccess) {
			throw std::runtime_error("failed to acquire next image!");
		}
		return res.value;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL
		debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
									VkDebugUtilsMessageTypeFlagsEXT messageTypes,
									VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData, void* pUserData)
	{
		std::cerr << pCallbackData->pMessage << std::endl;
		return VK_FALSE;
	}

	static inline GLFWwindow* window;
	static inline vk::DynamicLoader dl;
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
	enum class Type
	{
		Scratch,
		AccelInput,
		AccelStorage,
		ShaderBindingTable,
	};

	vk::UniqueBuffer buffer;
	vk::UniqueDeviceMemory memory;
	vk::DescriptorBufferInfo bufferInfo;
	uint64_t deviceAddress = 0;
	vk::DeviceSize size;

	Buffer() = default;

	void create(Type type, vk::DeviceSize size, const void* data = nullptr)
	{
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

		buffer = Context::device->createBufferUnique({ {}, size, usage });
		this->size = size;

		// Allocate memory
		vk::MemoryRequirements requirements = Context::device->getBufferMemoryRequirements(*buffer);
		uint32_t memoryTypeIndex = Context::findMemoryType(requirements.memoryTypeBits, memoryProps);

		vk::MemoryAllocateFlagsInfo flagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };

		memory = Context::device->allocateMemoryUnique(
			vk::MemoryAllocateInfo()
			.setAllocationSize(requirements.size)
			.setMemoryTypeIndex(memoryTypeIndex)
			.setPNext(&flagsInfo));

		Context::device->bindBufferMemory(*buffer, *memory, 0);

		// Get device address
		vk::BufferDeviceAddressInfoKHR bufferDeviceAI{ *buffer };
		deviceAddress = Context::device->getBufferAddressKHR(&bufferDeviceAI);

		bufferInfo = vk::DescriptorBufferInfo{ *buffer, 0, size };

		if (data) {
			void* mapped = Context::device->mapMemory(*memory, 0, size);
			memcpy(mapped, data, size);
			Context::device->unmapMemory(*memory);
		}
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
		image = Context::device->createImageUnique(
			vk::ImageCreateInfo()
			.setImageType(vk::ImageType::e2D)
			.setExtent({ extent.width, extent.height, 1 })
			.setMipLevels(1)
			.setArrayLayers(1)
			.setFormat(format)
			.setUsage(usage));

		// Allocate memory
		vk::MemoryRequirements requirements = Context::device->getImageMemoryRequirements(*image);
		uint32_t memoryTypeIndex = Context::findMemoryType(requirements.memoryTypeBits,
														   vk::MemoryPropertyFlagBits::eDeviceLocal);
		memory = Context::device->allocateMemoryUnique({ requirements.size, memoryTypeIndex });
		Context::device->bindImageMemory(*image, *memory, 0);

		// Create image view
		view = Context::device->createImageViewUnique(
			vk::ImageViewCreateInfo()
			.setImage(*image)
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(format)
			.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }));

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

	void create(vk::AccelerationStructureGeometryKHR geometry, uint32_t primitiveCount,
				vk::AccelerationStructureTypeKHR type)
	{
		auto buildGeometryInfo = vk::AccelerationStructureBuildGeometryInfoKHR()
			.setType(type)
			.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
			.setGeometries(geometry);

		// Create buffer
		vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = Context::device->getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
		vk::DeviceSize size = buildSizesInfo.accelerationStructureSize;
		buffer.create(Buffer::Type::AccelStorage, size);

		// Create accel
		accel = Context::device->createAccelerationStructureKHRUnique(
			vk::AccelerationStructureCreateInfoKHR()
			.setBuffer(*buffer.buffer)
			.setSize(size)
			.setType(type));

		// Build
		Buffer scratchBuffer;
		scratchBuffer.create(Buffer::Type::Scratch, size);
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
	void run()
	{
		outputImage.create({ WIDTH, HEIGHT }, vk::Format::eB8G8R8A8Unorm, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst);
		loadFromFile(vertices, indices, faces);

		vertexBuffer.create(Buffer::Type::AccelInput, sizeof(Vertex) * vertices.size(), vertices.data());
		indexBuffer.create(Buffer::Type::AccelInput, sizeof(uint32_t) * indices.size(), indices.data());
		faceBuffer.create(Buffer::Type::AccelInput, sizeof(Face) * faces.size(), faces.data());

		auto triangleData = vk::AccelerationStructureGeometryTrianglesDataKHR()
			.setVertexFormat(vk::Format::eR32G32B32Sfloat)
			.setVertexData(vertexBuffer.deviceAddress)
			.setVertexStride(sizeof(Vertex))
			.setMaxVertex(static_cast<uint32_t>(vertices.size()))
			.setIndexType(vk::IndexType::eUint32)
			.setIndexData(indexBuffer.deviceAddress);

		auto triangleGeometry = vk::AccelerationStructureGeometryKHR()
			.setGeometryType(vk::GeometryTypeKHR::eTriangles)
			.setGeometry({ triangleData })
			.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

		auto primitiveCount = static_cast<uint32_t>(indices.size() / 3);
		bottomAccel.create(triangleGeometry, primitiveCount, vk::AccelerationStructureTypeKHR::eBottomLevel);

		vk::TransformMatrixKHR transformMatrix = std::array{
			std::array{1.0f, 0.0f, 0.0f, 0.0f },
			std::array{0.0f, 1.0f, 0.0f, 0.0f },
			std::array{0.0f, 0.0f, 1.0f, 0.0f } };

		auto asInstance = vk::AccelerationStructureInstanceKHR()
			.setTransform(transformMatrix)
			.setMask(0xFF)
			.setAccelerationStructureReference(bottomAccel.buffer.deviceAddress)
			.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

		Buffer instancesBuffer;
		instancesBuffer.create(Buffer::Type::AccelInput, sizeof(vk::AccelerationStructureInstanceKHR), &asInstance);

		auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR()
			.setArrayOfPointers(false)
			.setData(instancesBuffer.deviceAddress);

		auto instanceGeometry = vk::AccelerationStructureGeometryKHR()
			.setGeometryType(vk::GeometryTypeKHR::eInstances)
			.setGeometry({ instancesData })
			.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

		topAccel.create(instanceGeometry, 1, vk::AccelerationStructureTypeKHR::eTopLevel);


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

		// create ray tracing pipeline
		using vkDT = vk::DescriptorType;
		using vkSS = vk::ShaderStageFlagBits;
		bindings.push_back({ 0, vkDT::eAccelerationStructureKHR, 1, vkSS::eRaygenKHR }); // Binding = 0 : TLAS
		bindings.push_back({ 1, vkDT::eStorageImage, 1, vkSS::eRaygenKHR });             // Binding = 1 : Storage image
		bindings.push_back({ 2, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 2 : Vertices
		bindings.push_back({ 3, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 3 : Indices
		bindings.push_back({ 4, vkDT::eStorageBuffer, 1, vkSS::eClosestHitKHR });        // Binding = 4 : Faces

		descSetLayout = Context::device->createDescriptorSetLayoutUnique({ {}, bindings });

		auto pushRange = vk::PushConstantRange()
			.setOffset(0)
			.setSize(sizeof(PushConstants))
			.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);
		pipelineLayout = Context::device->createPipelineLayoutUnique({ {}, *descSetLayout, pushRange });

		// Create pipeline
		auto res = Context::device->createRayTracingPipelineKHRUnique(
			nullptr, nullptr,
			vk::RayTracingPipelineCreateInfoKHR()
			.setStages(shaderStages)
			.setGroups(shaderGroups)
			.setMaxPipelineRayRecursionDepth(4)
			.setLayout(*pipelineLayout));
		if (res.result != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create ray tracing pipeline.");
		}
		pipeline = std::move(res.value);


		// Get Ray Tracing Properties
		using vkRTP = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR;
		vkRTP rtProperties = Context::physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vkRTP>().get<vkRTP>();

		// Calculate SBT size
		uint32_t handleSize = rtProperties.shaderGroupHandleSize;
		uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
		uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
		uint32_t sbtSize = groupCount * handleSizeAligned;

		// Get shader group handles
		std::vector<uint8_t> handleStorage(sbtSize);
		if (Context::device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to get ray tracing shader group handles.");
		}

		raygenSBT.create(Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 0 * handleSizeAligned);
		missSBT.create(Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 1 * handleSizeAligned);
		hitSBT.create(Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 2 * handleSizeAligned);

		uint32_t stride = rtProperties.shaderGroupHandleAlignment;
		uint32_t size = rtProperties.shaderGroupHandleAlignment;
		raygenRegion = vk::StridedDeviceAddressRegionKHR{ raygenSBT.deviceAddress, stride, size };
		missRegion = vk::StridedDeviceAddressRegionKHR{ missSBT.deviceAddress, stride, size };
		hitRegion = vk::StridedDeviceAddressRegionKHR{ hitSBT.deviceAddress, stride, size };

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

		drawCommandBuffers = Context::allocateCommandBuffers(Context::swapchainImages.size());

		imageAvailableSemaphores.resize(maxFramesInFlight);
		renderFinishedSemaphores.resize(maxFramesInFlight);
		inFlightFences.resize(maxFramesInFlight);

		for (size_t i = 0; i < maxFramesInFlight; i++) {
			imageAvailableSemaphores[i] = Context::device->createSemaphoreUnique({});
			renderFinishedSemaphores[i] = Context::device->createSemaphoreUnique({});
			inFlightFences[i] = Context::device->createFence({ vk::FenceCreateFlagBits::eSignaled });
		}

		while (!glfwWindowShouldClose(Context::window)) {
			glfwPollEvents();

			if (Context::device->waitForFences(inFlightFences[currentFrame], true, UINT64_MAX) != vk::Result::eSuccess) {
				throw std::runtime_error("Failed to wait for fences");
			}
			Context::device->resetFences(inFlightFences[currentFrame]);

			uint32_t imageIndex = Context::acquireNextImage(*imageAvailableSemaphores[currentFrame]);

			auto commandBuffer = *drawCommandBuffers[imageIndex];
			auto swapchainImage = Context::swapchainImages[imageIndex];
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

			// Submit draw command
			vk::PipelineStageFlags waitStage{ vk::PipelineStageFlagBits::eRayTracingShaderKHR };
			Context::queue.submit(
				vk::SubmitInfo()
				.setWaitSemaphores(*imageAvailableSemaphores[currentFrame])
				.setWaitDstStageMask(waitStage)
				.setCommandBuffers(*drawCommandBuffers[imageIndex])
				.setSignalSemaphores(*renderFinishedSemaphores[currentFrame]),
				inFlightFences[currentFrame]);

			// Present image
			auto presentInfo = vk::PresentInfoKHR()
				.setWaitSemaphores(*renderFinishedSemaphores[currentFrame])
				.setSwapchains(*Context::swapchain)
				.setImageIndices(imageIndex);
			if (Context::queue.presentKHR(presentInfo) != vk::Result::eSuccess) {
				throw std::runtime_error("Failed to present");
			}

			currentFrame = (currentFrame + 1) % maxFramesInFlight;
			pushConstants.frame++;
		}
		Context::device->waitIdle();
		for (size_t i = 0; i < maxFramesInFlight; i++) {
			Context::device->destroyFence(inFlightFences[i]);
		}
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

	vk::UniqueShaderModule createShaderModule(const std::string& filename)
	{
		const std::vector<char> code = readFile(filename);
		return Context::device->createShaderModuleUnique({ {}, code.size(), reinterpret_cast<const uint32_t*>(code.data()) });
	}

	void copyImage(vk::CommandBuffer& cmdBuf, vk::Image& srcImage, vk::Image& dstImage)
	{
		auto copyRegion = vk::ImageCopy()
			.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 })
			.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 })
			.setExtent({ WIDTH, HEIGHT, 1 });
		cmdBuf.copyImage(srcImage, vk::ImageLayout::eTransferSrcOptimal,
						 dstImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);
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
		glfwDestroyWindow(Context::window);
		glfwTerminate();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
