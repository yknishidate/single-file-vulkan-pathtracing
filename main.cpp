
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
	Context()
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

	uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const
	{
		vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();
		for (uint32_t i = 0; i != memProperties.memoryTypeCount; ++i) {
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
				return i;
			}
		}
		throw std::runtime_error("failed to find suitable memory type");
	}

	void oneTimeSubmit(const std::function<void(vk::CommandBuffer)>& func)
	{
		vk::UniqueCommandBuffer commandBuffer = std::move(device->allocateCommandBuffersUnique(
			vk::CommandBufferAllocateInfo()
			.setCommandPool(*commandPool)
			.setCommandBufferCount(1))
			.front());
		commandBuffer->begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
		func(*commandBuffer);
		commandBuffer->end();

		queue.submit(vk::SubmitInfo().setCommandBuffers(*commandBuffer));
		queue.waitIdle();
	}

	vk::UniqueDescriptorSet allocateDescSet(vk::DescriptorSetLayout descSetLayout)
	{
		return std::move(device->allocateDescriptorSetsUnique({ *descPool, descSetLayout }).front());
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL
		debugUtilsMessengerCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
									VkDebugUtilsMessageTypeFlagsEXT messageTypes,
									VkDebugUtilsMessengerCallbackDataEXT const* pCallbackData, void* pUserData)
	{
		std::cerr << pCallbackData->pMessage << std::endl;
		return VK_FALSE;
	}

	GLFWwindow* window;
	vk::DynamicLoader dl;
	vk::UniqueInstance instance;
	vk::UniqueDebugUtilsMessengerEXT messenger;
	vk::UniqueSurfaceKHR surface;
	vk::UniqueDevice device;
	vk::PhysicalDevice physicalDevice;
	uint32_t queueFamily;
	vk::Queue queue;
	vk::UniqueCommandPool commandPool;
	vk::UniqueDescriptorPool descPool;
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

	Buffer() = default;

	Buffer(Context& context, Type type, vk::DeviceSize size, const void* data = nullptr)
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

		buffer = context.device->createBufferUnique({ {}, size, usage });

		// Allocate memory
		vk::MemoryRequirements requirements = context.device->getBufferMemoryRequirements(*buffer);
		uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, memoryProps);

		vk::MemoryAllocateFlagsInfo flagsInfo{ vk::MemoryAllocateFlagBits::eDeviceAddress };

		memory = context.device->allocateMemoryUnique(
			vk::MemoryAllocateInfo()
			.setAllocationSize(requirements.size)
			.setMemoryTypeIndex(memoryTypeIndex)
			.setPNext(&flagsInfo));

		context.device->bindBufferMemory(*buffer, *memory, 0);

		// Get device address
		vk::BufferDeviceAddressInfoKHR bufferDeviceAI{ *buffer };
		deviceAddress = context.device->getBufferAddressKHR(&bufferDeviceAI);

		bufferInfo = vk::DescriptorBufferInfo{ *buffer, 0, size };

		if (data) {
			void* mapped = context.device->mapMemory(*memory, 0, size);
			memcpy(mapped, data, size);
			context.device->unmapMemory(*memory);
		}
	}

	vk::UniqueBuffer buffer;
	vk::UniqueDeviceMemory memory;
	vk::DescriptorBufferInfo bufferInfo;
	uint64_t deviceAddress = 0;
};

struct Image
{
	Image(Context& context, vk::Extent2D extent, vk::Format format, vk::ImageUsageFlags usage)
	{
		// Create image
		image = context.device->createImageUnique(
			vk::ImageCreateInfo()
			.setImageType(vk::ImageType::e2D)
			.setExtent({ extent.width, extent.height, 1 })
			.setMipLevels(1)
			.setArrayLayers(1)
			.setFormat(format)
			.setUsage(usage));

		// Allocate memory
		vk::MemoryRequirements requirements = context.device->getImageMemoryRequirements(*image);
		uint32_t memoryTypeIndex = context.findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
		memory = context.device->allocateMemoryUnique({ requirements.size, memoryTypeIndex });
		context.device->bindImageMemory(*image, *memory, 0);

		// Create image view
		view = context.device->createImageViewUnique(
			vk::ImageViewCreateInfo()
			.setImage(*image)
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(format)
			.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }));

		// Set image layout
		imageInfo = { {}, *view, vk::ImageLayout::eGeneral };
		context.oneTimeSubmit(
			[&](vk::CommandBuffer cmdBuf) {
			setImageLayout(cmdBuf, *image, vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral);
		});
	}

	static vk::AccessFlags toAccessFlags(vk::ImageLayout layout)
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

	static void setImageLayout(vk::CommandBuffer commandBuffer, vk::Image image,
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
		commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
									  vk::PipelineStageFlagBits::eAllCommands,
									  {}, {}, {}, barrier);
	}

	static void copyImage(vk::CommandBuffer commandBuffer, vk::Image srcImage, vk::Image dstImage)
	{
		auto copyRegion = vk::ImageCopy()
			.setSrcSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 })
			.setDstSubresource({ vk::ImageAspectFlagBits::eColor, 0, 0, 1 })
			.setExtent({ WIDTH, HEIGHT, 1 });
		commandBuffer.copyImage(srcImage, vk::ImageLayout::eTransferSrcOptimal,
								dstImage, vk::ImageLayout::eTransferDstOptimal, copyRegion);
	}

	vk::UniqueImage image;
	vk::UniqueImageView view;
	vk::UniqueDeviceMemory memory;
	vk::DescriptorImageInfo imageInfo;
};

struct Accel
{
	Accel(Context& context, vk::AccelerationStructureGeometryKHR geometry,
		  uint32_t primitiveCount, vk::AccelerationStructureTypeKHR type)
	{
		auto buildGeometryInfo = vk::AccelerationStructureBuildGeometryInfoKHR()
			.setType(type)
			.setFlags(vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace)
			.setGeometries(geometry);

		// Create buffer
		vk::AccelerationStructureBuildSizesInfoKHR buildSizesInfo = context.device->getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice, buildGeometryInfo, primitiveCount);
		vk::DeviceSize size = buildSizesInfo.accelerationStructureSize;
		buffer = Buffer{ context, Buffer::Type::AccelStorage, size };

		// Create accel
		accel = context.device->createAccelerationStructureKHRUnique(
			vk::AccelerationStructureCreateInfoKHR()
			.setBuffer(*buffer.buffer)
			.setSize(size)
			.setType(type));

		// Build
		Buffer scratchBuffer{ context, Buffer::Type::Scratch, size };
		buildGeometryInfo.setScratchData(scratchBuffer.deviceAddress);
		buildGeometryInfo.setDstAccelerationStructure(*accel);

		vk::AccelerationStructureBuildRangeInfoKHR buildRangeInfo{ primitiveCount, 0, 0, 0 };
		context.oneTimeSubmit(
			[&](vk::CommandBuffer commandBuffer) {
			commandBuffer.buildAccelerationStructuresKHR(buildGeometryInfo, &buildRangeInfo);
		});

		accelInfo = vk::WriteDescriptorSetAccelerationStructureKHR{ *accel };
	}

	Buffer buffer;
	vk::UniqueAccelerationStructureKHR accel;
	vk::WriteDescriptorSetAccelerationStructureKHR accelInfo;
};

struct Swapchain
{
	Swapchain() = default;

	void destroy() const;

	void acquireNextImage()
	{
		const vk::Semaphore imageAcquiredSemaphore = context->device->createSemaphore(vk::SemaphoreCreateInfo());
		frameIndex = context->device->acquireNextImageKHR(*swapchain, UINT64_MAX, imageAcquiredSemaphore).value;
		context->device->destroySemaphore(imageAcquiredSemaphore);
	}

	void submitCommands(const std::function<void(vk::CommandBuffer)>& function)
	{
		vk::CommandBuffer commandBuffer = *commandBuffers[frameIndex];
		commandBuffer.begin(vk::CommandBufferBeginInfo());

		function(commandBuffer);

		commandBuffer.end();
		context->queue.submit(vk::SubmitInfo().setCommandBuffers(commandBuffer));
	}

	void copyToBackImage(vk::Image srcImage)
	{
		vk::CommandBuffer commandBuffer = *commandBuffers[frameIndex];
		vk::Image backImage = swapchainImages[frameIndex];
		Image::setImageLayout(commandBuffer, srcImage, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferSrcOptimal);
		Image::setImageLayout(commandBuffer, backImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
		Image::copyImage(commandBuffer, srcImage, backImage);
		Image::setImageLayout(commandBuffer, srcImage, vk::ImageLayout::eTransferSrcOptimal, vk::ImageLayout::eGeneral);
		Image::setImageLayout(commandBuffer, backImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR);
	}

	void present()
	{
		auto result = context->queue.presentKHR(vk::PresentInfoKHR()
												.setSwapchains(*swapchain)
												.setImageIndices(frameIndex));
		if (result != vk::Result::eSuccess) {
			throw std::runtime_error("failed to present.");
		}
	}

	Swapchain(Context& ctx)
		: context(&ctx)
	{
		swapchain = context->device->createSwapchainKHRUnique(
			vk::SwapchainCreateInfoKHR()
			.setSurface(*context->surface)
			.setMinImageCount(3)
			.setImageFormat(vk::Format::eB8G8R8A8Unorm)
			.setImageColorSpace(vk::ColorSpaceKHR::eSrgbNonlinear)
			.setImageExtent({ WIDTH, HEIGHT })
			.setImageArrayLayers(1)
			.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst)
			.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity)
			.setPresentMode(vk::PresentModeKHR::eFifo)
			.setClipped(true)
			.setQueueFamilyIndices(context->queueFamily));

		// get images
		swapchainImages = context->device->getSwapchainImagesKHR(*swapchain);

		// create image views
		for (const auto& image : swapchainImages) {
			swapchainViews.push_back(context->device->createImageViewUnique(
				vk::ImageViewCreateInfo()
				.setImage(image)
				.setViewType(vk::ImageViewType::e2D)
				.setFormat(vk::Format::eB8G8R8A8Unorm)
				.setComponents({ vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA })
				.setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 })));
		}

		// allocate command buffers
		commandBuffers = context->device->allocateCommandBuffersUnique(
			vk::CommandBufferAllocateInfo()
			.setCommandPool(*context->commandPool)
			.setCommandBufferCount(swapchainImages.size()));
	}

	Context* context = nullptr;
	vk::UniqueSwapchainKHR swapchain;
	std::vector<vk::Image> swapchainImages;
	std::vector<vk::UniqueImageView> swapchainViews;
	std::vector<vk::UniqueCommandBuffer> commandBuffers;
	uint32_t frameIndex = 0;
};

int main()
{
	try {
		Context context;
		Swapchain swapchain{ context };
		Image outputImage{ context, {WIDTH, HEIGHT}, vk::Format::eB8G8R8A8Unorm, vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst };

		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
		std::vector<Face> faces;
		loadFromFile(vertices, indices, faces);

		Buffer vertexBuffer{ context, Buffer::Type::AccelInput, sizeof(Vertex) * vertices.size(), vertices.data() };
		Buffer indexBuffer{ context, Buffer::Type::AccelInput, sizeof(uint32_t) * indices.size(), indices.data() };
		Buffer faceBuffer{ context, Buffer::Type::AccelInput, sizeof(Face) * faces.size(), faces.data() };

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

		Accel bottomAccel{ context, triangleGeometry, primitiveCount, vk::AccelerationStructureTypeKHR::eBottomLevel };

		vk::TransformMatrixKHR transformMatrix = std::array{
			std::array{1.0f, 0.0f, 0.0f, 0.0f},
			std::array{0.0f, 1.0f, 0.0f, 0.0f},
			std::array{0.0f, 0.0f, 1.0f, 0.0f} };

		auto asInstance = vk::AccelerationStructureInstanceKHR()
			.setTransform(transformMatrix)
			.setMask(0xFF)
			.setAccelerationStructureReference(bottomAccel.buffer.deviceAddress)
			.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);

		Buffer instancesBuffer{ context, Buffer::Type::AccelInput, sizeof(vk::AccelerationStructureInstanceKHR), &asInstance };

		auto instancesData = vk::AccelerationStructureGeometryInstancesDataKHR()
			.setArrayOfPointers(false)
			.setData(instancesBuffer.deviceAddress);

		auto instanceGeometry = vk::AccelerationStructureGeometryKHR()
			.setGeometryType(vk::GeometryTypeKHR::eInstances)
			.setGeometry({ instancesData })
			.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

		Accel topAccel{ context, instanceGeometry, 1, vk::AccelerationStructureTypeKHR::eTopLevel };

		const std::vector<char> raygenCode = readFile("../shaders/raygen.rgen.spv");
		const std::vector<char> missCode = readFile("../shaders/miss.rmiss.spv");
		const std::vector<char> chitCode = readFile("../shaders/closesthit.rchit.spv");
		std::array shaderModules{
			context.device->createShaderModuleUnique({{}, raygenCode.size(), reinterpret_cast<const uint32_t*>(raygenCode.data())}),
			context.device->createShaderModuleUnique({{}, missCode.size(), reinterpret_cast<const uint32_t*>(missCode.data())}),
			context.device->createShaderModuleUnique({{}, chitCode.size(), reinterpret_cast<const uint32_t*>(chitCode.data())}) };

		std::array shaderStages{
			vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eRaygenKHR, *shaderModules[0], "main"},
			vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eMissKHR, *shaderModules[1], "main"},
			vk::PipelineShaderStageCreateInfo{{}, vk::ShaderStageFlagBits::eClosestHitKHR, *shaderModules[2], "main"} };

		std::array shaderGroups{
			vk::RayTracingShaderGroupCreateInfoKHR{vk::RayTracingShaderGroupTypeKHR::eGeneral, 0, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
			vk::RayTracingShaderGroupCreateInfoKHR{vk::RayTracingShaderGroupTypeKHR::eGeneral, 1, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR},
			vk::RayTracingShaderGroupCreateInfoKHR{vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup, VK_SHADER_UNUSED_KHR, 2, VK_SHADER_UNUSED_KHR, VK_SHADER_UNUSED_KHR} };

		// create ray tracing pipeline
		std::vector<vk::DescriptorSetLayoutBinding> bindings{
			{0, vk::DescriptorType::eAccelerationStructureKHR, 1, vk::ShaderStageFlagBits::eRaygenKHR},  // Binding = 0 : TLAS
			{1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eRaygenKHR},              // Binding = 1 : Storage image
			{2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 2 : Vertices
			{3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 3 : Indices
			{4, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eClosestHitKHR},         // Binding = 4 : Faces
		};

		vk::UniqueDescriptorSetLayout descSetLayout = context.device->createDescriptorSetLayoutUnique({ {}, bindings });

		auto pushRange = vk::PushConstantRange()
			.setOffset(0)
			.setSize(sizeof(int))
			.setStageFlags(vk::ShaderStageFlagBits::eRaygenKHR);
		vk::UniquePipelineLayout pipelineLayout = context.device->createPipelineLayoutUnique({ {}, *descSetLayout, pushRange });

		// Create pipeline
		auto res = context.device->createRayTracingPipelineKHRUnique(
			nullptr, nullptr,
			vk::RayTracingPipelineCreateInfoKHR()
			.setStages(shaderStages)
			.setGroups(shaderGroups)
			.setMaxPipelineRayRecursionDepth(4)
			.setLayout(*pipelineLayout));
		if (res.result != vk::Result::eSuccess) {
			throw std::runtime_error("failed to create ray tracing pipeline.");
		}
		vk::UniquePipeline pipeline = std::move(res.value);

		// Get Ray Tracing Properties
		using vkRTP = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR;
		vkRTP rtProperties = context.physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vkRTP>().get<vkRTP>();

		// Calculate SBT size
		uint32_t handleSize = rtProperties.shaderGroupHandleSize;
		uint32_t handleSizeAligned = rtProperties.shaderGroupHandleAlignment;
		uint32_t groupCount = static_cast<uint32_t>(shaderGroups.size());
		uint32_t sbtSize = groupCount * handleSizeAligned;

		// Get shader group handles
		std::vector<uint8_t> handleStorage(sbtSize);
		if (context.device->getRayTracingShaderGroupHandlesKHR(*pipeline, 0, groupCount, sbtSize, handleStorage.data()) != vk::Result::eSuccess) {
			throw std::runtime_error("failed to get ray tracing shader group handles.");
		}

		Buffer raygenSBT{ context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 0 * handleSizeAligned };
		Buffer missSBT{ context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 1 * handleSizeAligned };
		Buffer hitSBT{ context, Buffer::Type::ShaderBindingTable, handleSize, handleStorage.data() + 2 * handleSizeAligned };

		uint32_t stride = rtProperties.shaderGroupHandleAlignment;
		uint32_t size = rtProperties.shaderGroupHandleAlignment;

		vk::StridedDeviceAddressRegionKHR raygenRegion{ raygenSBT.deviceAddress, stride, size };
		vk::StridedDeviceAddressRegionKHR missRegion{ missSBT.deviceAddress, stride, size };
		vk::StridedDeviceAddressRegionKHR hitRegion{ hitSBT.deviceAddress, stride, size };

		vk::UniqueDescriptorSet descSet = context.allocateDescSet(*descSetLayout);
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
		context.device->updateDescriptorSets(writes, nullptr);

		int frame = 0;
		while (!glfwWindowShouldClose(context.window)) {
			glfwPollEvents();

			swapchain.acquireNextImage();
			swapchain.submitCommands([&](auto commandBuffer) {
				commandBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, *pipeline);
			commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, *pipelineLayout, 0, *descSet, nullptr);
			commandBuffer.pushConstants(*pipelineLayout, vk::ShaderStageFlagBits::eRaygenKHR, 0, sizeof(int), &frame);
			commandBuffer.traceRaysKHR(raygenRegion, missRegion, hitRegion, {}, WIDTH, HEIGHT, 1);
			swapchain.copyToBackImage(*outputImage.image);
			});
			swapchain.present();
			context.queue.waitIdle();
			frame++;
		}
		context.device->waitIdle();
		glfwDestroyWindow(context.window);
		glfwTerminate();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
	}
}
