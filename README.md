# Single-File Vulkan Pathtracing

![image](https://user-images.githubusercontent.com/30839669/167279645-c56a70ac-8941-4a2b-ba1c-05a5d03c3d27.png)

# Environment

- C++20
- Vulkan SDK 1.3.250.1 or later
- GPU / Driver that support Vulkan Ray Tracing
  - NVIDIA Vulkan beta driver Windows 531.83, Linux 525.47.22 or later

## Setup

### Manual

See [Vulkan Tutorial / Development environment](https://vulkan-tutorial.com/Development_environment)

### CMake

```
git clone --recursive https://github.com/yknishidate/single-file-vulkan-pathtracing
cd single-file-vulkan-pathtracing
cmake . -B build
```

# References

- [NVIDIA Vulkan Ray Tracing Tutorial](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/)
- [Vulkan-Hpp](https://github.com/KhronosGroup/Vulkan-Hpp)
- [Vulkan Tutorial](https://vulkan-tutorial.com/)
- [SaschaWillems/Vulkan](https://github.com/SaschaWillems/Vulkan)
- [vk_raytrace](https://github.com/nvpro-samples/vk_raytrace)
