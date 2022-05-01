# Single File Vulkan Pathtracing

Minimal pathtracer using Vulkan RayTracing

![image](https://user-images.githubusercontent.com/30839669/166169596-d3a30a02-0e2a-4a37-a08b-4cba5315486c.png)

# Environment

-   Vulkan SDK 1.2.162.0 or later
-   GPU / Driver that support Vulkan Ray Tracing
-   C++17
-   GLFW

## Setup

### Manual

See [Vulkan Tutorial / Development environment](https://vulkan-tutorial.com/Development_environment)

### cmake

```
git clone --recursive https://github.com/nishidate-yuki/single-file-vulkan-pathtracing.git
cd single-file-vulkan-pathtracing
cmake . -Bbuild
```

# References

-   [NVIDIA Vulkan Ray Tracing Tutorial](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/)
-   [Vulkan-Hpp](https://github.com/KhronosGroup/Vulkan-Hpp)
-   [Vulkan Tutorial](https://vulkan-tutorial.com/)
-   [SaschaWillems/Vulkan](https://github.com/SaschaWillems/Vulkan)
-   [vk_raytrace](https://github.com/nvpro-samples/vk_raytrace)
