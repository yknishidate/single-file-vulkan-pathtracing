%VULKAN_SDK%/Bin/glslc.exe raygen.rgen -o raygen.rgen.spv --target-env=vulkan1.3
%VULKAN_SDK%/Bin/glslc.exe closesthit.rchit -o closesthit.rchit.spv --target-env=vulkan1.3
%VULKAN_SDK%/Bin/glslc.exe miss.rmiss -o miss.rmiss.spv --target-env=vulkan1.3