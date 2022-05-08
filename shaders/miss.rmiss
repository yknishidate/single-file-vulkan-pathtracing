#version 460
#extension GL_EXT_ray_tracing : enable

struct HitPayload
{
    vec3 position;
    vec3 normal;
    vec3 emission;
    vec3 brdf;
    bool done;
};

layout(location = 0) rayPayloadInEXT HitPayload payLoad;

void main()
{
    payLoad.emission = vec3(0.7, 0.6, 0.5);
    payLoad.done = true;
}
