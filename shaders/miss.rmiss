#version 460
#extension GL_EXT_ray_tracing : enable

struct HitPayload
{
    vec3 contribution;
    vec3 position;
    vec3 normal;
    bool done;
};

layout(location = 0) rayPayloadInEXT HitPayload payLoad;

void main()
{
    payLoad.contribution = vec3(0.0);
    payLoad.done = true;
}
