#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_ray_tracing_position_fetch : enable
#include "common.glsl"

layout(binding = 2, set = 0) buffer Vertices{float vertices[];};
layout(binding = 3, set = 0) buffer Indices{uint indices[];};
layout(binding = 4, set = 0) buffer Faces{float faces[];};

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec3 attribs;

struct Vertex
{
    vec3 pos;
};

struct Face
{
    vec3 diffuse;
    vec3 emission;
};

Face unpackFace(uint index)
{
    uint stride = 6;
    uint offset = index * stride;
    Face f;
    f.diffuse = vec3(faces[offset +  0], faces[offset +  1], faces[offset + 2]);
    f.emission = vec3(faces[offset +  3], faces[offset +  4], faces[offset + 5]);
    return f;
}

vec3 calcNormal(vec3 vertPos0, vec3 vertPos1, vec3 vertPos2)
{
    return -normalize(cross(vertPos1 - vertPos0, vertPos2 - vertPos0));
}

void main()
{
    vec3 vertPos0 = gl_HitTriangleVertexPositionsEXT[0];
    vec3 vertPos1 = gl_HitTriangleVertexPositionsEXT[1];
    vec3 vertPos2 = gl_HitTriangleVertexPositionsEXT[2];

    const vec3 barycentricCoords = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 pos = vertPos0 * barycentricCoords.x + vertPos1 * barycentricCoords.y + vertPos2 * barycentricCoords.z;
    vec3 normal = calcNormal(vertPos0, vertPos1, vertPos2);

    Face face = unpackFace(gl_PrimitiveID);
    payload.brdf = face.diffuse / M_PI;
    payload.emission = face.emission * 2.0;
    payload.position = pos;
    payload.normal = normal;
}
