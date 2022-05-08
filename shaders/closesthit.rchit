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

layout(binding = 2, set = 0) buffer Vertices{float v[];} vertices;
layout(binding = 3, set = 0) buffer Indices{uint i[];} indices;
layout(binding = 4, set = 0) buffer Faces{float f[];} faces;

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec3 attribs;

const highp float M_PI = 3.14159265358979323846;

struct Vertex
{
    vec3 pos;
};

struct Face
{
    vec3 diffuse;
    vec3 emission;
};

Vertex unpack(uint index)
{
    uint stride = 3;
    uint offset = index * stride;
    Vertex v;
    v.pos = vec3(vertices.v[offset +  0], vertices.v[offset +  1], vertices.v[offset + 2]);
    return v;
}

Face unpackFace(uint index)
{
    uint stride = 6;
    uint offset = index * stride;
    Face f;
    f.diffuse = vec3(faces.f[offset +  0], faces.f[offset +  1], faces.f[offset + 2]);
    f.emission = vec3(faces.f[offset +  3], faces.f[offset +  4], faces.f[offset + 5]);
    return f;
}

vec3 calcNormal(Vertex v0, Vertex v1, Vertex v2)
{
    vec3 e01 = v1.pos - v0.pos;
    vec3 e02 = v2.pos - v0.pos;
    return -normalize(cross(e01, e02));
}

void main()
{
    Vertex v0 = unpack(indices.i[3 * gl_PrimitiveID + 0]);
    Vertex v1 = unpack(indices.i[3 * gl_PrimitiveID + 1]);
    Vertex v2 = unpack(indices.i[3 * gl_PrimitiveID + 2]);

    const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 pos = v0.pos * barycentricCoords.x + v1.pos * barycentricCoords.y + v2.pos * barycentricCoords.z;
    vec3 normal = calcNormal(v0, v1, v2);

    Face face = unpackFace(gl_PrimitiveID);
    payload.brdf = face.diffuse / M_PI;
    payload.emission = face.emission * 2.0;
    payload.position = pos;
    payload.normal = normal;
}
