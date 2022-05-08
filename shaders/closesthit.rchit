#version 460
#extension GL_EXT_ray_tracing : enable

struct HitPayload
{
    vec3 position;
    vec3 normal;
    vec3 emittion;
    vec3 brdf;
    bool done;
};

layout(binding = 3, set = 0) buffer Vertices{float v[];} vertices;
layout(binding = 4, set = 0) buffer Indices{uint i[];} indices;
layout(binding = 5, set = 0) buffer Materials{float m[];} materials;

layout(location = 0) rayPayloadInEXT HitPayload payLoad;
hitAttributeEXT vec3 attribs;

const highp float M_PI = 3.14159265358979323846;

struct Vertex
{
    vec3 pos;
};

struct Material
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

Material unpackMaterial(uint index)
{
    uint stride = 6;
    uint offset = index * stride;
    Material m;
    m.diffuse = vec3(materials.m[offset +  0], materials.m[offset +  1], materials.m[offset + 2]);
    m.emission = vec3(materials.m[offset +  3], materials.m[offset +  4], materials.m[offset + 5]);
    return m;
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

    Material mat = unpackMaterial(gl_PrimitiveID);
    payLoad.brdf = mat.diffuse / M_PI;
    payLoad.position = pos;
    payLoad.normal = normal;
    payLoad.emittion = mat.emission;

    // int materialID = materials.m[gl_PrimitiveID];
    // if(materialID == LIGHT){
    //     payLoad.emittion = vec3(3.0);
    //     payLoad.done = true;
    // }else{
    //     payLoad.brdf = colors[materialID] / M_PI;
    //     payLoad.position = pos;
    //     payLoad.normal = normal;
    // }
}
