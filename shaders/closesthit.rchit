#version 460
#extension GL_EXT_ray_tracing : enable

struct HitPayload
{
    vec3 contribution;
    vec3 position;
    vec3 normal;
    bool done;
};

layout(binding = 2, set = 0) buffer Vertices{float v[];} vertices;
layout(binding = 3, set = 0) buffer Indices{uint i[];} indices;
layout(binding = 4, set = 0) buffer Material{int m[];} materials;

layout(location = 0) rayPayloadInEXT HitPayload payLoad;
hitAttributeEXT vec3 attribs;

struct Vertex
{
    vec3 pos;
};

Vertex unpack(uint index)
{
    uint vertexSize = 3;
    uint offset = index * vertexSize;
    Vertex v;
    v.pos = vec3(vertices.v[offset +  0], vertices.v[offset +  1], vertices.v[offset + 2]);
	return v;
}

vec3 calcNormal(Vertex v0, Vertex v1, Vertex v2)
{
    vec3 e01 = v1.pos - v0.pos;
    vec3 e02 = v2.pos - v0.pos;
    return -normalize(cross(e01, e02));
}

int WHITE  = 0;
int RED    = 1;
int GREEN  = 2;
int LIGHT  = 3;

void main()
{
    vec3 colors[] = { vec3(0.9, 0.9, 0.9), vec3(1.0, 0.5, 0.5), vec3(0.5, 1.0, 0.5), vec3(1.0, 1.0, 1.0) };

    Vertex v0 = unpack(indices.i[3 * gl_PrimitiveID + 0]);
    Vertex v1 = unpack(indices.i[3 * gl_PrimitiveID + 1]);
    Vertex v2 = unpack(indices.i[3 * gl_PrimitiveID + 2]);

    const vec3 barycentricCoords = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 pos = v0.pos * barycentricCoords.x + v1.pos * barycentricCoords.y + v2.pos * barycentricCoords.z;
    vec3 normal = calcNormal(v0, v1, v2);

    int materialID = materials.m[gl_PrimitiveID];
    if(materialID == LIGHT){
        payLoad.contribution *= vec3(5.0);
        payLoad.done = true;
    }else{
        payLoad.contribution *= colors[materialID];
        payLoad.position = pos;
        payLoad.normal = normal;
    }
}
