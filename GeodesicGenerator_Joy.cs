using System.Collections.Generic;
using UnityEngine;
using MeshUtils;
using Unity.Jobs;
using Unity.Burst;
using Unity.Collections;
using Unity.Mathematics;
public class GeodesicGenerator : MonoBehaviour
{
    public Mesh mesh;
    public Material mat;


    List<float3> vertList = new List<float3>();
    List<TriangleIndices> faces = new List<TriangleIndices>();
    [Range(2, 12)]
    public int recursionLevel;
    public bool refresh;
    float radius = 1;

    [BurstCompile]
    struct IcosphereJob : IJob
    {
        public int subdivisions;
        public NativeList<float3> vertices;
        public NativeList<TriangleIndices> faces;
        float radius;
        private static long generateKey(int i1, int i2)
        {
            long key;
            bool firstIsSmaller = i1 < i2;
            long smallerIndex = math.select(i2, i1, firstIsSmaller);
            long greaterIndex = math.select(i1, i2, firstIsSmaller);
            key = ((smallerIndex & 0xFFFF0000) << 48) + ((greaterIndex & 0xFFFF0000) << 32) + ((smallerIndex & 0x0000FFFF) << 16) + (greaterIndex & 0x0000FFFF);
            return key;
        }
        public void Execute()
        {
            float phi = (1f + math.sqrt(5f)) / 2f;

            vertices.Add(math.normalize(new float3(-1f, phi, 0f)) * radius);
            vertices.Add(math.normalize(new float3(1f, phi, 0f)) * radius);
            vertices.Add(math.normalize(new float3(-1f, -phi, 0f)) * radius);
            vertices.Add(math.normalize(new float3(1f, -phi, 0f)) * radius);

            vertices.Add(math.normalize(new float3(0f, -1f, phi)) * radius);
            vertices.Add(math.normalize(new float3(0f, 1f, phi)) * radius);
            vertices.Add(math.normalize(new float3(0f, -1f, -phi)) * radius);
            vertices.Add(math.normalize(new float3(0f, 1f, -phi)) * radius);

            vertices.Add(math.normalize(new float3(phi, 0f, -1f)) * radius);
            vertices.Add(math.normalize(new float3(phi, 0f, 1f)) * radius);
            vertices.Add(math.normalize(new float3(-phi, 0f, -1f)) * radius);
            vertices.Add(math.normalize(new float3(-phi, 0f, 1f)) * radius);

            // 5 faces around point 0
            faces.Add(new TriangleIndices(0, 11, 5));
            faces.Add(new TriangleIndices(0, 5, 1));
            faces.Add(new TriangleIndices(0, 1, 7));
            faces.Add(new TriangleIndices(0, 7, 10));
            faces.Add(new TriangleIndices(0, 10, 11));

            // 5 adjacent faces
            faces.Add(new TriangleIndices(1, 5, 9));
            faces.Add(new TriangleIndices(5, 11, 4));
            faces.Add(new TriangleIndices(11, 10, 2));
            faces.Add(new TriangleIndices(10, 7, 6));
            faces.Add(new TriangleIndices(7, 1, 8));

            // 5 faces around point 3
            faces.Add(new TriangleIndices(3, 9, 4));
            faces.Add(new TriangleIndices(3, 4, 2));
            faces.Add(new TriangleIndices(3, 2, 6));
            faces.Add(new TriangleIndices(3, 6, 8));
            faces.Add(new TriangleIndices(3, 8, 9));

            // 5 adjacent faces
            faces.Add(new TriangleIndices(4, 9, 5));
            faces.Add(new TriangleIndices(2, 4, 11));
            faces.Add(new TriangleIndices(6, 2, 10));
            faces.Add(new TriangleIndices(8, 6, 7));
            faces.Add(new TriangleIndices(9, 8, 1));
            for (int i = 0; i < subdivisions; i++)
            {
                NativeList<TriangleIndices> subdivision = new NativeList<TriangleIndices>(Allocator.Temp);
                NativeHashMap<long, int> keys;
                NativeList<Edge> edges;
                NativeHashMap<long, int> midpoints;
                subdivision.Clear();

                keys = new NativeHashMap<long,int>(1, Allocator.Temp);
                edges = new NativeList<Edge>(Allocator.Temp);
                midpoints = new NativeHashMap<long, int>(1,Allocator.Temp);

                // Get all the unique edges.
                for (int t = 0; t < faces.Length; t++)
                {

                    TriangleIndices triangle = faces[t];
                    int ia = triangle.a;
                    int ib = triangle.b;
                    int ic = triangle.c;

                    long k0 = generateKey(ia, ib);
                    long k1 = generateKey(ib, ic);
                    long k2 = generateKey(ic, ia);

                    if (keys.TryAdd(k0,0))
                    {
                        edges.Add(new Edge(k0, ia, ib));
                    }

                    if (keys.TryAdd(k1, 0))
                    {
                        
                        edges.Add(new Edge(k1, ib, ic));
                    }

                    if (keys.TryAdd(k2, 0))
                    {
                        
                        edges.Add(new Edge(k2, ic, ia));
                    }  
                }

                // Calculate all their midpoints.
                for (int t = 0; t < edges.Length; t++)
                {
                    Edge e = edges[t];
                    float3 v0 = vertices[e.i0];
                    float3 v1 = vertices[e.i1];

                    float mx = (v0.x + v1.x) * 0.5f;
                    float my = (v0.y + v1.y) * 0.5f;
                    float mz = (v0.z + v1.z) * 0.5f;
                    float3 midpoint = radius * math.normalize(new float3(mx, my, mz));

                    midpoints.TryAdd(e.key, vertices.Length);
                    vertices.Add(midpoint);
                }

                // Generate the new subdivision.
                for (int t = 0; t < faces.Length; t++)
                {

                    TriangleIndices triangle = faces[t];
                    int ia = triangle.a;
                    int ib = triangle.b;
                    int ic = triangle.c;

                    long k0 = generateKey(ia, ib);
                    long k1 = generateKey(ib, ic);
                    long k2 = generateKey(ic, ia);

                    int iab;
                    int ibc;
                    int ica;

                    midpoints.TryGetValue(k0, out iab);
                    midpoints.TryGetValue(k1, out ibc);
                    midpoints.TryGetValue(k2, out ica);

                    subdivision.Add(new TriangleIndices(ia, iab, ica));
                    subdivision.Add(new TriangleIndices(iab, ib, ibc));
                    subdivision.Add(new TriangleIndices(ica, ibc, ic));
                    subdivision.Add(new TriangleIndices(iab, ibc, ica));
                }

                faces = subdivision;

                keys.Dispose();
                edges.Dispose();
                midpoints.Dispose();
            }
        }
        public IcosphereJob(int subdivisions, float radius)
        {
            vertices = new NativeList<float3>(Allocator.Persistent);
            faces = new NativeList<TriangleIndices>(Allocator.Persistent);
            //vertices.Dispose();
            //faces.Dispose();
            this.subdivisions = subdivisions;
            //result = new Mesh();
            //result.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            this.radius = radius;
        }
    }

    
}
