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
        //public Mesh result;
        public int subdivisions;
        public float radius;
        public NativeList<float3> vertList;
        public NativeList<TriangleIndices> faces;
        private static int getMiddlePoint(int p1, int p2, NativeList<float3> vertices, NativeHashMap<long, int> cache, float radius)
        {
            // first check if we have it already
            bool firstIsSmaller = p1 < p2;
            long smallerIndex = firstIsSmaller ? p1 : p2;
            long greaterIndex = firstIsSmaller ? p2 : p1;
            long key = ((smallerIndex & 0xFFFF0000) << 48) + ((greaterIndex & 0xFFFF0000) << 32) + ((smallerIndex & 0x0000FFFF) << 16) + (greaterIndex & 0x0000FFFF);

            int ret;

            if (cache.TryGetValue(key, out ret))
            {
                return ret;
            }

            // not in cache, calculate it
            float3 point1 = vertices[p1];
            float3 point2 = vertices[p2];
            float3 middle = math.lerp(point1, point2, 0.5f);

            // add vertex makes sure point is on unit sphere
            int i = vertices.Length;
            vertices.Add(math.normalize(middle) * radius);

            // store it, return index
            cache.Add(key, i);

            return i;
        }
        public void Execute()
        {
            //Create the isocahedron
            NativeHashMap<long, int> middlePointIndexCache = new NativeHashMap<long, int>(1, Allocator.Temp);

            // create 12 vertices of a icosahedron
            #region verts and tris
            float phi = (1f + math.sqrt(5f)) / 2f;

            vertList.Add(math.normalize(new float3(-1f, phi, 0f)) * radius);
            vertList.Add(math.normalize(new float3(1f, phi, 0f)) * radius);
            vertList.Add(math.normalize(new float3(-1f, -phi, 0f)) * radius);
            vertList.Add(math.normalize(new float3(1f, -phi, 0f)) * radius);

            vertList.Add(math.normalize(new float3(0f, -1f, phi)) * radius);
            vertList.Add(math.normalize(new float3(0f, 1f, phi)) * radius);
            vertList.Add(math.normalize(new float3(0f, -1f, -phi)) * radius);
            vertList.Add(math.normalize(new float3(0f, 1f, -phi)) * radius);

            vertList.Add(math.normalize(new float3(phi, 0f, -1f)) * radius);
            vertList.Add(math.normalize(new float3(phi, 0f, 1f)) * radius);
            vertList.Add(math.normalize(new float3(-phi, 0f, -1f)) * radius);
            vertList.Add(math.normalize(new float3(-phi, 0f, 1f)) * radius);

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
            #endregion
            NativeList<TriangleIndices> faces2 = new NativeList<TriangleIndices>(Allocator.Temp);

            for (int i = 0; i < subdivisions; i++)
            {
                faces2.Clear();
                for (int t = 0; t < faces.Length; t++)
                {
                    TriangleIndices tri = faces[t];
                    // replace triangle by 4 triangles
                    int a = getMiddlePoint(tri.a, tri.b, vertList, middlePointIndexCache, radius);
                    int b = getMiddlePoint(tri.b, tri.c, vertList, middlePointIndexCache, radius);
                    int c = getMiddlePoint(tri.c, tri.a, vertList, middlePointIndexCache, radius);

                    faces2.Add(new TriangleIndices(tri.a, a, c));
                    faces2.Add(new TriangleIndices(tri.b, b, a));
                    faces2.Add(new TriangleIndices(tri.c, c, b));
                    faces2.Add(new TriangleIndices(a, b, c));
                }
                faces.Clear();
                faces.AddRange(faces2);
            }
            faces2.Dispose();
            middlePointIndexCache.Dispose();
        }
        public IcosphereJob(int subdivisions, float radius)
        {
            vertList = new NativeList<float3>(Allocator.Persistent);
            faces = new NativeList<TriangleIndices>(Allocator.Persistent);
            //vertList.Dispose();
            //faces.Dispose();
            this.subdivisions = subdivisions;
            //result = new Mesh();
            //result.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
            this.radius = radius;
        }

    }
    [BurstCompile]
    struct CentroidJob : IJobParallelFor
    {
        [ReadOnly]
        NativeList<float3> verts;
        [ReadOnly]
        NativeList<TriangleIndices> tris;
        [WriteOnly]
        public NativeArray<float3> centroids;
        public void Execute(int i)
        {
            TriangleIndices t = tris[i];
            centroids[i] = (verts[t.a] + verts[t.b] + verts[t.c]) / 3f;
        }
        public CentroidJob(NativeList<float3> verts, NativeList<TriangleIndices> faces)
        {
            this.verts = verts;
            tris = faces;
            centroids = new NativeArray<float3>(tris.Length, Allocator.Persistent);
        }
    }
}
