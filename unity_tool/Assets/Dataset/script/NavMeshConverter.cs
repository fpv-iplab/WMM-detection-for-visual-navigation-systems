using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor;
using UnityEngine.AI;
using Unity.AI.Navigation;
public class NavMeshConverter
{
    public static string meshDirectory = "Assets/Maps";
    // Update is called once per frame
    [MenuItem("Assets/Convert NavMesh")]
    static void NaveMesh2Mesh()
    {
        //if main directory doesnt exist create it
        if (Directory.Exists(meshDirectory))
        {
            Directory.Delete(meshDirectory, true);
        }
        Directory.CreateDirectory(meshDirectory);
        string[] guids = AssetDatabase.FindAssets("t:NavMeshData");
        foreach (var guid in guids)
        {
            NavMesh.RemoveAllNavMeshData();
            var path = AssetDatabase.GUIDToAssetPath(guid);
            NavMesh.AddNavMeshData(AssetDatabase.LoadAssetAtPath<NavMeshData>(path));
            NavMeshTriangulation triangles = NavMesh.CalculateTriangulation();
            Mesh my_mesh = new Mesh();
            my_mesh.vertices = triangles.vertices;
            my_mesh.triangles = triangles.indices;
            AssetDatabase.CreateAsset(my_mesh, meshDirectory + '/' + "Mesh_" + Path.GetFileName(path));
        }
        NavMesh.RemoveAllNavMeshData();
    }
    void bake_runtime(NavMeshSurface[] surfaces, Transform[] objectsToRotate)
    {
        for (int j = 0; j < objectsToRotate.Length; j++)
        {
            objectsToRotate[j].localRotation = Quaternion.Euler(new Vector3(0, Random.Range(0, 360), 0));
        }
        for (int i = 0; i < surfaces.Length; i++)
        {
            surfaces[i].BuildNavMesh();
        }
    }
}