using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.Perception.GroundTruth;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEditor;
using UnityEngine.Perception.Settings;
using UnityEngine.Perception.GroundTruth.Consumers;
using UnityEditor.Perception.GroundTruth;
using UnityEngine.Perception.Randomization.Scenarios;

public class logging_manager : MonoBehaviour
{
    public NavMeshAgent agent;
    public GameObject random_fov_obj;
    public NavMeshAgent line;
    public GameObject goal;
    private GameObject goal_parent;
    private GameObject starting_point_parent;
    public ParticleSystem particel;
    public ParticleSystem particel_wrong;
    public Transform[] childs_goal;
    public Transform[] childs_starting_point;
    public GameObject wrong_floor;
    public GameObject wrong_floor_detection;
    public GameObject pivot;
    public Vector3[] corners;
    public List<GameObject> flares;
    bool agent_started = false;
    public LineRenderer correct_line;
    public LineRenderer wrong_line;
    public GameObject random_seed;
    public GameObject Cameras;
    public int seed;
    float timer = 0.0f;
    float counter = 0.0f;
    public float framerate;
     float global_timer;
    public GameObject light_parent;
    public Light[] lights;
    string diff;
    string[] difficulty_list;
    public PerceptionCamera cam_to_save;
    public GameObject difficulty_labeler;
    public GameObject placeholder;

    private Transform original_starting_pos;
    public float delayTime = 3.0f;
    bool start_done;

    public int frame_cap;
    private int frame_counter=0, frame_to_save=0;
    private GameObject[] model_array;
    private int model_idex=0;

    private  GameObject navmesh_manager;
    public float checkInterval = 2.0f; // Check interval in seconds
    public float stuckThreshold = 0.1f; // Minimum movement threshold to consider not stuck
    public float velocityThreshold = 0.1f; // Minimum velocity threshold to consider not stuck
    private Vector3 lastPosition;
    private float lastTime;

    private string models_path="Assets\\Dataset\\Modelli";

    private string[] dirs;

    private string dir_path="<set your path here>\\Dataset_creator\\Acquisition-"+System.DateTime.Now.ToString("dd-MM-yyyy-HH-mm-ss");
    

    GameObject model_name;
    void Start()
    { 
        start_done=false;
        Directory.CreateDirectory(dir_path);
        PerceptionSettings.SetOutputBasePath(dir_path);
        model_name=GameObject.Find("Test");
        random_fov_obj = GameObject.Find("Random_fov");
        
        init_models();
        instantiate_actual_model();
        bakemodel();

        StartCoroutine(DelayedStartCoroutine());
    }

    private void init_models(){
        navmesh_manager =GameObject.Find("Manager_mesh");
        dirs = Directory.GetDirectories(models_path, "*", SearchOption.TopDirectoryOnly);
        model_array=new GameObject[dirs.Length];
        int j=0;
        foreach(string dir in dirs){
            string dirr=(string)dir.Clone();
            Debug.Log(dirr.Replace("/","\\")+"\\Model\\MODEL.prefab");
            model_array[j]=AssetDatabase.LoadAssetAtPath<GameObject>(dirr.Replace("/","\\")+"\\\\Model\\\\MODEL.prefab");
            //"Assets\\Dataset\\Modelli\\1-Back Rooms\\Model\\MODEL.prefab";
            if(model_array[j]!=null) Debug.Log("OBJ NAME: "+model_array[j].name);
            j++;
        }
        
        

    }

    private void instantiate_actual_model(){
        GameObject model =GameObject.Find("MODEL");
        if(model!=null) Destroy(model);
        
        model=Instantiate(model_array[model_idex], new Vector3(0,0,0), Quaternion.identity);
        model.name="MODEL";

        model_name.name=dirs[model_idex].Replace(models_path+"\\","");;
       

        model_idex++;
        
    }

    private void bakemodel(){
        navmesh_manager.GetComponent<manager>().SaroInit();
       
    }
    private Vector3 GetRandomNavMeshPoint()
    {
        // Get the NavMesh triangles
        NavMeshTriangulation navMeshTriangulation = NavMesh.CalculateTriangulation();

        // Select a random triangle from the list
        int randomTriangleIndex = Random.Range(0, navMeshTriangulation.indices.Length / 3);
        int startIndex = randomTriangleIndex * 3;

        // Calculate a random point within the selected triangle
        Vector3 pointInTriangle = Vector3.Lerp(
            navMeshTriangulation.vertices[navMeshTriangulation.indices[startIndex]],
            navMeshTriangulation.vertices[navMeshTriangulation.indices[startIndex + 1]],
            Random.value
        );

        pointInTriangle = Vector3.Lerp(pointInTriangle, navMeshTriangulation.vertices[navMeshTriangulation.indices[startIndex + 2]], Random.value);

        return pointInTriangle;
    }
        private bool CheckStuck()
    {
        Vector3 currentPosition = agent.transform.position;
        float deltaTime = Time.time - lastTime;
        Vector3 displacement = currentPosition - lastPosition;
        float velocity = displacement.magnitude / deltaTime;

        if (displacement.sqrMagnitude < stuckThreshold * stuckThreshold && velocity < velocityThreshold)
        {
            
            Debug.LogWarning("Agent is stuck!");
            return true;
            // You can perform actions here when the agent is stuck.
        }

        lastPosition = currentPosition;
        lastTime = Time.time;
        
        return false;
    }
    private IEnumerator DelayedStartCoroutine()
        {
            
            // Wait for the specified delay time
            yield return new WaitForSeconds(delayTime);

            //agent.gameObject.SetActive(false);
        difficulty_list = new string[3];
        difficulty_list[0] = "easy";
        difficulty_list[1] = "medium";
        difficulty_list[2] = "hard";
        diff = difficulty_list[Random.Range(0, difficulty_list.Length)];
        Debug.Log(diff);
        difficulty_labeler.name = diff;
        cam_to_save.gameObject.SetActive(true);
        Random.seed = seed;
        //size del modello 
        

        

        
        GameObject model =GameObject.Find("MODEL").gameObject;
        GameObject ingombro = GameObject.Find("ingombro");
        if(ingombro){ingombro.SetActive(false);}
        var renderers = model.GetComponentsInChildren<Renderer>();
        var bounds = renderers[0].bounds;
        for (var i = 1; i < renderers.Length; ++i)
            bounds.Encapsulate(renderers[i].bounds);

        
        
        frame_counter=0;

        /*wrong_floor=GameObject.Find("WrongNavmesh");
        wrong_floor_detection=GameObject.Find("Floor-detector-wrong");
        if(wrong_floor!=null) Debug.Log("FLOOR FOUND");*/


       
        
        
        
        
        agent.enabled = false;
        //Debug.Log("OBJ SIZE: "+bounds.size);
        frame_to_save=(int)bounds.size.x*(int)bounds.size.z*10;
       
        frame_to_save= frame_to_save/2;
        
        if(frame_to_save>frame_cap) frame_to_save=frame_cap;
        Debug.Log("FRAME TO SAVE: "+frame_to_save);
        random_seed.name = seed.ToString();
        
     
        

     
        Cameras.transform.rotation = Cameras.transform.rotation * Quaternion.Euler(Random.Range(0, 30), 0, 0);
        
        goal= placeholder;
        
        /////////////////////////
        
        for (int attempt = 0; attempt < 111; attempt++)
        {
            Vector3 startPoint, endPoint;
            NavMeshPath path = new NavMeshPath();

            // Generate random start and end points
            startPoint = GetRandomNavMeshPoint();
            endPoint = GetRandomNavMeshPoint();

            // Ensure the points are on the NavMesh and calculate the path
            if (NavMesh.CalculatePath(startPoint, endPoint, NavMesh.AllAreas, path) && path.corners.Length>10)
            {
                Debug.Log("Valid path exists between random points.");
                agent.transform.position = startPoint;
                line.transform.position = startPoint;
                goal.transform.position = endPoint;
                break; // Exit the loop once a valid path is found
            }
            else
            {
                Debug.Log("No valid path between random points. Retrying...");
            }
        }
        ///////////////////
      
        agent.path.ClearCorners();
        line.destination = goal.transform.position;
       
        NavMeshPath path_line = new NavMeshPath();
        NavMesh.CalculatePath(line.transform.position, goal.transform.position, NavMesh.AllAreas, path_line); //Saves the path in the path variable.
        corners = path_line.corners;
        for (int i = 0; i < corners.Length; i++)
        {
            var tmp_y = corners[i].y;
            corners[i].y = corners[i].z*(-1);
            corners[i].z = tmp_y;
        }
        correct_line.positionCount = corners.Length;
        correct_line.SetPositions(corners);
        wrong_line.positionCount = corners.Length;
        wrong_line.SetPositions(corners);
        agent_started = false;
        GameObject light_parent = model.transform.Find("Lighting").gameObject;
        lights = light_parent.transform.GetComponentsInChildren<Light>();

        // Loop through the lights and check if they are directional

        Cameras.transform.localPosition = new Vector3(0, Random.Range(1, 1.9f), 0);
        float random_light_intensity = Random.Range(2000, 77942);
        float random_light_temperature = Random.Range(3500, 14000);
        
        foreach (Light light in lights)
        {

            if (light.type == LightType.Directional)
            {
            }
            if (light.gameObject.GetComponent<LensFlare>() != null)
            {
                if (light.gameObject.GetComponent<LensFlare>().enabled == true)
                {
                    
                    light.gameObject.GetComponent<LensFlare>().brightness = random_light_intensity;

                }
            }
            // Set the intensity to the specified value
            light.intensity = random_light_intensity;
            light.colorTemperature = random_light_temperature;
            start_done=true;
            
        }
        }
    public static float RandomGaussian(float minValue = 0.0f, float maxValue = 1.0f)
    {
        float u, v, S;

        do
        {
            u = 2.0f * UnityEngine.Random.value - 1.0f;
            v = 2.0f * UnityEngine.Random.value - 1.0f;
            S = u * u + v * v;
        }
        while (S >= 1.0f);

        // Standard Normal Distribution
        float std = u * Mathf.Sqrt(-2.0f * Mathf.Log(S) / S);

        // Normal Distribution centered between the min and max value
        // and clamped following the "three-sigma rule"
        float mean = (minValue + maxValue) / 2.0f;
        float sigma = (maxValue - mean) / 3.0f;
        return Mathf.Clamp(std * sigma + mean, minValue, maxValue);
    }

    float get_error_translation(string difficulty)
    {
        if(difficulty == "easy") { return (Random.Range(0, 2) * 2 - 1) * RandomGaussian(0.4f, 0.6f); }
        if (difficulty == "medium") { return (Random.Range(0, 2) * 2 - 1) * RandomGaussian(0.1f, 0.2f); }
        if (difficulty == "hard") { return (Random.Range(0, 2) * 2 - 1) * RandomGaussian(0.01f, 0.1f); }
        return 0;

    }

    Quaternion get_error_rotation(string difficulty)
    {
        if (difficulty == "easy") { return (Quaternion.Euler(pivot.transform.localEulerAngles.x, pivot.transform.localEulerAngles.y + (Random.Range(0, 2) * 2 - 1) * RandomGaussian(20f, 50f), pivot.transform.localEulerAngles.z + (Random.Range(0, 2) * 2 - 1) * RandomGaussian(2f, 5f))); }
        if (difficulty == "medium") { return (Quaternion.Euler(pivot.transform.localEulerAngles.x, pivot.transform.localEulerAngles.y + (Random.Range(0, 2) * 2 - 1) * RandomGaussian(10f, 20f), pivot.transform.localEulerAngles.z )); }
        if (difficulty == "hard") { return (Quaternion.Euler(pivot.transform.localEulerAngles.x, pivot.transform.localEulerAngles.y + (Random.Range(0, 2) * 2 - 1) * RandomGaussian(5f, 10f), pivot.transform.localEulerAngles.z)); }
        return new Quaternion(0, 0, 0, 0);

    }

    // Update is called once per frame
    void Update()
    {

        if(start_done==true ){
            if( frame_counter<frame_to_save){
                //starting_point_parent.transform.position=original_starting_pos.position;
                timer += Time.deltaTime;
                global_timer += Time.deltaTime;

                if (agent_started==false)
                {
                    agent.gameObject.SetActive(true);
                    agent.enabled = true;
                    agent.destination = goal.transform.position;
                    agent_started = true;
                
                }
                if (!is_arrived(agent, goal))
                {

                    
                    if (timer >= framerate)
                    {

                        counter += 1;
                        timer = timer - framerate;


                        wrong_floor.transform.position = new Vector3(0, 0, 0);
                        wrong_floor_detection.transform.position = new Vector3(0, 0, 0);
                        cam_to_save.transform.localPosition = new Vector3(0, 0, 0);
                        wrong_floor.transform.rotation = Quaternion.identity;
                        wrong_floor_detection.transform.rotation = Quaternion.identity;

                        pivot.transform.position = agent.transform.position;   
                        wrong_floor.transform.parent = pivot.transform;
                        wrong_line.transform.parent = wrong_floor.transform;
                        wrong_line.transform.localPosition = new Vector3(0,0,0);
                        wrong_line.transform.localEulerAngles = new Vector3(-90,0,0);
                        wrong_floor_detection.transform.parent = pivot.transform;
                        wrong_floor_detection.transform.parent = wrong_floor.transform;


                        Vector3 translation_selected =new Vector3(get_error_translation(diff), 0 , get_error_translation(diff));
                        Quaternion rotation_selected = get_error_rotation(diff);





                        Vector3 translation = translation_selected;


                        //DatasetCapture implementare le varie difficolta. salvare il metadata in relazione alla difficolta


                        pivot.transform.position = pivot.transform.position + translation;
                        cam_to_save.transform.position = cam_to_save.transform.position + translation;
                        pivot.transform.localRotation = rotation_selected;
                        cam_to_save.transform.localRotation = rotation_selected;


                        foreach (Transform child in Cameras.transform)
                        {
                            child.GetComponent<PerceptionCamera>().RequestCapture();
                            cam_to_save.RequestCapture();
                            
                            //wrong_line.transform.parent = wrong_floor.transform;
                            //wrong_floor.transform.eulerAngles = euler;              
                        }
                        Debug.Log("salvo");
                        
                        wrong_floor.transform.parent = null;
                        wrong_floor_detection.transform.parent = null;
                        frame_counter++;
                    }
                }


                if (is_arrived(agent, goal) || CheckStuck())
            {
                agent.enabled = false;
                line.gameObject.SetActive(false);
            
                
                
                
                line.path.ClearCorners();
                agent.path.ClearCorners();
                diff = difficulty_list[Random.Range(0, difficulty_list.Length)];
                //Debug.Log(diff);
                difficulty_labeler.name = diff;
                
                /////////////////////////
                
                for (int attempt = 0; attempt < 111; attempt++)
                {
                    Vector3 startPoint, endPoint;
                    NavMeshPath path = new NavMeshPath();

                    // Generate random start and end points
                    startPoint = GetRandomNavMeshPoint();
                    endPoint = GetRandomNavMeshPoint();

                    // Ensure the points are on the NavMesh and calculate the path
                    if (NavMesh.CalculatePath(startPoint, endPoint, NavMesh.AllAreas, path) && path.corners.Length>10)
                    {
                        Debug.Log("Valid path exists between random points.");
                        agent.transform.position = startPoint;
                        line.transform.position = startPoint;
                        goal.transform.position = endPoint;
                        break; // Exit the loop once a valid path is found
                    }
                    else
                    {
                        Debug.Log("No valid path between random points. Retrying...");
                    }
                }
                ///////////////////

                line.path.ClearCorners();
                agent.path.ClearCorners();

                line.enabled = true;
                line.gameObject.SetActive(true);
                line.destination = goal.transform.position;
                NavMeshPath path_line = new NavMeshPath();
                NavMesh.CalculatePath(line.transform.position, goal.transform.position, NavMesh.AllAreas, path_line); //Saves the path in the path variable.
                corners = path_line.corners;
                for (int i = 0; i < corners.Length; i++)
                {
                    var tmp_y = corners[i].y;
                    corners[i].y = corners[i].z * (-1);
                    corners[i].z = tmp_y;
                }
                correct_line.positionCount = corners.Length;
                correct_line.SetPositions(corners);
                wrong_line.positionCount = corners.Length;
                wrong_line.SetPositions(corners);
                agent_started = false;
                float random_light_intensity = Random.Range(2000, 77942);
                float random_light_temperature = Random.Range(3500, 14000);
                foreach (Light light in lights)
                {
                    if (light.type == LightType.Directional)
                    {
                    }
                    if (light.gameObject.GetComponent<LensFlare>() != null)
                    {
                        if (light.gameObject.GetComponent<LensFlare>().enabled == true)
                        {
                     
                            light.gameObject.GetComponent<LensFlare>().brightness = random_light_intensity;

                        }
                    }
                    // Set the intensity to the specified value
                    light.intensity = random_light_intensity;
                    light.colorTemperature = random_light_temperature;
                    

                }
            
                Cameras.transform.localPosition = new Vector3(0, Random.Range(1, 1.9f), 0);
                agent.speed=Random.Range(0.5f, 1f);
                float randomFOV = Random.Range(60, 90);
                foreach (Camera childCamera in Cameras.GetComponentsInChildren<Camera>())
                {
                    // Generate a random FOV value within the specified range
                    

                    // Set the FOV of the current camera to the random value
                    childCamera.fieldOfView = randomFOV;
                }
                random_fov_obj.name = randomFOV.ToString();
                Invoke("change_havetosave", 2.0f);
            }
            }
            else if(is_arrived(agent, goal)|| CheckStuck()){
                start_done=false;
                instantiate_actual_model();
                bakemodel();
                StartCoroutine(DelayedStartCoroutine());
            }
        }
        
    
    
    
    }

    void OnGUI()
    {
        GUI.Label(new Rect(10, 10, 200, 20), "Update rate: " + (counter/ global_timer).ToString() + " FPS");
    }
    bool is_arrived(NavMeshAgent agent, GameObject goal)
    {
        if (Vector3.Distance(agent.transform.position, goal.transform.position) <= 1.5f) { return true; }
        else { return false; }
    }




    public float MyRemainingDistance(Vector3[] points)
    {
        if (points.Length < 2) return 0;
        float distance = 0;
        for (int i = 0; i < points.Length - 1; i++)
            distance += Vector3.Distance(points[i], points[i + 1]);
        return distance;
    }
}
