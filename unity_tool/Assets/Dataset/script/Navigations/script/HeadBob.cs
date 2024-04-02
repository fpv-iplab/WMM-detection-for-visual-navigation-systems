// original by Mr. Animator
// adapted to C# by @torahhorse
// http://wiki.unity3d.com/index.php/Headbobber

using UnityEngine;
using System.Collections;

public class HeadBob : MonoBehaviour
{
    public float speed = 2.0f; // speed of the camera movement
    public float distance = 1.0f; // distance between left and right movements
    public float smoothing = 0.1f; // smoothing factor for the camera movement
    public float rotationAmount = 10.0f; // amount of rotation on the z-axis
    public float lookAtFloorInterval = 5.0f; // interval for looking at the floor
    public float lookAtFloorDuration = 10.0f; // duration for looking at the floor
    private float targetPosition; // target position for the camera
    private Vector3 currentVelocity; // current velocity of the camera
    private float targetRotation; // target rotation for the camera
    private float currentRotationVelocity; // current rotation velocity of the camera
    private bool lookingAtFloor; // flag indicating if the camera is currently looking at the floor
    private float lookAtFloorTimeLeft; // time left for the current look-at-floor sequence
    public float starting_angle;
    public float floor_angle;
     public logging_manager this_logging_manager;
    void Start()
    {
        Random.seed = this_logging_manager.seed;
        targetPosition = transform.position.x; // set the initial target position to the camera's current position
        targetRotation = transform.rotation.eulerAngles.z; // set the initial target rotation to the camera's current rotation
        lookingAtFloor = false;
        lookAtFloorTimeLeft = 0.0f;
        starting_angle = Random.Range(0f, 20f);
        floor_angle = Random.Range(0f, 80f);
    }
    void Update()
    {
        if (!lookingAtFloor)
        {
            // calculate the new target position based on the current direction
            float direction = Mathf.Sign(Mathf.Sin(Time.time * speed));
            float offset = direction * distance / 2.0f;
            targetPosition = transform.position.x + offset;
            // calculate the new target rotation based on the current direction
            targetRotation = direction * rotationAmount;
            // smoothly move the camera towards the target position and rotation
            transform.position = Vector3.SmoothDamp(transform.position, new Vector3(targetPosition, transform.position.y, transform.position.z), ref currentVelocity, smoothing);
            transform.rotation = Quaternion.Euler(transform.rotation.eulerAngles.x, transform.rotation.eulerAngles.y, Mathf.SmoothDampAngle(transform.rotation.eulerAngles.z, targetRotation, ref currentRotationVelocity, smoothing));
            // check if it's time to start a new look-at-floor sequence
            if (Random.Range(0.0f, 1.0f) < Time.deltaTime / lookAtFloorInterval)
            {
                lookingAtFloor = true;
                lookAtFloorTimeLeft = lookAtFloorDuration;
            }
        }
        else
        {
            float direction = Mathf.Sign(Mathf.Sin(Time.time * speed));
            float offset = direction * distance / 2.0f;
            targetPosition = transform.position.x + offset;
            // calculate the new target rotation based on the current direction
            targetRotation = direction * rotationAmount;
            // smoothly move the camera towards the target position and rotation
            transform.position = Vector3.SmoothDamp(transform.position, new Vector3(targetPosition, transform.position.y, transform.position.z), ref currentVelocity, smoothing);
            transform.rotation = Quaternion.Euler(transform.rotation.eulerAngles.x, transform.rotation.eulerAngles.y, Mathf.SmoothDampAngle(transform.rotation.eulerAngles.z, targetRotation, ref currentRotationVelocity, smoothing));
            // update the time left for the current look-at-floor sequence
            lookAtFloorTimeLeft -= Time.deltaTime;
            if (lookAtFloorTimeLeft > 0.0f)
            {
                // smoothly rotate the camera downwards
                transform.rotation = Quaternion.Lerp(transform.rotation, Quaternion.Euler(floor_angle, transform.rotation.eulerAngles.y, transform.rotation.eulerAngles.z), Time.deltaTime);
            }
            else
            {
                // smoothly rotate the camera back to the front
                transform.rotation = Quaternion.Lerp(transform.rotation, Quaternion.Euler(starting_angle, transform.rotation.eulerAngles.y, transform.rotation.eulerAngles.z), Time.deltaTime);
                // check if the camera has returned to the front
                if (Mathf.Abs(transform.rotation.eulerAngles.x) <= starting_angle+5)
                {
                    starting_angle = Random.Range(0f, 20f);
                    floor_angle = Random.Range(0f, 80f);
                    lookingAtFloor = false;
   
                }
            }
        }
    }
}