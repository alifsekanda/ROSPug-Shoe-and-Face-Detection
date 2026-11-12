# ROSPug Office Navigation & Waypoint Patrol Manual

## 1. System Information

| Component | Configuration |
|---|---|
| Robot | Hiwonder ROSPug |
| Computer | NVIDIA Jetson Orin Nano |
| Operating System | Ubuntu 18.04.6 |
| ROS Distribution | ROS Melodic |
| LiDAR | MS200 |
| Camera | CSI |
| ROS Workspace | `~/pug` |
| Robot IP Address | `192.168.0.253` |
| ROS Navigation | AMCL + Move Base |
| Map | `map_03` |
| Waypoints | `waypointsoffice.yaml` |

### Important File Locations

**Office map**
```bash
/home/hiwonder/pug/src/pug_slam/maps/map_03.yaml
/home/hiwonder/pug/src/pug_slam/maps/map_03.pgm
```

**Waypoint file**
```bash
/home/hiwonder/pug/src/carto_ws/cartographer_detailed_comments_ws/src/cartographer_ros/cartographer_ros/waypoints/waypointsoffice.yaml
```

**Navigation launch file**
```bash
~/pug/src/pug_navigation/launch/navigation.launch
```

**RViz launch file**
```bash
~/pug/src/pug_navigation/launch/rviz_navigation.launch
```

**Waypoint player launch file**
```bash
~/pug/src/pug_navigation/launch/waypoint_player.launch
```

---

## 2. Connect to the Jetson

From Windows PowerShell:

```powershell
ssh hiwonder@192.168.0.253
```

Enter the Jetson password when prompted.

### Access the GNOME Desktop with TigerVNC

RViz should run inside the Jetson desktop rather than through X11 forwarding.

**Step 1: Start the VNC server on the Jetson**

Open an SSH terminal and execute:

```bash
sudo x11vnc -display :0 -auth /run/user/1000/gdm/Xauthority -forever -shared -rfbauth /home/hiwonder/.vnc/passwd -localhost -rfbport 5900
```

Keep this terminal open.

**Step 2: Create an SSH tunnel on Windows**

Open another Windows PowerShell window:

```powershell
ssh -L 5900:localhost:5900 hiwonder@192.168.0.253
```

Keep this connection open.

**Step 3: Connect using TigerVNC Viewer**

Enter:

```text
localhost:5900
```

You should now see the Jetson's GNOME desktop.

---

## 3. Stop the Default ROSPug Bringup Service

**Important: Perform this step before starting the manual navigation system.**

The `pug_bringup.service` normally starts the robot's default application bringup through its systemd service.

The separate navigation launch will start the required navigation components. Running both bringup systems simultaneously can cause conflicting ROS nodes, drivers or hardware access.

### Stop the service

Open an SSH terminal and execute:

```bash
sudo systemctl stop pug_bringup.service
```

### Verify that it stopped

```bash
sudo systemctl status pug_bringup.service
```

Expected status:

```text
Active: inactive (dead)
```

If the service is inactive, continue to the next step.

**Important:**

- Do not start `navigation.launch` before stopping the service.
- Do not run the default bringup and manual navigation stack simultaneously.
- Stopping the service does not uninstall or disable it permanently. It only stops it for the current session.

---

## 4. Prepare the ROS Environment

Every new terminal used for ROS commands must load the ROS and workspace environments.

Execute:

```bash
source /opt/ros/melodic/setup.bash
source ~/pug/devel/setup.bash
```

These commands should be executed in every new terminal before running ROS packages.

---

## 5. Start the Navigation System

The navigation launch file starts the main components required for autonomous navigation.

These include:

- LiDAR and sensor drivers
- Robot odometry and TF
- Map Server
- AMCL localisation
- Move Base navigation
- Global and local costmaps
- Navigation planners

### Terminal 1: Start Navigation

Open a new Jetson terminal and execute:

```bash
source /opt/ros/melodic/setup.bash
source ~/pug/devel/setup.bash

roslaunch --screen pug_navigation navigation.launch map:=map_03
```

**Important:**

- Keep this terminal running.
- Do not close it while the robot is navigating.
- The `map:=map_03` argument tells the launch file to load `map_03.yaml`.
- Navigation launch already starts its own Map Server. Do not start another Map Server separately.

Expected map loading output:

```text
Loading map from image ".../map_03.pgm"
Read a 1984 X 1984 map @ 0.050 m/cell
```

---

## 6. Open RViz

RViz is used to visualise the map, robot position, LiDAR scan, TF frames and navigation goals.

Open a terminal inside the Jetson GNOME desktop using TigerVNC.

### Terminal 2: Start RViz

```bash
source /opt/ros/melodic/setup.bash
source ~/pug/devel/setup.bash

roslaunch pug_navigation rviz_navigation.launch
```

The existing RViz configuration should load automatically.

### RViz Configuration

| Display | Setting |
|---|---|
| Fixed Frame | `map` |
| Map | `/map` |
| LaserScan | `/scan` |
| RobotModel | Enabled |
| TF | Enabled |

### RViz Tools

**2D Pose Estimate**

Used to tell AMCL the robot's approximate starting position and direction.

1. Select `2D Pose Estimate`.
2. Click the robot's actual location on the map.
3. Drag in the direction the robot is facing.
4. Release.

**2D Nav Goal**

Used to manually send the robot to a destination.

1. Select `2D Nav Goal`.
2. Click the destination on the map.
3. Drag to specify the desired final direction.
4. Release.

**Publish Point**

Publishes a selected map coordinate to `/clicked_point`. This can be used to inspect or capture waypoint coordinates.

---

## 7. Verify Navigation Is Running

Open another terminal.

### Check Navigation Nodes

```bash
rosnode list | grep -E 'map_server|amcl|move_base'
```

Expected nodes:

```text
/amcl
/map_server
/move_base_node
```

### Check Navigation Topics

```bash
rostopic list | grep -E 'map|amcl|move_base'
```

Expected topics include:

```text
/map
/map_metadata
/amcl_pose
/move_base/goal
/move_base/status
/move_base/result
/move_base_simple/goal
```

### Check LiDAR

```bash
rostopic hz /scan
```

The LiDAR should publish approximately 10 Hz.

### Check Robot Position

```bash
rostopic echo /amcl_pose
```

This displays the robot's estimated position in the map coordinate system.

### Check TF

```bash
rosrun tf tf_echo map base_footprint
```

This displays the transformation between the map and robot base.

Press `Ctrl+C` to stop the TF output.

---

## 8. Test Manual Navigation First

Before starting waypoint patrol, verify that the robot can navigate to a manually selected destination.

1. Make sure `pug_bringup.service` is stopped.
2. Make sure `navigation.launch` is running.
3. Make sure RViz is running.
4. Confirm the robot is correctly positioned on the map.
5. Set the initial pose using `2D Pose Estimate` if necessary.
6. Select `2D Nav Goal`.
7. Choose a clear destination on the map.
8. Observe the robot's movement.

Do not proceed to autonomous patrol until manual navigation works correctly.

---

## 9. Waypoint Patrol Configuration

The waypoint player reads a YAML file containing the robot's navigation destinations.

The current waypoint file is:

```bash
/home/hiwonder/pug/src/carto_ws/cartographer_detailed_comments_ws/src/cartographer_ros/cartographer_ros/waypoints/waypointsoffice.yaml
```

It contains 20 waypoints:

```text
Point_0
Point_1
Point_2
...
Point_19
```

Each waypoint contains:

- A waypoint name
- X position
- Y position
- Z position
- Quaternion orientation

The file uses the `map` coordinate frame.

### Waypoint YAML Structure

Example:

```yaml
frame_id: map

waypoints:
  - name: Point_0
    pose:
      position: {x: 0.477572, y: -0.029322, z: 0.0}
      orientation: {x: 0.0, y: 0.0, z: -0.008782, w: 0.999961}
```

The waypoint player reads this structure and sends each destination to Move Base.

### Patrol Configuration

| Parameter | Value | Description |
|---|---|---|
| `frame_id` | `map` | Coordinate reference |
| `loop` | `true` | Repeat patrol indefinitely |
| `reverse` | `false` | Do not reverse waypoint order |
| `start_index` | `0` | Start at Point_0 |
| `timeout_sec` | `60.0` | Maximum time per attempt |
| `retry_limit` | `2` | Retry failed destinations |
| `start_paused` | `false` | Start moving immediately |
| `tick_period` | `1.0` | Player update interval |
| `move_base_action` | `/move_base` | Navigation action server |
| `draw_markers` | `true` | Publish waypoint markers |

With `retry_limit=2`, a waypoint can receive up to three attempts in total.

If all attempts fail, the current script advances to the next waypoint.

---

## 10. Start Autonomous Waypoint Patrol

**Safety check before starting:**

- `pug_bringup.service` is stopped.
- Navigation system is running.
- RViz displays the correct map.
- Robot is correctly localised.
- The route is clear.
- The robot is supervised.

The current launch configuration has `start_paused=false`. The robot will begin moving as soon as the waypoint player starts.

### Terminal 3: Start Waypoint Player

Open another terminal:

```bash
source /opt/ros/melodic/setup.bash
source ~/pug/devel/setup.bash

roslaunch pug_navigation waypoint_player.launch yaml_path:=/home/hiwonder/pug/src/carto_ws/cartographer_detailed_comments_ws/src/cartographer_ros/cartographer_ros/waypoints/waypointsoffice.yaml
```

The robot should begin travelling through the waypoints in order:

```text
Point_0
   ↓
Point_1
   ↓
Point_2
   ↓
...
   ↓
Point_19
   ↓
Point_0
   ↓
Repeat
```

Keep the waypoint player terminal open.

---

## 11. Monitor Waypoint Patrol

### Check Waypoint Count

```bash
rosservice call /waypoint_player/count
```

Expected result should report 20 waypoints.

### Check Patrol Topics

```bash
rostopic list | grep patrol
```

The marker topic should normally be:

```text
/patrol_points
```

If it is available, add a `MarkerArray` display in RViz and select the corresponding topic.

### Check Navigation Status

```bash
rostopic echo /move_base/status
```

This displays the current navigation goal status.

### Check Navigation Result

```bash
rostopic echo /move_base/result
```

This displays the result of completed navigation goals.

---

## 12. Control the Waypoint Player

The waypoint player provides ROS services to control patrol.

### Stop Patrol

```bash
rosservice call /waypoint_player/stop
```

Pauses the player and cancels its current navigation goal.

Use this when you need to stop the patrol.

### Pause Patrol

```bash
rosservice call /waypoint_player/pause
```

Pauses waypoint progression. The current implementation does not cancel an active navigation goal, so the robot may continue moving towards its current destination.

### Resume Patrol

```bash
rosservice call /waypoint_player/resume
```

Resumes waypoint progression.

### Skip to Next Waypoint

```bash
rosservice call /waypoint_player/next
```

Changes the waypoint index. It does not cancel an active navigation goal.

### Go to Previous Waypoint

```bash
rosservice call /waypoint_player/prev
```

Changes the waypoint index. It does not cancel an active navigation goal.

### Reload Waypoints

```bash
rosservice call /waypoint_player/reload
```

Reloads the YAML/JSON waypoint file and rebuilds the waypoint sequence.

### Check Waypoint Count

```bash
rosservice call /waypoint_player/count
```

---

## 13. Stop the System

To stop autonomous patrol and return to the normal robot bringup:

**Step 1: Stop the waypoint player**

```bash
rosservice call /waypoint_player/stop
```

**Step 2: Stop the navigation system**

Go to Terminal 1 and press:

```text
Ctrl+C
```

**Step 3: Stop RViz**

Go to Terminal 2 and press:

```text
Ctrl+C
```

**Step 4: Restart the default ROSPug bringup service, if required**

Only after the manual navigation stack has stopped:

```bash
sudo systemctl start pug_bringup.service
```

Verify:

```bash
sudo systemctl status pug_bringup.service
```

**Step 5: Close remaining SSH and VNC sessions**

Close the SSH tunnel and x11vnc terminal when remote desktop access is no longer needed.

---

## 14. Complete Command Checklist

This is the normal startup sequence after the Jetson has booted.

### Step 1 — Stop Default Bringup

Run through SSH:

```bash
sudo systemctl stop pug_bringup.service
sudo systemctl status pug_bringup.service
```

Confirm it shows `inactive (dead)`.

### Terminal 1 — Navigation

```bash
source /opt/ros/melodic/setup.bash
source ~/pug/devel/setup.bash

roslaunch --screen pug_navigation navigation.launch map:=map_03
```

### Terminal 2 — RViz

Run inside the Jetson desktop:

```bash
source /opt/ros/melodic/setup.bash
source ~/pug/devel/setup.bash

roslaunch pug_navigation rviz_navigation.launch
```

### Before Patrol — Verify Localisation

```bash
rosnode list | grep -E 'map_server|amcl|move_base'
rostopic hz /scan
rostopic echo /amcl_pose
```

### Terminal 3 — Waypoint Patrol

```bash
source /opt/ros/melodic/setup.bash
source ~/pug/devel/setup.bash

roslaunch pug_navigation waypoint_player.launch yaml_path:=/home/hiwonder/pug/src/carto_ws/cartographer_detailed_comments_ws/src/cartographer_ros/cartographer_ros/waypoints/waypointsoffice.yaml
```

### Stop Patrol

```bash
rosservice call /waypoint_player/stop
```

### Stop Navigation

Press `Ctrl+C` in Terminal 1.

### Restart Default Bringup (Optional)

```bash
sudo systemctl start pug_bringup.service
```

---

## 15. Troubleshooting

### Problem: `pug_bringup.service` is still running

Check:

```bash
sudo systemctl status pug_bringup.service
```

Stop it:

```bash
sudo systemctl stop pug_bringup.service
```

### Problem: `Resource not found: pug_navigation`

Load the workspace environment:

```bash
source /opt/ros/melodic/setup.bash
source ~/pug/devel/setup.bash
```

Check package location:

```bash
rospack find pug_navigation
```

### Problem: Map does not appear in RViz

Check:

```bash
rostopic list | grep map
```

Check map publishing:

```bash
rostopic echo /map_metadata
```

Ensure the Fixed Frame in RViz is:

```text
map
```

### Problem: Robot does not move

Check:

```bash
rosnode list | grep move_base
```

Check LiDAR:

```bash
rostopic hz /scan
```

Check navigation status:

```bash
rostopic echo /move_base/status
```

Make sure the robot has been correctly localised using `2D Pose Estimate`.

### Problem: Waypoint player cannot find the YAML file

Check that the file exists:

```bash
ls -l /home/hiwonder/pug/src/carto_ws/cartographer_detailed_comments_ws/src/cartographer_ros/cartographer_ros/waypoints/waypointsoffice.yaml
```

Make sure the launch command includes the correct `yaml_path:=` argument.

### Problem: Waypoint player reports zero waypoints

Check the YAML file:

```bash
cat /home/hiwonder/pug/src/carto_ws/cartographer_detailed_comments_ws/src/cartographer_ros/cartographer_ros/waypoints/waypointsoffice.yaml
```

Confirm that it contains a top-level `waypoints:` list and that the entries contain `pose`, `position` and `orientation`.

---

## 16. Important Notes

1. Always stop `pug_bringup.service` before starting the manual navigation stack.
2. Always start navigation before starting the waypoint player.
3. RViz is a visualisation tool. It does not start the navigation stack.
4. `navigation.launch` already starts Map Server, AMCL and Move Base.
5. Use the same map that was used to record the waypoint coordinates.
6. The waypoint YAML uses the `map` coordinate frame.
7. Do not run two waypoint players simultaneously.
8. Keep the navigation and waypoint player terminals open while the robot is operating.
9. The current waypoint player can advance after a waypoint fails all its attempts. Check the robot's progress during the first patrol cycle.
10. The current pause service does not cancel an active navigation goal. Use the stop service when the robot must stop its current goal.
11. Restart `pug_bringup.service` only after shutting down the manual navigation stack.
12. Always supervise the robot during autonomous navigation.

---

**End of Manual**

Project: ROSPug Office Autonomous Navigation and Waypoint Patrol  
Platform: Hiwonder ROSPug + Jetson Orin Nano + ROS Melodic
