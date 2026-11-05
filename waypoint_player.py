#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Plays back waypoints from YAML/JSON and navigates through them using move_base action.
# ROS: Melodic (Python 2)

from __future__ import print_function
import os
import json
import yaml
import rospy
import actionlib
import signal
from math import isnan
from copy import deepcopy

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_srvs.srv import Empty, EmptyResponse
from std_srvs.srv import Trigger, TriggerResponse

# ------------ Parameters (filled in main) ------------
WAYPOINT_YAML  = ""
WAYPOINT_JSON  = ""
FRAME_ID       = "map"
LOOP           = True
REVERSE        = False
START_INDEX    = 0
TIMEOUT_SEC    = 60.0      # per goal
RETRY_LIMIT    = 2         # per goal, after timeout/aborted
CLEAR_COSTMAPS = "/move_base/clear_costmaps"
DRAW_MARKERS   = True
MARKER_NS      = "patrol_points"
MARKER_SCALE   = 0.5

GOAL_TOPIC_NAME = "/move_base"  # action name

# ------------ Runtime state ------------
waypoints = []     # list of dicts with 'name' and 'pose'
order = []         # indices in play order
idx = 0            # position within 'order'
paused = False
client = None
marker_pub = None

# ------------ Helpers ------------
def _load_waypoints():
    """Load waypoints from YAML first (if present), else JSON."""
    global waypoints, FRAME_ID
    waypoints = []

    if WAYPOINT_YAML and os.path.exists(WAYPOINT_YAML):
        try:
            with open(WAYPOINT_YAML, "r") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            rospy.logerr("Failed to load YAML: %s", e)
            data = {}
    elif WAYPOINT_JSON and os.path.exists(WAYPOINT_JSON):
        try:
            with open(WAYPOINT_JSON, "r") as f:
                data = json.load(f)
        except Exception as e:
            rospy.logerr("Failed to load JSON: %s", e)
            data = {}
    else:
        rospy.logwarn("No waypoint file found.")
        data = {}

    if isinstance(data, dict):
        if "frame_id" in data and data["frame_id"]:
            FRAME_ID_local = data["frame_id"]
        else:
            FRAME_ID_local = FRAME_ID  # fall back to param

        raw = data.get("waypoints", [])
        if not isinstance(raw, list):
            rospy.logwarn("Waypoints content malformed; expected list.")
            raw = []

        # normalize & validate
        for w in raw:
            try:
                name = w.get("name", "")
                pose = w.get("pose", {})
                pos  = pose.get("position", {})
                ori  = pose.get("orientation", {})

                x = float(pos.get("x", 0.0))
                y = float(pos.get("y", 0.0))
                z = float(pos.get("z", 0.0))
                ox = float(ori.get("x", 0.0))
                oy = float(ori.get("y", 0.0))
                oz = float(ori.get("z", 0.0))
                ow = float(ori.get("w", 1.0))
                if any(isnan(v) for v in [x, y, z, ox, oy, oz, ow]):
                    raise ValueError("NaN in pose")

                waypoints.append({
                    "name": name,
                    "pose": {
                        "frame_id": FRAME_ID_local,
                        "position": {"x": x, "y": y, "z": z},
                        "orientation": {"x": ox, "y": oy, "z": oz, "w": ow}
                    }
                })
            except Exception as e:
                rospy.logwarn("Skipping malformed waypoint: %s", e)

        # prefer the file's frame for playback
        if waypoints:
            FRAME_ID_from_file = waypoints[0]["pose"].get("frame_id", FRAME_ID)
            if FRAME_ID_from_file:
                return FRAME_ID_from_file
    return FRAME_ID

def _build_order():
    global order, idx
    n = len(waypoints)
    order = range(n) if not REVERSE else list(reversed(range(n)))
    # clamp start
    s = START_INDEX if 0 <= START_INDEX < n else 0
    if REVERSE:
        # if reversed, interpret START_INDEX in natural (0..n-1) then find its position in reversed order
        try:
            s = order.index(START_INDEX)
        except ValueError:
            s = 0
    idx = s

def _draw_markers():
    if not DRAW_MARKERS or marker_pub is None:
        return
    arr = MarkerArray()
    # delete-all first
    mdel = Marker()
    mdel.header.frame_id = FRAME_ID
    mdel.action = Marker.DELETEALL
    arr.markers.append(mdel)

    for i, w_i in enumerate(order):
        w = waypoints[w_i]
        pos = w["pose"]["position"]
        m = Marker()
        m.header.frame_id = FRAME_ID
        m.ns = MARKER_NS
        m.id = i
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.scale.x = MARKER_SCALE
        m.scale.y = MARKER_SCALE
        m.scale.z = MARKER_SCALE
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.pose.position.x = pos["x"]
        m.pose.position.y = pos["y"]
        m.pose.position.z = 0.0
        m.pose.orientation.w = 1.0
        label = waypoints[w_i].get("name", "")
        m.text = "{}".format(label if label else (i+1))
        arr.markers.append(m)

    marker_pub.publish(arr)

def _goal_from(w):
    g = MoveBaseGoal()
    g.target_pose = PoseStamped()
    g.target_pose.header.frame_id = w["pose"].get("frame_id", FRAME_ID)
    g.target_pose.header.stamp = rospy.Time.now()
    pos = w["pose"]["position"]
    ori = w["pose"]["orientation"]
    g.target_pose.pose.position.x = pos["x"]
    g.target_pose.pose.position.y = pos["y"]
    g.target_pose.pose.position.z = pos["z"]
    g.target_pose.pose.orientation.x = ori["x"]
    g.target_pose.pose.orientation.y = ori["y"]
    g.target_pose.pose.orientation.z = ori["z"]
    g.target_pose.pose.orientation.w = ori["w"]
    return g

def _clear_costmaps():
    try:
        rospy.wait_for_service(CLEAR_COSTMAPS, timeout=3.0)
        srv = rospy.ServiceProxy(CLEAR_COSTMAPS, Empty)
        srv()
        rospy.loginfo("Cleared costmaps.")
    except Exception as e:
        rospy.logwarn("Clear costmaps failed: %s", e)

def _send_and_wait(g):
    """Send goal, wait with retries; return True if succeeded."""
    client.send_goal(g)
    finished = client.wait_for_result(rospy.Duration(TIMEOUT_SEC))
    if finished:
        state = client.get_state()
        # 3 = SUCCEEDED
        return (state == 3)
    else:
        client.cancel_goal()
        return False

# ------------ Control flow ------------
def _play_once():
    """Play current waypoint once with retries; advance idx accordingly."""
    global idx
    if paused or not waypoints or not order:
        return

    w = waypoints[order[idx]]
    name = w.get("name", "#{}".format(order[idx]))
    g = _goal_from(w)

    rospy.loginfo("Navigating to [%s] (index %d/%d).", name, idx+1, len(order))

    # try + retries
    for attempt in range(RETRY_LIMIT + 1):
        ok = _send_and_wait(g)
        if ok:
            rospy.loginfo("Reached [%s].", name)
            break
        if paused:
            # Goal cancelled by ~stop: don't retry or advance, so ~resume re-sends this waypoint.
            rospy.loginfo("Stopped before reaching [%s].", name)
            return
        rospy.logwarn("Failed to reach [%s] (attempt %d/%d).", name, attempt+1, RETRY_LIMIT+1)
        _clear_costmaps()
        rospy.sleep(0.5)
    # advance
    idx += 1
    if idx >= len(order):
        if LOOP:
            idx = 0
        else:
            rospy.loginfo("Finished route.")
            # pause at the end if not looping
            _pause_impl(True)

def _tick(event):
    """Timer callback — drives the player when not paused."""
    if not paused:
        _play_once()

# ------------ Services ------------
def _pause_impl(set_pause):
    global paused
    paused = bool(set_pause)
    rospy.loginfo("Player %s.", "paused" if paused else "running")

def srv_pause(_):
    _pause_impl(True)
    return EmptyResponse()

def srv_resume(_):
    _pause_impl(False)
    return EmptyResponse()

def srv_stop(_):
    _pause_impl(True)
    try:
        client.cancel_all_goals()
    except:
        pass
    return EmptyResponse()

def srv_next(_):
    global idx
    # skip to next and run immediately if playing
    if waypoints:
        idx = (idx + 1) % len(order)
    return EmptyResponse()

def srv_prev(_):
    global idx
    if waypoints:
        idx = (idx - 1) % len(order)
    return EmptyResponse()

def srv_reload(_):
    global FRAME_ID
    FRAME_ID = _load_waypoints()
    _build_order()
    _draw_markers()
    msg = "Reloaded {} waypoints.".format(len(waypoints))
    rospy.loginfo(msg)
    return TriggerResponse(success=True, message=msg)

def srv_count(_):
    return TriggerResponse(success=True, message="{} waypoints loaded.".format(len(waypoints)))

# ------------ Shutdown ------------
def _signal_handler(signum, frame):
    try:
        client.cancel_all_goals()
    except:
        pass
    rospy.signal_shutdown("SIGINT")

# ------------ Main ------------
if __name__ == "__main__":
    rospy.init_node("waypoint_player")

    # params
    FRAME_ID       = rospy.get_param("~frame_id", "map")
    WAYPOINT_YAML  = rospy.get_param("~yaml_path", "/home/hiwonder/waypoints/waypoints.yaml")
    WAYPOINT_JSON  = rospy.get_param("~json_path", "/home/hiwonder/waypoints/waypoints.json")
    LOOP           = rospy.get_param("~loop", True)
    REVERSE        = rospy.get_param("~reverse", False)
    START_INDEX    = int(rospy.get_param("~start_index", 0))
    TIMEOUT_SEC    = float(rospy.get_param("~timeout_sec", 60.0))
    RETRY_LIMIT    = int(rospy.get_param("~retry_limit", 2))
    CLEAR_COSTMAPS = rospy.get_param("~clear_costmaps_srv", "/move_base/clear_costmaps")
    DRAW_MARKERS   = rospy.get_param("~draw_markers", True)
    MARKER_NS      = rospy.get_param("~marker_ns", "patrol_points")
    MARKER_SCALE   = float(rospy.get_param("~marker_scale", 0.5))
    GOAL_TOPIC_NAME = rospy.get_param("~move_base_action", "/move_base")

    # client & pubs
    client = actionlib.SimpleActionClient(GOAL_TOPIC_NAME, MoveBaseAction)
    rospy.loginfo("Waiting for action server %s ...", GOAL_TOPIC_NAME)
    client.wait_for_server()
    rospy.loginfo("Connected to %s.", GOAL_TOPIC_NAME)

    if DRAW_MARKERS:
        marker_pub = rospy.Publisher("patrol_points", MarkerArray, queue_size=10, latch=True)

    # load route
    FRAME_ID = _load_waypoints()
    _build_order()
    _draw_markers()

    # services
    rospy.Service("~pause",  Empty,  srv_pause)
    rospy.Service("~resume", Empty,  srv_resume)
    rospy.Service("~stop",   Empty,  srv_stop)
    rospy.Service("~next",   Empty,  srv_next)
    rospy.Service("~prev",   Empty,  srv_prev)
    rospy.Service("~reload", Trigger, srv_reload)
    rospy.Service("~count",  Trigger, srv_count)

    # start paused? choose via param
    start_paused = rospy.get_param("~start_paused", False)
    _pause_impl(start_paused)

    # drive with a timer (so services remain responsive)
    period = max(0.5, float(rospy.get_param("~tick_period", 1.0)))
    rospy.Timer(rospy.Duration(period), _tick, oneshot=False)

    # clean shutdown
    signal.signal(signal.SIGINT, _signal_handler)

    rospy.loginfo("Waypoint Player ready. Waypoints: %d | loop=%s reverse=%s start_index=%d",
                  len(waypoints), LOOP, REVERSE, START_INDEX)
    rospy.spin()
