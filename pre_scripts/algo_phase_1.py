#!/usr/bin/env python

import rospy
import actionlib
import numpy as np
import tf
import math
import threading
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker

class CostmapCoverageExplorer:
    def __init__(self):
        rospy.init_node('turtlebot3_costmap_coverage_explorer')

        self.raw_map = None
        self.global_costmap = None
        self.local_costmap = None
        self.merged_map = None

        self.persistent_merged = None

        self.map_pub = rospy.Publisher("/merged_map", OccupancyGrid, queue_size=1)
        self.coverage_pub = rospy.Publisher("/coverage_marker", Marker, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.listener = tf.TransformListener()

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.global_costmap_callback)
        rospy.Subscriber("/move_base/local_costmap/costmap", OccupancyGrid, self.local_costmap_callback)

        self.covered_points = []
        self.cover_radius = 0.6
        self.obstacle_buffer = 0.2
        self.max_failures = 10

        self.lock = threading.Lock()
        self.merge_thread = threading.Thread(target=self.publish_merged_map_loop)
        self.merge_thread.daemon = True
        self.merge_thread.start()

        self.tracking_thread = threading.Thread(target=self.track_robot_pose_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()

        rospy.loginfo("Waiting for move_base server...")
        self.move_base.wait_for_server()
        rospy.loginfo("Connected to move_base")

    def map_callback(self, msg):
        with self.lock:
            self.raw_map = msg

    def global_costmap_callback(self, msg):
        with self.lock:
            self.global_costmap = msg

    def local_costmap_callback(self, msg):
        with self.lock:
            self.local_costmap = msg

    def publish_merged_map_loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            with self.lock:
                self.try_merge_maps()
            rate.sleep()

    def try_merge_maps(self):
        if not self.raw_map or not self.global_costmap or not self.local_costmap:
            return

        w, h = self.raw_map.info.width, self.raw_map.info.height
        res = self.raw_map.info.resolution
        origin = self.raw_map.info.origin.position

        map_np = np.array(self.raw_map.data).reshape((h, w))
        base_layer = np.full_like(map_np, -1)
        for y in range(h):
            for x in range(w):
                val = map_np[y][x]
                if val == 0:
                    base_layer[y][x] = 0
                elif 1 <= val <= 99 or val >= 100:
                    base_layer[y][x] = 100

        if self.persistent_merged is None:
            self.persistent_merged = base_layer.copy()
        else:
            for y in range(h):
                for x in range(w):
                    if base_layer[y][x] == 0 and self.persistent_merged[y][x] != 100:
                        self.persistent_merged[y][x] = 0
                    elif base_layer[y][x] == -1 and self.persistent_merged[y][x] != 100:
                        self.persistent_merged[y][x] = -1

        def overlay_costmap(costmap):
            cw, ch = costmap.info.width, costmap.info.height
            cres = costmap.info.resolution
            corigin = costmap.info.origin.position
            data = np.array(costmap.data).reshape((ch, cw))
            for y in range(ch):
                for x in range(cw):
                    val = data[y][x]
                    if val > 50:
                        wx = corigin.x + x * cres
                        wy = corigin.y + y * cres
                        mx = int((wx - origin.x) / res)
                        my = int((wy - origin.y) / res)
                        if 0 <= mx < w and 0 <= my < h:
                            self.persistent_merged[my][mx] = 100

        overlay_costmap(self.global_costmap)
        overlay_costmap(self.local_costmap)

        if self.covered_points:
            radius_cells = int(0.6 / res)
            for cx, cy in self.covered_points:
                robot_map_x = int((cx - origin.x) / res)
                robot_map_y = int((cy - origin.y) / res)
                for dy in range(-radius_cells, radius_cells + 1):
                    for dx in range(-radius_cells, radius_cells + 1):
                        if math.hypot(dx, dy) > radius_cells:
                            continue
                        nx = robot_map_x + dx
                        ny = robot_map_y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            if self.persistent_merged[ny][nx] == 0:
                                self.persistent_merged[ny][nx] = 50  # Yellow for trail

        self.merged_map = self.persistent_merged.copy()
        out = OccupancyGrid()
        out.header = self.raw_map.header
        out.info = self.raw_map.info
        out.data = self.merged_map.flatten().tolist()
        self.map_pub.publish(out)

    def track_robot_pose_loop(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            try:
                (trans, rot) = self.listener.lookupTransform("map", "base_link", rospy.Time(0))
                self.covered_points.append((trans[0], trans[1]))
            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                pass
            rate.sleep()


    def mark_coverage(self, x, y):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.scale.x = marker.scale.y = marker.scale.z = self.cover_radius
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.05
        marker.pose.orientation.w = 1.0
        marker.id = len(self.covered_points)
        self.coverage_pub.publish(marker)

    def mark_as_covered_in_map(self, cx, cy, radius=0.3):
        if self.merged_map is None:
            return

        res = self.raw_map.info.resolution
        ox, oy = self.raw_map.info.origin.position.x, self.raw_map.info.origin.position.y
        w, h = self.raw_map.info.width, self.raw_map.info.height

        mx = int((cx - ox) / res)
        my = int((cy - oy) / res)
        cell_radius = int(radius / res)

        for dy in range(-cell_radius, cell_radius + 1):
            for dx in range(-cell_radius, cell_radius + 1):
                dist = math.hypot(dx, dy) * res
                if dist <= radius:
                    nx = mx + dx
                    ny = my + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        if self.merged_map[ny][nx] == 0:
                            self.merged_map[ny][nx] = -2

    def rotate_in_place(self, duration=3):
        twist = Twist()
        twist.angular.z = 0.5
        start = rospy.Time.now()
        r = rospy.Rate(10)
        while (rospy.Time.now() - start).to_sec() < duration:
            self.cmd_vel_pub.publish(twist)
            r.sleep()
        self.cmd_vel_pub.publish(Twist())

    def get_robot_pose(self):
        try:
            (trans, rot) = self.listener.lookupTransform("/map", "/base_footprint", rospy.Time(0))
            x, y = trans[0], trans[1]
            _, _, yaw = tf.transformations.euler_from_quaternion(rot)
            return x, y, yaw
        except:
            return None

    def angle_diff(self, a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def is_far_from_obstacles(self, mx, my):
        buffer_cells = int(self.obstacle_buffer / self.raw_map.info.resolution)
        h, w = self.merged_map.shape
        for dy in range(-buffer_cells, buffer_cells + 1):
            for dx in range(-buffer_cells, buffer_cells + 1):
                nx, ny = mx + dx, my + dy
                if 0 <= nx < w and 0 <= ny < h:
                    if self.merged_map[ny][nx] == 100:
                        return False
        return True

    def find_next_goal(self, rx, ry, yaw):
        res = self.raw_map.info.resolution
        ox, oy = self.raw_map.info.origin.position.x, self.raw_map.info.origin.position.y
        w, h = self.raw_map.info.width, self.raw_map.info.height
        robot_map_x = int((rx - ox) / res)
        robot_map_y = int((ry - oy) / res)

        max_radius = 5.0
        step = 1.0

        for radius in np.arange(step, max_radius + step, step):
            candidates = []
            search_range = int(radius / res)

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    dist = math.hypot(dx, dy) * res
                    if dist > radius:
                        continue

                    x = robot_map_x + dx
                    y = robot_map_y + dy

                    if not (0 <= x < w and 0 <= y < h):
                        continue

                    if self.merged_map[y][x] != 0:
                        continue

                    wx = ox + x * res
                    wy = oy + y * res

                    if any(math.hypot(wx - cx, wy - cy) < self.cover_radius for cx, cy in self.covered_points):
                        continue

                    if not self.is_far_from_obstacles(x, y):
                        continue

                    angle = math.atan2(dy, dx)
                    front_score = math.cos(self.angle_diff(yaw, angle))
                    candidates.append((front_score, dist, wx, wy))

            if candidates:
                candidates.sort(key=lambda c: (-c[0], c[1]))
                return (candidates[0][2], candidates[0][3])

        return None
    
    def send_goal(self, x, y, rx, ry):
        angle = math.atan2(y - ry, x - rx)
        q = tf.transformations.quaternion_from_euler(0, 0, angle)

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]

        self.move_base.send_goal(goal)
        return self.move_base.wait_for_result(rospy.Duration(30))


    def send_goal_(self, x, y):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.w = 1.0

        self.move_base.send_goal(goal)
        return self.move_base.wait_for_result(rospy.Duration(30))

    def run(self):
        rospy.sleep(2)
        self.rotate_in_place()
        rate = rospy.Rate(1)
        failures = 0

        while not rospy.is_shutdown():
            pose = self.get_robot_pose()
            if not pose:
                rospy.logwarn("Robot pose not available.")
                continue

            rx, ry, yaw = pose
            goal = self.find_next_goal(rx, ry, yaw)

            if goal:
                rospy.loginfo("Goal: %s", goal)
                # success = self.send_goal(*goal)
                success = self.send_goal(goal[0], goal[1], rx, ry)

                if success:
                    rospy.loginfo("Goal reached.")
                    self.covered_points.append(goal)
                    self.mark_coverage(*goal)
                    self.mark_as_covered_in_map(*goal, radius=0.3)
                    self.rotate_in_place()
                    failures = 0
                else:
                    rospy.logwarn("Goal failed.")
                    failures += 1
            else:
                rospy.loginfo("No more goals.")
                failures += 1

            if failures >= self.max_failures:
                rospy.loginfo("Exploration complete or stuck.")
                break

            rate.sleep()

        rospy.loginfo("Stopping.")
        self.cmd_vel_pub.publish(Twist())

if __name__ == '__main__':
    try:
        CostmapCoverageExplorer().run()
    except rospy.ROSInterruptException:
        pass
