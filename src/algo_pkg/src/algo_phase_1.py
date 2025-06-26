#!/usr/bin/env python

import rospy
import actionlib
import numpy as np
import tf
import math
import threading
import roslaunch
from nav_msgs.msg import OccupancyGrid
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class CostmapCoverageExplorer:
    def __init__(self):
        rospy.init_node('turtlebot3_costmap_coverage_explorer')

        self.raw_map = None
        self.global_costmap = None
        self.local_costmap = None
        self.merged_map = None

        self.dose_map_latest = None
        self.dose_map_lock = threading.Lock()

        self.dose_map = None
        rospy.Subscriber("/dose_map", OccupancyGrid, self.dose_map_callback)

        self.persistent_merged = None

        self.map_pub = rospy.Publisher("/merged_map", OccupancyGrid, queue_size=1)
        self.coverage_pub = rospy.Publisher("/coverage_marker", Marker, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        self.path_marker_pub = rospy.Publisher("/robot_path_marker", Marker, queue_size=10)

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
        self.launch_additional_file('algo_pkg', 'master_dose.launch')

    def launch_additional_file(self, package, launch_filename):
        try:
            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)

            launch_file = roslaunch.rlutil.resolve_launch_arguments(
                [package, launch_filename]
            )[0]

            self.launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_file])
            self.launch.start()

            rospy.on_shutdown(self.launch.shutdown)  # Optional: stop it on shutdown
            rospy.loginfo("Successfully launched %s", launch_filename)
        except Exception as e:
            rospy.logerr("Failed to launch %s from %s: %s", launch_filename, package, str(e))

    def map_callback(self, msg):
        with self.lock:
            self.raw_map = msg

    def global_costmap_callback(self, msg):
        with self.lock:
            self.global_costmap = msg

    def dose_map_callback(self, msg):
        # global dose_map_latest
        with self.dose_map_lock:
            self.dose_map_latest = msg
            print("Dose map received..>>>>>>>>>>>>>>>>>>>>>.")


    def count_trail_color1_cells(self):
        with self.dose_map_lock:
            if self.dose_map_latest is None:
                rospy.logwarn("Dose map not received yet.")
                return

        data = np.array(self.dose_map_latest.data, dtype=np.int8)
        count = np.sum(data == -2)
        rospy.loginfo(f"Total cells with dosage level TRAIL_COLOR1 (-2): {count}")


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
                self.publish_path_marker()
            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                pass
            rate.sleep()

    def visit_trail_color_cells(self):
        rospy.sleep(1.0)
        with self.dose_map_lock:
            if self.dose_map_latest is None:
                rospy.logwarn("Dose map not available.")
                return

            res = self.dose_map_latest.info.resolution
            ox = self.dose_map_latest.info.origin.position.x
            oy = self.dose_map_latest.info.origin.position.y
            w = self.dose_map_latest.info.width
            h = self.dose_map_latest.info.height

            data = np.array(self.dose_map_latest.data, dtype=np.int8).reshape((h, w))

            trail_cells = []
            for y in range(h):
                for x in range(w):
                    if data[y][x] == -2:
                        wx = ox + x * res
                        wy = oy + y * res
                        trail_cells.append((wx, wy))

        if not trail_cells:
            rospy.loginfo("No trail color (-2) cells found to visit.")
            return

        # Cluster trail points
        rospy.loginfo(f"Found {len(trail_cells)} trail-colored cells. Clustering...")
        clusters = []
        cluster_radius = 0.5
        for pt in trail_cells:
            assigned = False
            for cluster in clusters:
                if any(math.hypot(pt[0] - c[0], pt[1] - c[1]) < cluster_radius for c in cluster):
                    cluster.append(pt)
                    assigned = True
                    break
            if not assigned:
                clusters.append([pt])

        rospy.loginfo(f"Visiting {len(clusters)} clustered trail regions...")

        for i, cluster in enumerate(clusters):
            goal_pt = cluster[len(cluster) // 2]
            pose = self.get_robot_pose()
            if not pose:
                rospy.logwarn(f"[Trail Visit] Robot pose not available. Skipping cluster {i+1}")
                continue

            rx, ry, _ = pose
            rospy.loginfo(f"[Trail Visit] Moving to trail cluster {i+1}/{len(clusters)} at {goal_pt}")
            success = self.send_goal(goal_pt[0], goal_pt[1], rx, ry)

            if success:
                rospy.loginfo(f"[Trail Visit] Arrived at cluster {i+1}, waiting 5 seconds...")
                rospy.sleep(5.0)
            else:
                rospy.logwarn(f"[Trail Visit] Failed to reach cluster {i+1}, skipping.")


    def publish_path_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "robot_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # Line width

        marker.color.r = 0.0
        marker.color.g = 1.0  # Green
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.pose.orientation.w = 1.0

        for pt in self.covered_points:
            p = Point()
            p.x = pt[0]
            p.y = pt[1]
            p.z = 0.05
            marker.points.append(p)

        self.path_marker_pub.publish(marker)


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

        self.last_finder()
        rospy.loginfo("✅ All coverage done.")

        # Delay to ensure last dose_map is received
        rospy.sleep(1.0)
        self.count_trail_color1_cells()

        self.visit_trail_color_cells()

        rospy.loginfo("🚩 Trail region visiting complete.")
        # res = self.raw_map.info.resolution
        # ox, oy = self.raw_map.info.origin.position.x, self.raw_map.info.origin.position.y
        # w, h = self.raw_map.info.width, self.raw_map.info.height

        # unvisited_free_cells = []

        # for y in range(h):
        #     for x in range(w):
        #         if self.merged_map[y][x] != 0:
        #             continue  # Not free space

        #         wx = ox + x * res
        #         wy = oy + y * res

        #         if any(math.hypot(wx - cx, wy - cy) < self.cover_radius for cx, cy in self.covered_points):
        #             continue  # Already covered

        #         unvisited_free_cells.append((wx, wy))

        # rospy.loginfo(f"Remaining uncovered free cells: {len(unvisited_free_cells)}")

        # # Optionally print some of them
        # for i, pt in enumerate(unvisited_free_cells[:20]):
        #     rospy.loginfo(f"Unvisited cell {i + 1}: x={pt[0]:.2f}, y={pt[1]:.2f}")
        rospy.loginfo("Stopping.")
        self.cmd_vel_pub.publish(Twist())

    def last_finder(self):

        # Final pass to cover remaining uncovered cells
        res = self.raw_map.info.resolution
        ox, oy = self.raw_map.info.origin.position.x, self.raw_map.info.origin.position.y
        w, h = self.raw_map.info.width, self.raw_map.info.height

        unvisited_free_cells = []

        for y in range(h):
            for x in range(w):
                if self.merged_map[y][x] != 0:
                    continue

                wx = ox + x * res
                wy = oy + y * res

                if any(math.hypot(wx - cx, wy - cy) < self.cover_radius for cx, cy in self.covered_points):
                    continue

                unvisited_free_cells.append((wx, wy))

        if not unvisited_free_cells:
            rospy.loginfo("[Final Pass] No uncovered cells left.")
            return

        rospy.loginfo(f"[Final Pass] Remaining uncovered free cells: {len(unvisited_free_cells)}")

        # Cluster remaining points
        cluster_radius = 0.5
        clusters = []

        for pt in unvisited_free_cells:
            assigned = False
            for cluster in clusters:
                if any(math.hypot(pt[0] - c[0], pt[1] - c[1]) < cluster_radius for c in cluster):
                    cluster.append(pt)
                    assigned = True
                    break
            if not assigned:
                clusters.append([pt])

        rospy.loginfo(f"[Final Pass] Clustered into {len(clusters)} regions")

        # Attempt to visit one point from each cluster
        for i, cluster in enumerate(clusters):
            goal_pt = cluster[len(cluster)//2]  # Use middle point for robustness
            rospy.loginfo(f"[Final Pass] Attempting cluster {i+1}/{len(clusters)} at {goal_pt}")

            pose = self.get_robot_pose()
            if not pose:
                rospy.logwarn("[Final Pass] Robot pose not available. Skipping cluster.")
                continue

            rx, ry, _ = pose
            success = self.send_goal(goal_pt[0], goal_pt[1], rx, ry)

            if success:
                rospy.loginfo(f"[Final Pass] Cluster {i+1} goal reached.")
                self.covered_points.append(goal_pt)
                self.mark_coverage(*goal_pt)
                self.mark_as_covered_in_map(*goal_pt, radius=0.3)
                self.rotate_in_place()
            else:
                rospy.logwarn(f"[Final Pass] Failed to reach cluster {i+1}. Moving to next cluster.")

        # Continue coverage for remaining uncovered free cells
        # res = self.raw_map.info.resolution
        # ox, oy = self.raw_map.info.origin.position.x, self.raw_map.info.origin.position.y
        # w, h = self.raw_map.info.width, self.raw_map.info.height

        # unvisited_free_cells = []

        # for y in range(h):
        #     for x in range(w):
        #         if self.merged_map[y][x] != 0:
        #             continue  # Not free space

        #         wx = ox + x * res
        #         wy = oy + y * res

        #         if any(math.hypot(wx - cx, wy - cy) < self.cover_radius for cx, cy in self.covered_points):
        #             continue  # Already covered

        #         unvisited_free_cells.append((wx, wy))

        # rospy.loginfo(f"[Final Pass] Remaining uncovered free cells: {len(unvisited_free_cells)}")

        # # Cluster unvisited cells (simple grid-based grouping)
        # cluster_radius = 0.5  # meters
        # clusters = []

        # for pt in unvisited_free_cells:
        #     assigned = False
        #     for cluster in clusters:
        #         if any(math.hypot(pt[0] - c[0], pt[1] - c[1]) < cluster_radius for c in cluster):
        #             cluster.append(pt)
        #             assigned = True
        #             break
        #     if not assigned:
        #         clusters.append([pt])

        # rospy.loginfo(f"[Final Pass] Clustered into {len(clusters)} regions")

        # # Try to send one goal per cluster
        # for i, cluster in enumerate(clusters):
        #     goal_pt = cluster[len(cluster)//2]  # take middle point
        #     rospy.loginfo(f"[Final Pass] Visiting cluster {i+1} at {goal_pt}")
            
        #     rx, ry, _ = self.get_robot_pose()
        #     success = self.send_goal(goal_pt[0], goal_pt[1], rx, ry)
            
        #     if success:
        #         rospy.loginfo(f"[Final Pass] Cluster {i+1} covered.")
        #         self.covered_points.append(goal_pt)
        #         self.mark_coverage(*goal_pt)
        #         self.mark_as_covered_in_map(*goal_pt, radius=0.3)
        #         self.rotate_in_place()
        #     else:
        #         rospy.logwarn(f"[Final Pass] Failed to reach cluster {i+1}")


if __name__ == '__main__':
    try:
        CostmapCoverageExplorer().run()
    except rospy.ROSInterruptException:
        pass