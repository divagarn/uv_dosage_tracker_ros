#!/usr/bin/env python

import rospy
import numpy as np
import tf
import threading
import time
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

TRAIL_COLOR1 = -2    
TRAIL_COLOR2 = -50  
TRAIL_COLOR3 = -100  
TRAIL_COLOR4 = -150  
TRAIL_UNKNOWN = -1  

robot_radius = 0.60  

# Global state
persistent_map = None    # Base occupancy values
persistent_trail = None  # Trail overlay
dosage_map = None        # Dosage accumulation per cell
map_msg_latest = None
lock = threading.Lock()

dosage_value = 0.41      # Dosage applied per second
target_rate = 1.0        # Hz
last_update_time = None
origin_x = origin_y = 0.0
resolution = None

# Threshold mapping based on disinfect_type
disinfect_type = rospy.get_param('disinfect_type', 0)
threshold_map = {0: 10, 1: 20, 2: 30, 3: 40}
threshold = threshold_map.get(disinfect_type, 70)

# Calculate segments
s1 = round(threshold * 0.25, 2)
s2 = round(threshold * 0.50, 2)
s3 = round(threshold * 0.75, 2)
s4 = threshold

def get_robot_position(listener, target_frame="base_link", reference_frame="map"):
    try:
        (trans, _) = listener.lookupTransform(reference_frame, target_frame, rospy.Time(0))
        return trans[:2]
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logwarn("TF lookup failed for robot position")
        return None
    
def map_callback(msg, source="global"):
    global persistent_map, persistent_trail, dosage_map, map_msg_latest, resolution, origin_x, origin_y
    with lock:
        h, w = msg.info.height, msg.info.width
        res = msg.info.resolution
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y

        data = np.array(msg.data, dtype=np.int8).reshape((h, w))

        # Only initialize once
        if persistent_map is None:
            persistent_map = data.copy()
            persistent_trail = np.full((h, w), TRAIL_UNKNOWN, dtype=np.int8)
            dosage_map = np.zeros((h, w), dtype=np.float32)
            resolution = res
            origin_x = ox
            origin_y = oy
        else:
            if (h, w) != persistent_map.shape:
                rospy.logwarn(f"[{source}] Skipped update due to mismatched shape")
                return

            # Merge unknown trail regions only
            mask = (persistent_trail == TRAIL_UNKNOWN)
            persistent_map[mask] = data[mask]

            # Always preserve obstacles
            persistent_map[data == 100] = 100

        map_msg_latest = msg


# def map_callback(msg):
#     global persistent_map, persistent_trail, dosage_map, map_msg_latest, resolution, origin_x, origin_y
#     with lock:
#         h, w = msg.info.height, msg.info.width
#         resolution = msg.info.resolution
#         origin_x = msg.info.origin.position.x
#         origin_y = msg.info.origin.position.y

#         data = np.array(msg.data, dtype=np.int8).reshape((h, w))
#         if persistent_map is None:
#             persistent_map = data.copy()
#             persistent_trail = np.full((h, w), TRAIL_UNKNOWN, dtype=np.int8)
#             dosage_map = np.zeros((h, w), dtype=np.float32)
#         else:
#             mask = (persistent_trail == TRAIL_UNKNOWN)
#             persistent_map[mask] = data[mask]
#             persistent_map[data == 100] = 100

#         map_msg_latest = msg

def process_map():
    global last_update_time
    listener = tf.TransformListener()
    rate = rospy.Rate(target_rate)
    last_update_time = time.time()

    while not rospy.is_shutdown():
        with lock:
            msg = map_msg_latest
            if persistent_map is not None:
                base_map = persistent_map.copy()
            else:
                base_map = None
        if msg is None or base_map is None:
            rate.sleep()
            continue

        # elapsed
        now = time.time()
        dt = now - last_update_time
        last_update_time = now

        # robot pos
        try:
            listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(1.0))
            pos = get_robot_position(listener)
        except tf.Exception as e:
            rospy.logwarn("TF exception: {}".format(e))
            rate.sleep()
            continue

        if pos is None:
            rate.sleep()
            continue
        rx, ry = pos

        # grid center
        cx = int((rx - origin_x) / resolution)
        cy = int((ry - origin_y) / resolution)
        r_cells = int(robot_radius / resolution)

        # collect footprint and outline
        footprint = []
        outline = []
        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                dist2 = dx*dx + dy*dy
                if dist2 <= r_cells*r_cells:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < base_map.shape[1] and 0 <= y < base_map.shape[0]:
                        if base_map[y, x] >= 0 and base_map[y, x] < 100:
                            footprint.append((x, y))
                            # check circumference band for outline
                            if dist2 >= (r_cells - 1)*(r_cells - 1):
                                outline.append((x, y))

        # update dosage for footprint
        with lock:
            for x, y in footprint:
                dosage_map[y, x] += dosage_value * dt
                current_dosage = dosage_map[y, x]

                if current_dosage >= s3:
                    persistent_trail[y, x] = TRAIL_COLOR4
                elif current_dosage >= s2:
                    persistent_trail[y, x] = TRAIL_COLOR3
                elif current_dosage >= s1:
                    persistent_trail[y, x] = TRAIL_COLOR2
                else:
                    persistent_trail[y, x] = TRAIL_COLOR1

            # combine map and trail
            combined = base_map.copy()
            mask = (persistent_trail != TRAIL_UNKNOWN)
            combined[mask] = persistent_trail[mask]

            # draw black outline (occupied=100) around footprint
            for x, y in outline:
                combined[y, x] = 100

            # publish
            out = OccupancyGrid()
            out.header = Header()
            out.header.stamp = rospy.Time.now()
            out.header.frame_id = msg.header.frame_id
            out.info = msg.info
            out.data = tuple(combined.flatten())
            map_pub.publish(out)

        rate.sleep()

# if __name__ == '__main__':
#     rospy.init_node('map_trail_persistence')
#     map_pub = rospy.Publisher('/dose_map', OccupancyGrid, queue_size=10)
#     rospy.Subscriber('/move_base/global_costmap/costmap', OccupancyGrid, map_callback)

#     # Display parameters
#     rospy.loginfo(f"Dosage value per second: {dosage_value}")
#     rospy.loginfo(f"Disinfect type: {disinfect_type}")
#     rospy.loginfo(f"Threshold: {threshold}")

#     thread = threading.Thread(target=process_map)
#     thread.daemon = True
#     thread.start()
#     rospy.spin()


if __name__ == '__main__':
    rospy.init_node('map_trail_persistence')
    map_pub = rospy.Publisher('/dose_map', OccupancyGrid, queue_size=10)

    rospy.Subscriber('/move_base/global_costmap/costmap',
                     OccupancyGrid,
                     lambda msg: map_callback(msg, source="global"))

    rospy.Subscriber('/move_base/local_costmap/costmap',
                     OccupancyGrid,
                     lambda msg: map_callback(msg, source="local"))

    rospy.Subscriber('/merged_map',
                     OccupancyGrid,
                     lambda msg: map_callback(msg, source="merged"))

    # Log parameters
    rospy.loginfo(f"Dosage value per second: {dosage_value}")
    rospy.loginfo(f"Disinfect type: {disinfect_type}")
    rospy.loginfo(f"Threshold: {threshold}")

    thread = threading.Thread(target=process_map)
    thread.daemon = True
    thread.start()
    rospy.spin()

