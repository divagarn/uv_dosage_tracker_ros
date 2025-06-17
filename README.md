# uv_dosage_tracker_ros & ROS Trail Persistence and Dosage Mapping

This ROS node tracks and visualizes a robot's movement trail on a map by simulating dosage accumulation (e.g., UV disinfection exposure). The accumulated dosage is encoded into the occupancy grid using colored values, allowing for easy trail visualization in RViz.

---

##  Overview

This package overlays a persistent trail on the robot's occupancy grid. As the robot moves, each cell under its footprint accumulates a dosage value over time. The trail is colored based on dosage levels, representing how long the robot has stayed in any location.

---

##  Key Features

- Persistent trail tracking using `OccupancyGrid`
- Real-time dosage simulation based on robot footprint and duration
- Trail coloring based on dosage thresholds
- Black outline to highlight the current robot footprint
- ROS TF integration to get robot's current position
- Publishes modified map to `/map_modified`

---

##  Dependencies

- ROS Noetic
- `rospy`
- `nav_msgs/OccupancyGrid`
- `tf`
- `numpy`

---



