#!/usr/bin/env bash
# Gazebo の diff_drive が既に動いているとき、青いロボットの橋だけを起動する。
set -euo pipefail

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy が見つかりません: /opt/ros/jazzy/setup.bash" >&2
  exit 1
fi

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
exec ros2 run ros_gz_bridge parameter_bridge \
  /model/vehicle_blue/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist \
  /model/vehicle_blue/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry \
  --ros-args -p qos_overrides./model/vehicle_blue.subscriber.reliability:=reliable
