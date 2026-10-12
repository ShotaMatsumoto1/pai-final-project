
#!/usr/bin/env bash
# 公式の差動二輪デモを起動する。
# Gazeboとcmd_velブリッジを起動する。

set -euo pipefail

# ROS 2のインストール確認
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy が見つかりません: /opt/ros/jazzy/setup.bash" >&2
  exit 1
fi

# ROS 2の環境設定で未定義変数エラーを防止
set +u

# ROS 2の環境を読み込む
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

# 未定義変数チェックを再度有効化
set -u

# Gazeboの差動二輪ロボットを起動
# RVizはデフォルトで無効
exec ros2 launch ros_gz_sim_demos diff_drive.launch.py rviz:="${RVIZ:-false}"
