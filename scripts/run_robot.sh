#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

# ROS 2のインストール確認
if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy が見つかりません: /opt/ros/jazzy/setup.bash" >&2
  exit 1
fi

# Python仮想環境の確認
if [[ ! -f .venv/bin/activate ]]; then
  echo "仮想環境がありません。README の手順で .venv を作成してください。" >&2
  exit 1
fi

# ROS 2の環境設定で未定義変数エラーを防止
set +u

# ROS 2の環境を読み込む
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash

# Python仮想環境を有効化
# shellcheck disable=SC1091
source .venv/bin/activate

# 未定義変数チェックを再度有効化
set -u

# AIロボット制御プログラムを起動
exec python -m jp_command_robot "$@"