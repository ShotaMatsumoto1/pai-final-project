"""分類結果を geometry_msgs/Twist にして Gazebo のロボットへ送る。"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from jp_command_robot.classifier import MOTION_LABELS, CommandClassifier, Prediction

CMD_VEL_TOPIC = "/model/vehicle_blue/cmd_vel"
LINEAR_SPEED = 0.4
ANGULAR_SPEED = 0.8
MOTION_SEC = 2.0
STOP_HOLD_SEC = 0.4
PUBLISH_HZ = 10

LABEL_JA = {
    "forward": "前進",
    "backward": "後退",
    "left": "左旋回",
    "right": "右旋回",
    "stop": "停止",
}


@dataclass(frozen=True)
class Motion:
    linear_x: float
    angular_z: float
    duration_sec: float


def motion_for(label: str) -> Motion:
    if label == "forward":
        return Motion(LINEAR_SPEED, 0.0, MOTION_SEC)
    if label == "backward":
        return Motion(-LINEAR_SPEED, 0.0, MOTION_SEC)
    if label == "left":
        return Motion(0.0, ANGULAR_SPEED, MOTION_SEC)
    if label == "right":
        return Motion(0.0, -ANGULAR_SPEED, MOTION_SEC)
    if label == "stop":
        return Motion(0.0, 0.0, STOP_HOLD_SEC)
    raise KeyError(label)


def make_twist(linear_x: float, angular_z: float) -> Twist:
    message = Twist()
    message.linear.x = float(linear_x)
    message.angular.z = float(angular_z)
    return message


class VelocityPublisher:
    """速度指令の送信口。テストではこのクラスを置き換える。"""

    def publish(self, linear_x: float, angular_z: float) -> None:
        raise NotImplementedError

    def sleep(self, duration_sec: float) -> None:
        time.sleep(duration_sec)


class RosVelocityPublisher(VelocityPublisher):
    def __init__(self, node: Node, topic: str) -> None:
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._publisher = node.create_publisher(Twist, topic, qos)

    def publish(self, linear_x: float, angular_z: float) -> None:
        self._publisher.publish(make_twist(linear_x, angular_z))


def publish_for(
    publisher: VelocityPublisher,
    linear_x: float,
    angular_z: float,
    duration_sec: float,
    cancel: threading.Event | None = None,
    rate_hz: float = PUBLISH_HZ,
) -> None:
    period = 1.0 / rate_hz
    elapsed = 0.0
    while elapsed < duration_sec:
        if cancel is not None and cancel.is_set():
            return
        publisher.publish(linear_x, angular_z)
        step = min(period, duration_sec - elapsed)
        publisher.sleep(step)
        elapsed += step


def send_stop(publisher: VelocityPublisher, rate_hz: float = PUBLISH_HZ) -> None:
    publish_for(publisher, 0.0, 0.0, STOP_HOLD_SEC, cancel=None, rate_hz=rate_hz)


def execute_prediction(
    publisher: VelocityPublisher,
    prediction: Prediction,
    cancel: threading.Event | None = None,
) -> str:
    if cancel is None:
        cancel = threading.Event()
    if not prediction.accepted or prediction.label not in MOTION_LABELS:
        return "拒否したため速度指令は送っていません。"

    motion = motion_for(prediction.label)
    publish_for(
        publisher,
        motion.linear_x,
        motion.angular_z,
        motion.duration_sec,
        cancel,
    )
    send_stop(publisher)
    if cancel.is_set():
        return "停止命令を割り込ませ、停止指令を送りました。"
    return "動作後に停止指令を送りました。"


def format_prediction(prediction: Prediction) -> str:
    score = f"出力スコア={prediction.score:.4f} マージン={prediction.margin:.4f}"
    if not prediction.accepted or prediction.label is None:
        return f"拒否: {prediction.reason}（{score}）"
    return f"分類: {prediction.label} / {LABEL_JA[prediction.label]}（{score}）"


class CommandRobot(Node):
    def __init__(self, classifier: CommandClassifier, topic: str = CMD_VEL_TOPIC) -> None:
        super().__init__("jp_command_robot")
        self.classifier = classifier
        self.topic = topic
        self.publisher = RosVelocityPublisher(self, topic)
        self.cancel = threading.Event()
        self._motion_active = threading.Event()
        self._worker: threading.Thread | None = None

    def stop(self) -> None:
        self.cancel.set()
        self.publisher.publish(0.0, 0.0)

    def wait_until_idle(self, timeout_sec: float = MOTION_SEC + 2.0) -> None:
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout_sec)

    def handle_line(self, line: str) -> str:
        prediction = self.classifier.predict(line)
        message = format_prediction(prediction)
        if self._motion_active.is_set():
            if prediction.accepted and prediction.label == "stop":
                self.cancel.set()
                self.publisher.publish(0.0, 0.0)
                return message + " 動作を中断します。"
            return message + " 動作中のため、この命令は実行しません。"
        if not prediction.accepted:
            return message

        self.cancel = threading.Event()
        cancel = self.cancel
        self._motion_active.set()

        def run() -> None:
            try:
                print(execute_prediction(self.publisher, prediction, cancel))
            except Exception as exc:  # noqa: BLE001
                print(f"エラーのため停止指令を送りました: {exc}", file=sys.stderr)
            finally:
                self.stop()
                self._motion_active.clear()

        self._worker = threading.Thread(target=run, daemon=True)
        self._worker.start()
        return message

    def run_and_wait(self, line: str) -> str:
        prediction = self.classifier.predict(line)
        message = format_prediction(prediction)
        print(message)
        if not prediction.accepted:
            return message
        self.cancel = threading.Event()
        try:
            outcome = execute_prediction(self.publisher, prediction, self.cancel)
        finally:
            self.stop()
        print(outcome)
        return f"{message}\n{outcome}"


def _stdin_loop(robot: CommandRobot, done: threading.Event) -> None:
    print("日本語の命令を入力してください。終了は exit です。")
    print(f"速度指令トピック: {robot.topic}")
    print("動作中に「止まれ」と入力すると、その動作を中断します。")
    while not done.is_set():
        try:
            line = input("> ")
        except EOFError:
            done.set()
            robot.stop()
            return
        command = line.strip()
        if command in {"exit", "quit", "終了"}:
            done.set()
            robot.stop()
            return
        try:
            print(robot.handle_line(command))
        except Exception as exc:  # noqa: BLE001
            robot.stop()
            print(f"エラーのため停止指令を送りました: {exc}", file=sys.stderr)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="日本語命令で vehicle_blue を動かす")
    parser.add_argument(
        "--once",
        action="append",
        default=[],
        help="対話せず、この命令を順に実行する。複数指定できる。",
    )
    parser.add_argument("--topic", default=CMD_VEL_TOPIC, help="速度指令トピック")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rclpy.init()
    robot = None
    spin_thread = None
    done = threading.Event()
    try:
        classifier = CommandClassifier.from_data_dir()
        robot = CommandRobot(classifier, topic=args.topic)
        spin_thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
        spin_thread.start()
        print(
            "閾値: "
            f"score>{classifier.thresholds.min_score:.4f}, "
            f"margin>{classifier.thresholds.min_margin:.4f} "
            f"({classifier.thresholds.rule})"
        )
        if args.once:
            for command in args.once:
                print(f"> {command}")
                robot.run_and_wait(command)
        else:
            _stdin_loop(robot, done)
    except KeyboardInterrupt:
        print("\n中断しました。")
    finally:
        done.set()
        if robot is not None:
            robot.stop()
            robot.wait_until_idle()
            robot.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
