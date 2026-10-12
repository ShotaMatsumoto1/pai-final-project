"""起動中の Gazebo に速度指令を送り、オドメトリの変化で動作を確認する。"""

from __future__ import annotations

import math
import threading
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from jp_command_robot.classifier import CommandClassifier
from jp_command_robot.controller import (
    CMD_VEL_TOPIC,
    RosVelocityPublisher,
    execute_prediction,
    format_prediction,
)

ODOM_TOPIC = "/model/vehicle_blue/odometry"


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class OdomProbe(Node):
    def __init__(self) -> None:
        super().__init__("jp_command_odom_probe")
        self._condition = threading.Condition()
        self._latest: Odometry | None = None
        self._count = 0
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, qos)
        self.publisher = RosVelocityPublisher(self, CMD_VEL_TOPIC)

    def _on_odom(self, message: Odometry) -> None:
        with self._condition:
            self._latest = message
            self._count += 1
            self._condition.notify_all()

    def wait_pose(self, previous_count: int | None = None, timeout_sec: float = 5.0) -> tuple[float, float, float, int]:
        deadline = time.monotonic() + timeout_sec
        with self._condition:
            while True:
                if self._latest is not None and (previous_count is None or self._count > previous_count):
                    pose = self._latest.pose.pose
                    yaw = yaw_from_quaternion(
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    )
                    return pose.position.x, pose.position.y, yaw, self._count
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("オドメトリを受信できませんでした。")
                self._condition.wait(timeout=remaining)

    def stop(self) -> None:
        self.publisher.publish(0.0, 0.0)


def forward_distance(x0: float, y0: float, yaw0: float, x1: float, y1: float) -> float:
    return (x1 - x0) * math.cos(yaw0) + (y1 - y0) * math.sin(yaw0)


def main() -> None:
    rclpy.init()
    probe = OdomProbe()
    classifier = CommandClassifier.from_data_dir()
    spin = threading.Thread(target=rclpy.spin, args=(probe,), daemon=True)
    spin.start()
    failures: list[str] = []
    try:
        def run_command(text: str) -> None:
            prediction = classifier.predict(text)
            print(format_prediction(prediction))
            print(execute_prediction(probe.publisher, prediction))

        def settled_pose() -> tuple[float, float, float, int]:
            # 公式SDFのオドメトリは 1 Hz なので、停止後に次の値を待つ。
            time.sleep(1.3)
            return probe.wait_pose()

        def check(name: str, ok: bool, detail: str) -> None:
            mark = "OK" if ok else "NG"
            print(f"{mark} {name}: {detail}")
            if not ok:
                failures.append(name)

        x0, y0, yaw0, _count = settled_pose()
        print(f"初期姿勢: x={x0:.3f} y={y0:.3f} yaw={yaw0:.3f}")

        run_command("前に進め")
        x1, y1, yaw1, count1 = settled_pose()
        distance = forward_distance(x0, y0, yaw0, x1, y1)
        check("forward", distance > 0.25, f"前方変位={distance:.3f} m")

        x2, y2, yaw2, _count2 = probe.wait_pose(count1, timeout_sec=3.0)
        coast = abs(forward_distance(x1, y1, yaw1, x2, y2))
        check("stop_after_forward", coast < 0.15, f"停止後の変位={coast:.3f} m")

        run_command("後ろに下がれ")
        x3, y3, yaw3, _count3 = settled_pose()
        distance = forward_distance(x2, y2, yaw2, x3, y3)
        check("backward", distance < -0.25, f"前方変位={distance:.3f} m")

        run_command("左に曲がって")
        _x4, _y4, yaw4, _count4 = settled_pose()
        turned = wrap_angle(yaw4 - yaw3)
        check("left", turned > 0.3, f"yaw変化={turned:.3f} rad")

        run_command("右折して")
        _x5, _y5, yaw5, count5 = settled_pose()
        turned = wrap_angle(yaw5 - yaw4)
        check("right", turned < -0.3, f"yaw変化={turned:.3f} rad")

        start = probe.wait_pose(count5, timeout_sec=3.0)
        prediction = classifier.predict("前に進め")
        cancel = threading.Event()

        def interrupted() -> None:
            print(execute_prediction(probe.publisher, prediction, cancel))

        worker = threading.Thread(target=interrupted)
        worker.start()
        time.sleep(0.4)
        cancel.set()
        probe.publisher.publish(0.0, 0.0)
        worker.join(timeout=3.0)
        interrupted_pose = settled_pose()
        interrupted_distance = forward_distance(start[0], start[1], start[2], interrupted_pose[0], interrupted_pose[1])
        check("interrupt", interrupted_distance < 0.35, f"中断までの前方変位={interrupted_distance:.3f} m")
        held = probe.wait_pose(interrupted_pose[3], timeout_sec=3.0)
        coast = abs(forward_distance(interrupted_pose[0], interrupted_pose[1], interrupted_pose[2], held[0], held[1]))
        check("stop_after_interrupt", coast < 0.15, f"中断後の変位={coast:.3f} m")

        before = held
        run_command("こんにちは")
        time.sleep(1.3)
        after = probe.wait_pose(before[3], timeout_sec=3.0)
        drift = abs(forward_distance(before[0], before[1], before[2], after[0], after[1]))
        yaw_drift = abs(wrap_angle(after[2] - before[2]))
        check("reject", drift < 0.05 and yaw_drift < 0.05, f"変位={drift:.3f} m yaw={yaw_drift:.3f} rad")
    finally:
        probe.stop()
        probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin.join(timeout=2.0)

    if failures:
        print("失敗:", ", ".join(failures))
        raise SystemExit(1)
    print("シミュレータ上の動作確認はすべて成功しました。")


if __name__ == "__main__":
    main()
