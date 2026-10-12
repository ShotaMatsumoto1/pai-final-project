import threading
import unittest

from jp_command_robot.classifier import Prediction
from jp_command_robot.controller import execute_prediction, motion_for, publish_for


class FakePublisher:
    def __init__(self) -> None:
        self.messages: list[tuple[float, float]] = []

    def publish(self, linear_x: float, angular_z: float) -> None:
        self.messages.append((linear_x, angular_z))

    def sleep(self, duration_sec: float) -> None:
        return None


class ControllerLogicTest(unittest.TestCase):
    def test_motion_table(self) -> None:
        self.assertGreater(motion_for("forward").linear_x, 0.0)
        self.assertLess(motion_for("backward").linear_x, 0.0)
        self.assertGreater(motion_for("left").angular_z, 0.0)
        self.assertLess(motion_for("right").angular_z, 0.0)
        self.assertEqual(motion_for("stop").linear_x, 0.0)
        self.assertEqual(motion_for("stop").angular_z, 0.0)

    def test_rejected_prediction_publishes_nothing(self) -> None:
        publisher = FakePublisher()
        prediction = Prediction("こんにちは", False, None, 0.2, 0.01, "拒否")
        execute_prediction(publisher, prediction)
        self.assertEqual(publisher.messages, [])

    def test_forward_ends_with_stop(self) -> None:
        publisher = FakePublisher()
        prediction = Prediction("前に進め", True, "forward", 0.8, 0.5, "分類しました。")
        execute_prediction(publisher, prediction)
        self.assertGreater(publisher.messages[0][0], 0.0)
        self.assertEqual(publisher.messages[-1], (0.0, 0.0))

    def test_cancel_still_sends_stop(self) -> None:
        publisher = FakePublisher()
        cancel = threading.Event()
        cancel.set()
        publish_for(publisher, 0.4, 0.0, 2.0, cancel)
        self.assertEqual(publisher.messages, [])
        prediction = Prediction("前に進め", True, "forward", 0.8, 0.5, "分類しました。")
        execute_prediction(publisher, prediction, cancel)
        self.assertEqual(publisher.messages[-1], (0.0, 0.0))
        self.assertTrue(all(message == (0.0, 0.0) for message in publisher.messages))


if __name__ == "__main__":
    unittest.main()
