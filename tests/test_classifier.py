import unittest
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from jp_command_robot.classifier import (
    DATA_DIR,
    MOTION_LABELS,
    CommandClassifier,
    ambiguity_reason,
    load_rows,
)

REQUIRED = {
    "前に進め": "forward",
    "まっすぐ進んで": "forward",
    "後退してください": "backward",
    "後ろに下がれ": "backward",
    "左に曲がって": "left",
    "左折してください": "left",
    "右に回転して": "right",
    "右折して": "right",
    "止まって": "stop",
    "停止してください": "stop",
}


class ClassifierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.classifier = CommandClassifier.from_data_dir()
        cls.train = load_rows(DATA_DIR / "train.csv")
        cls.calibration = load_rows(DATA_DIR / "calibration.csv")
        cls.evaluation = load_rows(DATA_DIR / "eval.csv")

    def test_training_data_has_at_least_30_examples_per_motion(self) -> None:
        counts = Counter(row["label"] for row in self.train)
        for label in MOTION_LABELS:
            self.assertGreaterEqual(counts[label], 30)
        self.assertGreaterEqual(counts["unknown"], 30)

    def test_datasets_do_not_share_sentences(self) -> None:
        groups = {
            "train": {row["text"] for row in self.train},
            "calibration": {row["text"] for row in self.calibration},
            "eval": {row["text"] for row in self.evaluation},
        }
        self.assertTrue(groups["train"].isdisjoint(groups["calibration"]))
        self.assertTrue(groups["train"].isdisjoint(groups["eval"]))
        self.assertTrue(groups["calibration"].isdisjoint(groups["eval"]))

    def test_model_is_char_tfidf_logistic_regression(self) -> None:
        vectorizer = self.classifier.model.named_steps["tfidfvectorizer"]
        logistic = self.classifier.model.named_steps["logisticregression"]
        self.assertIsInstance(vectorizer, TfidfVectorizer)
        self.assertEqual(vectorizer.analyzer, "char")
        self.assertEqual(vectorizer.ngram_range, (1, 3))
        self.assertIsInstance(logistic, LogisticRegression)
        self.assertGreaterEqual(logistic.max_iter, 1000)

    def test_required_unseen_commands(self) -> None:
        train_texts = {row["text"] for row in self.train}
        for text, label in REQUIRED.items():
            self.assertNotIn(text, train_texts)
            prediction = self.classifier.predict(text)
            self.assertTrue(prediction.accepted, prediction)
            self.assertEqual(prediction.label, label)

    def test_unknown_and_ambiguous_commands_do_not_move(self) -> None:
        for row in self.evaluation:
            if row["split"] != "reject":
                continue
            prediction = self.classifier.predict(row["text"])
            self.assertFalse(prediction.accepted, row["text"])
            self.assertIsNone(prediction.label)

    def test_empty_command_is_rejected(self) -> None:
        prediction = self.classifier.predict("   ")
        self.assertFalse(prediction.accepted)

    def test_contrastive_command_is_ambiguous(self) -> None:
        self.assertIsNotNone(ambiguity_reason("進め、ただし止まれ"))
        self.assertIsNone(ambiguity_reason("前に進め"))


class DatasetPathTest(unittest.TestCase):
    def test_data_files_exist(self) -> None:
        for name in ("train.csv", "calibration.csv", "eval.csv"):
            self.assertTrue((Path(DATA_DIR) / name).is_file())


if __name__ == "__main__":
    unittest.main()
