"""文字 n-gram TF-IDF とロジスティック回帰による日本語命令の分類。"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

MOTION_LABELS = ("forward", "backward", "left", "right", "stop")
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Prediction:
    text: str
    accepted: bool
    label: str | None
    score: float
    margin: float
    reason: str


@dataclass(frozen=True)
class Thresholds:
    min_score: float
    min_margin: float
    rule: str


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_model() -> Pipeline:
    return make_pipeline(
        TfidfVectorizer(
            analyzer="char",
            ngram_range=(1, 3),
        ),
        LogisticRegression(
            max_iter=1000,
            random_state=0,
        ),
    )


def train_model(rows: list[dict[str, str]]) -> Pipeline:
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    model = build_model()
    model.fit(texts, labels)
    return model


def _ranked(model: Pipeline, text: str) -> tuple[str, float, float]:
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    order = np.argsort(probabilities)[::-1]
    top = int(order[0])
    second = int(order[1])
    score = float(probabilities[top])
    margin = float(probabilities[top] - probabilities[second])
    return str(classes[top]), score, margin


def ambiguity_reason(text: str) -> str | None:
    """複数方向や逆接を含む文は、分類スコアに関係なく動かさない。"""
    has_forward = any(keyword in text for keyword in ("前", "直進", "正面", "まっすぐ", "真っ直ぐ"))
    has_backward = any(keyword in text for keyword in ("後", "バック", "下が", "後退", "後進"))
    has_left = "左" in text
    has_right = "右" in text
    has_stop = any(keyword in text for keyword in ("止", "停", "ストップ", "ブレーキ"))
    forward_is_only_prefix = any(
        keyword in text for keyword in ("名前", "前回", "手前", "前置き", "午前")
    )
    if has_forward and forward_is_only_prefix:
        has_other_forward = any(
            keyword in text for keyword in ("直進", "正面", "まっすぐ", "真っ直ぐ", "前進")
        )
        if not has_other_forward:
            has_forward = False

    direction_count = sum((has_forward, has_backward, has_left, has_right))
    contrast = any(keyword in text for keyword in ("ただし", "しかし", "でも", "かつ", "または", "あるいは"))
    if has_left and has_right:
        return "左右が同時に含まれるため、ロボットは動かしません。"
    if has_forward and has_backward:
        return "前後が同時に含まれるため、ロボットは動かしません。"
    if direction_count >= 2:
        return "複数の方向が含まれるため、ロボットは動かしません。"
    if contrast:
        return "複数の意図が含まれるため、ロボットは動かしません。"
    return None


def _accepted(label: str, score: float, margin: float, min_score: float, min_margin: float) -> bool:
    return label in MOTION_LABELS and score > min_score and margin > min_margin


def _gap_threshold(good_values: list[float], bad_values: list[float], floor: float) -> float:
    """誤って動作とみなした校正例と、正しい校正例の中央のあいだに閾値を置く。"""
    if not good_values:
        raise RuntimeError("校正用の正解例がありません。")
    max_bad = max(bad_values) if bad_values else 0.0
    center = float(median(good_values))
    if center > max_bad:
        threshold = max_bad + 0.5 * (center - max_bad)
    else:
        threshold = max_bad
    return float(max(threshold, floor))


def derive_thresholds(
    model: Pipeline,
    calibration_rows: list[dict[str, str]],
) -> Thresholds:
    """校正データだけで拒否閾値を決める。

    出力スコアは校正済みの正解確率ではない。曖昧さの規則で除外したあと、
    動作クラスへ誤って寄った校正例の最大値と、正しく分類できた校正例の
    中央値の中間を閾値にする。
    """
    good_scores: list[float] = []
    good_margins: list[float] = []
    bad_scores: list[float] = []
    bad_margins: list[float] = []
    for row in calibration_rows:
        if ambiguity_reason(row["text"]):
            continue
        predicted, score, margin = _ranked(model, row["text"])
        if row["split"] == "in_domain" and predicted == row["label"]:
            good_scores.append(score)
            good_margins.append(margin)
            continue
        if row["split"] not in {"in_domain", "reject"}:
            raise ValueError(f"未知の split です: {row['split']}")
        if predicted in MOTION_LABELS:
            bad_scores.append(score)
            bad_margins.append(margin)

    return Thresholds(
        min_score=_gap_threshold(good_scores, bad_scores, floor=0.0),
        min_margin=_gap_threshold(good_margins, bad_margins, floor=0.15),
        rule="calibration_midpoint_above_false_motion",
    )


class CommandClassifier:
    def __init__(self, model: Pipeline, thresholds: Thresholds) -> None:
        self.model = model
        self.thresholds = thresholds

    @classmethod
    def from_data_dir(cls, data_dir: Path = DATA_DIR) -> "CommandClassifier":
        model = train_model(load_rows(data_dir / "train.csv"))
        thresholds = derive_thresholds(model, load_rows(data_dir / "calibration.csv"))
        return cls(model, thresholds)

    def predict(self, text: str) -> Prediction:
        normalized = " ".join(text.strip().split())
        if not normalized:
            return Prediction(text, False, None, 0.0, 0.0, "空の命令です。")
        label, score, margin = _ranked(self.model, normalized)
        ambiguous = ambiguity_reason(normalized)
        if ambiguous:
            return Prediction(normalized, False, None, score, margin, ambiguous)
        if label == "unknown":
            return Prediction(
                normalized,
                False,
                None,
                score,
                margin,
                "未知の命令と分類したため、ロボットは動かしません。",
            )
        thresholds = self.thresholds
        if not _accepted(label, score, margin, thresholds.min_score, thresholds.min_margin):
            return Prediction(
                normalized,
                False,
                None,
                score,
                margin,
                "出力スコアが閾値以下のため、ロボットは動かしません。",
            )
        return Prediction(
            normalized,
            True,
            label,
            score,
            margin,
            "分類しました。",
        )
