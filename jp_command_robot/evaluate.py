"""学習データに含まれない文での分類評価。数値は実行時に計算する。"""

from __future__ import annotations

from jp_command_robot.classifier import (
    DATA_DIR,
    CommandClassifier,
    load_rows,
    train_model,
    _ranked,
)


def _print_group(title: str, rows: list[tuple[bool, str, str, float, float]]) -> None:
    print(f"\n## {title}")
    correct = 0
    for ok, text, detail, score, margin in rows:
        mark = "OK" if ok else "NG"
        print(f"{mark}\t{score:.4f}\t{margin:.4f}\t{text}\t{detail}")
        correct += int(ok)
    total = len(rows)
    rate = correct / total if total else 0.0
    print(f"件数: {correct}/{total} ({rate:.4f})")


def main() -> None:
    train_rows = load_rows(DATA_DIR / "train.csv")
    calibration_rows = load_rows(DATA_DIR / "calibration.csv")
    eval_rows = load_rows(DATA_DIR / "eval.csv")
    model = train_model(train_rows)
    classifier = CommandClassifier.from_data_dir()
    thresholds = classifier.thresholds

    print("学習件数:", len(train_rows))
    print("閾値ルール:", thresholds.rule)
    print(f"min_score: {thresholds.min_score:.6f}")
    print(f"min_margin: {thresholds.min_margin:.6f}")

    print("\n## 校正データ（閾値決定に使用）")
    for row in calibration_rows:
        label, score, margin = _ranked(model, row["text"])
        print(
            f"{row['split']}\t期待={row['label']}\t予測={label}\t"
            f"score={score:.4f}\tmargin={margin:.4f}\t{row['text']}"
        )

    in_results = []
    reject_results = []
    for row in eval_rows:
        prediction = classifier.predict(row["text"])
        if row["split"] == "in_domain":
            ok = prediction.accepted and prediction.label == row["label"]
            detail = f"期待={row['label']} 予測={prediction.label}"
            in_results.append((ok, row["text"], detail, prediction.score, prediction.margin))
        elif row["split"] == "reject":
            ok = not prediction.accepted
            detail = f"accepted={prediction.accepted} label={prediction.label}"
            reject_results.append((ok, row["text"], detail, prediction.score, prediction.margin))
        else:
            raise ValueError(row["split"])

    _print_group("評価: 未学習の動作命令（受理かつ正解）", in_results)
    _print_group("評価: 未学習の拒否命令（不受理）", reject_results)


if __name__ == "__main__":
    main()
