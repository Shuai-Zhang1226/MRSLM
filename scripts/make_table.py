from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def format_method(name: str, frame: pd.DataFrame) -> dict[str, str]:
    row = {"Method": name}
    for phase in ("ED", "ES"):
        item = frame.loc[frame["phase"] == phase].iloc[0]
        row[f"{phase} Dice"] = f"{item['dice_mean']:.4f} ± {item['dice_std']:.4f}"
        row[f"{phase} IoU"] = f"{item['iou_mean']:.4f} ± {item['iou_std']:.4f}"
        row[f"{phase} HD95 (mm)"] = f"{item['hd95_mm_mean']:.2f} ± {item['hd95_mm_std']:.2f}"
        row[f"{phase} Box IoU"] = (
            f"{item['box_iou_mean']:.4f}" if "box_iou_mean" in frame.columns else "1.0000 (GT)"
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-missing-autobox", action="store_true")
    args = parser.parse_args()
    gt = pd.read_csv(PROJECT / "results" / "gt_box" / "summary.csv")
    rows = [format_method("SAM-Echo + GT Box", gt)]
    result_paths = [
        ("SAM-Echo + AutoBox (Box-only)", PROJECT / "results" / "autobox_box_only" / "summary.csv"),
        ("SAM-Echo + AutoBox (joint loss)", PROJECT / "results" / "autobox_segaware" / "summary.csv"),
    ]
    for name, auto_path in result_paths:
        if auto_path.exists():
            rows.append(format_method(name, pd.read_csv(auto_path)))
        elif not args.allow_missing_autobox:
            raise FileNotFoundError(f"Run the corresponding evaluation first: {auto_path}")
    table = pd.DataFrame(rows)
    table.to_csv(PROJECT / "results" / "main_table.csv", index=False)
    headers = list(table.columns)
    markdown = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    markdown.extend("| " + " | ".join(str(row[column]) for column in headers) + " |" for _, row in table.iterrows())
    (PROJECT / "results" / "main_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
