from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


project = Path(__file__).resolve().parents[1]
history = pd.read_csv(project / "logs" / "autobox_segaware_history.csv")
best = history.loc[history["val_dice"].idxmax()]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
axes[0].plot(history["epoch"], history["train_total_loss"], label="Total")
axes[0].plot(history["epoch"], history["train_segmentation_loss"], label="Dice + BCE")
axes[0].plot(history["epoch"], 0.2 * history["train_box_loss"], label="0.2 × Box")
axes[0].set(title="Training losses", xlabel="Epoch", ylabel="Loss")
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].plot(history["epoch"], history["val_dice"], label="Validation Dice")
axes[1].plot(history["epoch"], history["val_box_iou"], label="Validation Box IoU")
axes[1].scatter([best["epoch"]], [best["val_dice"]], color="crimson", zorder=3)
axes[1].annotate(
    f"best Dice={best['val_dice']:.4f}\nepoch {int(best['epoch'])}",
    (best["epoch"], best["val_dice"]),
    xytext=(8, -35),
    textcoords="offset points",
    fontsize=9,
)
axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0.6, 0.86))
axes[1].grid(alpha=0.25)
axes[1].legend(loc="lower right")

fig.suptitle("Segmentation-aware AutoBox training")
fig.tight_layout()
output = project / "results" / "segaware_training_curves.png"
fig.savefig(output, bbox_inches="tight")
print(output)
