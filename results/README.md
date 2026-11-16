# 实验结果

当前实验结果包括：

- `gt_box/`：SAM-Echo + GT Box 参考结果
- `autobox_box_only/`：仅使用 Box Loss 的 AutoBox 对照结果
- `autobox_segaware/`：联合 Dice、BCE、Box Loss 的最终结果
- `main_table.csv`
- `main_table.md`
- `segaware_training_curves.png`

各方法的 `per_frame_metrics.csv` 保留 patient、view 和 phase；最终主表先在患者内合并 A2C/A4C，再分别聚合 ED、ES。

