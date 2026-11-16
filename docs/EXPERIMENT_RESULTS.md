# AutoBox 联合损失实验结果

## 实验设置

- 数据：CAMUS 官方划分，400/50/50 位患者用于训练/验证/测试。
- 样本：每位患者 A2C、A4C 的 ED/ES 帧；分割目标为左心室心腔（标签 1）。
- 基础模型：作者公开的 SAM-Echo ViT-B `sam_model_best.pth`，参数全程冻结。
- 训练对象：从 SAM-Echo 图像特征生成框提示的轻量 AutoBox。
- batch size：4；优化器：AdamW；初始学习率：3e-4；余弦退火。
- 训练损失：`L = L_Dice + L_BCE + 0.2 L_Box`。
- 选择策略：仅按验证集掩码 Dice 保存最佳权重。
- 汇总方式：同一患者、同一时相先平均 A2C/A4C，再在 50 位测试患者上计算均值与标准差。

## 训练结果

- 实际完成 70 个 epoch；第 71 个 epoch 未完成且未写入日志。
- 最佳 epoch：50。
- 最佳验证 Dice：0.8290。
- 同轮验证 Box IoU：0.7737。
- 同轮训练总损失：0.3147，其中分割损失 0.2604、Box 损失 0.2715。

训练后期总损失继续下降，但验证 Dice 没有继续提升。因此正式测试使用 epoch 50 的检查点，而不是最后一轮权重。

## 测试集主结果

| 方法 | ED Dice | ED IoU | ED HD95 (mm) | ED Box IoU | ES Dice | ES IoU | ES HD95 (mm) | ES Box IoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SAM-Echo + GT Box | 0.8715 ± 0.0383 | 0.7761 ± 0.0544 | 8.59 ± 2.37 | 1.0000 | 0.8355 ± 0.0439 | 0.7220 ± 0.0601 | 8.09 ± 1.98 | 1.0000 |
| AutoBox（仅 Box Loss） | 0.8348 ± 0.0487 | 0.7214 ± 0.0687 | 10.20 ± 3.12 | 0.7923 | 0.7871 ± 0.0665 | 0.6568 ± 0.0862 | 10.38 ± 3.31 | 0.7658 |
| AutoBox（联合损失） | **0.8376 ± 0.0518** | **0.7262 ± 0.0722** | 10.48 ± 2.80 | 0.7869 | **0.8034 ± 0.0610** | **0.6774 ± 0.0820** | **10.24 ± 3.14** | **0.7662** |

## 结论与分析

- 相比只训练框，联合损失使 ED Dice 提升 0.0028、ES Dice 提升 0.0162；改善主要来自更困难的 ES 帧。
- 逐帧比较中，联合损失在 102/200 帧上取得更高 Dice，平均逐帧 Dice 提升 0.0095。
- 联合损失的平均逐帧 Box IoU 反而降低 0.0025，但 Dice 提高，说明优化“几何上最接近 GT 的框”和优化“最适合 SAM 解码的框”不是完全相同的目标。
- 联合模型全部 200 帧的 Box IoU 与 Dice 相关系数为 0.739，框质量仍是分割精度的重要影响因素。
- ED 的 Dice/IoU 提升但 HD95 略有变差（10.20 → 10.48 mm），因此不能声称所有指标均改善；ES 的三项分割指标均改善。
- GT Box 仍是人工定位条件下的参考上限，不属于全自动系统。

## 结果文件

- `logs/autobox_segaware_history.csv`：70 个完整 epoch 的训练与验证日志。
- `results/segaware_training_curves.png`：联合损失训练曲线。
- `results/autobox_segaware/per_frame_metrics.csv`：联合模型逐帧指标。
- `results/autobox_segaware/summary.csv`：联合模型患者级汇总。
- `results/autobox_box_only/`：旧的 Box-only 对照结果。
- `results/gt_box/`：SAM-Echo + GT Box 参考结果。
- `results/main_table.csv`、`results/main_table.md`：三种设置的最终对照表。
- `checkpoints/autobox_segaware_best.pt`：epoch 50 最佳检查点。
