# SAM-Echo GT Box 本地复现记录

## 复现对象

- 论文：*Foundation versus domain-specific models for left ventricular segmentation on cardiac ultrasound*，npj Digital Medicine，2025。
- 官方代码：`external/SAM-Echo/`。
- 官方检查点：`checkpoints/sam_echo_official/sam_model_best.pth`。
- 检查点信息：epoch 67，保存的验证 Dice 为 0.9439897。
- 模型：SAM ViT-B，1024×1024 输入，GT LV Box Prompt。

## 本地协议

- 数据：CAMUS 官方测试划分，共 50 个患者。
- 帧：A2C/A4C 的 ED/ES，共 200 帧。
- 目标：左心室心腔，CAMUS 标签 1。
- 提示：由 GT LV 掩膜生成紧致外接框。
- 汇总：每位患者、每个时相先对 A2C/A4C 求均值，然后在 50 位患者间报告均值和标准差。
- 指标：Dice、IoU、HD95（毫米）。

## 实际运行结果

| Phase | Dice | IoU | HD95 (mm) |
|---|---:|---:|---:|
| ED | 0.8715 ± 0.0383 | 0.7761 ± 0.0544 | 8.59 ± 2.37 |
| ES | 0.8355 ± 0.0439 | 0.7220 ± 0.0601 | 8.09 ± 1.98 |

- 本地 GPU：NVIDIA GeForce RTX 3060 Laptop GPU，6 GB。
- 200 帧推理时间：50.8 秒。
- 峰值已分配显存：1.55 GB。

逐帧数据见 `results/gt_box/per_frame_metrics.csv`，机器可读汇总见 `results/gt_box/summary.csv`，运行元信息见 `results/gt_box/run_metadata.json`。

## 与论文表格的关系

论文报告 CAMUS 时按 A2C/A4C 分开，并同时比较 SAM112、SAM1024 和 MedSAM；本项目为课程实验而按 ED/ES 汇总 A2C/A4C。因此本地结果是“使用作者公开 SAM1024 检查点，在本项目统一测试协议下的复现”，不能把它表述成论文原表数值的逐项复刻。作者公开文件夹目前提供的 `sam_model_best.pth` 是 70 epoch 分支，而论文中 CAMUS 最常引用的最佳结果来自 SAM112。
