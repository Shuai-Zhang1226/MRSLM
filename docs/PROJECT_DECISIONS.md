# 项目决策

## 已确定

- 放弃 SAM 3 心超视频路线，当前不建模时间信息。
- 使用 SAM-Echo ViT-B，而不是原始 SAM ViT-H。
- 只处理 CAMUS 的 ED/ES 单帧。
- 只分割左心室心腔。
- 使用自动 Box Prompt，不使用文本作为核心输入。
- 原因：SAM本身不支持文本，而且单类别固定文本会退化为类别标识；自动框与SAM-Echo现有接口和RTX 3060显存条件更匹配。
- 自动框由CAMUS GT mask自动生成监督，不增加人工标注。
- 最终A2C/A4C合并统计，ED和ES分别报告Dice、IoU、HD95。

## 已落实的实现选择

- CAMUS 默认路径为仓库根目录下的 `dataset/CAMUS_public`，读取 NIfTI ED/ES 帧；也可以在配置文件中改为其他位置。
- 使用 CAMUS 官方 400/50/50 patient-level train/validation/test 划分。
- SAM-Echo 使用作者公开的 `sam_model_best.pth`（ViT-B，epoch 67）。
- AutoBox 只读取冻结的 SAM embedding，不增加原图 CNN 分支。
- AutoBox 预测紧致框，推理时固定扩张 5%；GT Box 论文复现使用紧致框。

## 实现优先级

先完成最小闭环：

1. SAM-Echo + GT Box；
2. 冻结特征上的中心热图和LTRB框预测；
3. SAM-Echo + Auto Box；
4. Dice、IoU、HD95和Box IoU。

在最小闭环稳定之前，不增加文本分支、边界细化网络或额外损失。

