# AutoBox 权重

`autobox_segaware_best.pt` 是本项目在 CAMUS 训练集上使用联合损失得到的最佳 AutoBox 检查点。

- 最佳 epoch：50
- 验证集 Dice：0.8290
- 验证集 Box IoU：0.7737
- 损失：`Dice + BCE + 0.2 × Box`
- SHA256：`05B63C6824328D121B3F48E284649337033358A3D564D578434A32FC549B16F1`

该检查点只包含 AutoBox 及其训练状态，运行完整分割流程仍需单独下载 SAM-Echo 官方 `sam_model_best.pth`。
