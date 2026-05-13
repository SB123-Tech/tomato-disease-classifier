# 第四部分：迁移学习与农业图像识别

使用 PyTorch 预训练模型对番茄叶片病害进行分类，体验从零训练、特征提取到微调的完整迁移学习流程。

## 环境

复用第一部分构建的 `dl-env:latest` 镜像（CPU 用户）或 `dl-env-gpu:latest` 镜像（NVIDIA GPU 用户），通过 Jupyter Notebook 完成所有实验。

## 数据集

PlantVillage 番茄子集：约 6,200 张番茄叶片图片，4 个类别。
- GitHub: https://github.com/spMohanty/plantvillage-dataset

## 文件说明

| 文件 | 说明 |
|------|------|
| `任务指导书.md` | 详细步骤和代码 |
| `task_a_transfer_learning.py` | 任务 A 参考代码：特征提取 |
| `task_b_finetuning.py` | 任务 B 参考代码：微调 |
| `task_c_model_comparison.py` | 任务 C 参考代码：ResNet18 vs MobileNetV2 |
| `task_d_challenge.py` | 任务 D 参考代码：三种方案对比 |

## 任务列表

- **任务 A**（25 分钟）：迁移学习基础 — 冻结特征提取
- **任务 B**（25 分钟）：微调 — 解冻更多层提升准确率
- **任务 C**（20 分钟）：模型对比 — ResNet18 vs MobileNetV2
- **任务 D**（30 分钟）：综合项目 — 从零训练 vs 迁移学习大比拼
- **Git 实践**：用 Git 管理实验代码，推送到 GitHub
