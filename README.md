# 基于 MobileNetViT 的交通标志检测系统

一个面向**交通标志识别（Traffic Sign Recognition）**的轻量级图像分类系统。项目设计并实现了 **MobileNetViT** 混合模型：以 **MobileNet 深度可分离卷积**提取局部特征，用 **Vision Transformer 编码器**建模全局依赖，并引入 **类别感知注意力（Class-Aware Attention）** 强化分类判别能力。在公开数据集 **GTSRB** 上，将 43 个细类归并成 5 个大类进行识别，并与 VGG16、ResNet18 等经典模型进行精度与效率对比。

系统以约 **0.35 M** 参数量达到约 **99%** 的测试准确率，相比 VGG16（约 39.9 M 参数）在保持相近精度的同时，参数规模缩小超过 100 倍，适合部署到算力受限的移动端 / 边缘设备。

---

## 目录

- [项目简介](#项目简介)
- [数据集](#数据集)
- [模型结构](#模型结构)
- [功能特性](#功能特性)
- [环境依赖](#环境依赖)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [数据预处理](#数据预处理)
- [模型训练](#模型训练)
- [测试与评估](#测试与评估)
- [实验结果](#实验结果)
- [训练脚本与参数修改说明](#训练脚本与参数修改说明)
- [常见问题](#常见问题)
- [参考与致谢](#参考与致谢)

---

## 项目简介

交通标志通常具有尺寸小、类别多、光照与遮挡变化大等特点。传统 CNN 在局部特征提取上表现优秀，但建模长距离依赖的能力有限；纯 Transformer 又往往参数量大、对数据规模敏感。本项目将 MobileNet 的轻量化卷积结构与 Vision Transformer 的全局建模能力相结合，构建了一个高精度、低参数的 **MobileNetViT** 模型。

系统主要组成：

1. **数据预处理**：GTSRB 原始数据清洗、尺寸统一、模糊过滤、数据增强与类别平衡。
2. **MobileNetViT 模型**：深度可分离卷积 + Transformer 编码器 + 类别感知注意力。
3. **对比模型**：VGG16、ResNet18、GELUResNet18。
4. **训练**：加权交叉熵、AdamW / Adam 优化器、余弦退火学习率调度。
5. **评估**：准确率、精确率、召回率、F1、混淆矩阵、参数量与推理耗时对比。

---

## 数据集

本项目使用德国交通标志识别基准数据集 [GTSRB（German Traffic Sign Recognition Benchmark）](https://benchmark.ini.rub.de/)。

原始 GTSRB 包含 43 个细粒度类别，本实验将其归并为 5 个大类：

| 大类 | 含义 | 原始 ClassId |
| --- | --- | --- |
| class0 | 限速类 | 0 ~ 8 |
| class1 | 禁止类 | 9 ~ 16、38 ~ 40、42 |
| class2 | 指示类 | 17 ~ 22、24、28 ~ 30 |
| class3 | 警告类 | 25 ~ 27、31 ~ 32、36 |
| class4 | 其他类 | 33 ~ 35 |

> 数据集请自行从 GTSRB 官方渠道获取，项目中仅保留处理后的索引 CSV 与少量示例图片，未包含完整原始数据。

---

## 模型结构

`MobileNetViT` 的前向流程如下：

```text
输入图像 (3 × 64 × 64)
        │
        ▼
卷积 Stem（3×3 Conv + BN + GELU）
        │
        ▼
深度可分离卷积 ×2（Depthwise + Pointwise + BN + GELU）
   + 1×1 卷积增强特征表达
        │
        ▼
Patch Embedding（8×8 patch，共 16 个 patch）
        │
        ▼
拼接 class token + 位置编码
        │
        ▼
Transformer Encoder × N（多头自注意力 + FFN + LayerNorm）
        │
        ▼
类别感知注意力（Class-Aware Attention）
        │
        ▼
分类头（Linear + GELU + Dropout + Linear）
        │
        ▼
输出 logits（5 类）
```

核心模块说明：

- **`DepthwiseSeparableConv`**：MobileNet 核心组件，通过深度卷积 + 点卷积大幅降低计算量与参数量。
- **`TransformerEncoder`**：多头自注意力 + 前馈网络 + 残差连接与层归一化，建模全局依赖。
- **`ClassAwareAttention`**：引入可学习的类别权重矩阵，计算每个 token 与类别的相似度，为不同类别特征分配权重。
- **`MobileNetViT`**：融合上述模块，`num_transformer_layers` 可配置（基准为 2 层，可扩展为 3 层）。

对比模型（同样在 `model.py` 中实现）：

- **VGG16**：经典卷积网络，参数约 39.9 M。
- **ResNet18**：残差网络，参数约 11.2 M。
- **GELUResNet18**：将 ResNet18 中的 ReLU 替换为 GELU 的变体。

---

## 功能特性

- **轻量高精度**：MobileNetViT 仅约 0.35 M 参数，即可达到约 99% 的 5 分类准确率。
- **混合架构**：结合卷积局部建模与 Transformer 全局建模。
- **类别感知注意力**：针对交通标志类别间的相似性进行判别增强。
- **类别不平衡处理**：数据增强 + 过采样 / 欠采样 + 加权交叉熵。
- **完整评估体系**：Accuracy、Precision、Recall、F1、分类报告、混淆矩阵、参数量、推理耗时。
- **多组对比实验**：不同学习率、批大小、优化器、激活函数的消融实验。

---

## 环境依赖

推荐 Python 3.8+，使用 PyTorch 深度学习框架（无 GPU 时自动回退到 CPU）。

| 依赖 | 说明 |
| --- | --- |
| Python | 3.8+（开发环境为 3.10） |
| PyTorch | 2.x |
| torchvision | 提供预训练 backbone 与图像变换 |
| opencv-python | 图像读取与预处理 |
| numpy / pandas | 数值计算与数据管理 |
| scikit-learn | 数据集划分与评价指标 |
| matplotlib / seaborn | 曲线与混淆矩阵可视化 |
| tqdm | 进度显示 |
| imbalanced-learn | 过采样 / 欠采样（类别平衡） |

安装示例：

```bash
pip install torch torchvision
pip install opencv-python numpy pandas scikit-learn matplotlib seaborn tqdm imbalanced-learn
```

---

## 目录结构

```text
dazuoye/
├── model.py                  # MobileNetViT 与对比模型定义
├── prepose.py                # 数据预处理 / 训练验证集划分
├── 9.py                      # 数据清洗 + 增强 + 类别平衡（完整版）
├── number.py                 # 类别样本数量统计
├── train.py                  # 基线训练（batch=32, lr=1e-4, AdamW）
├── train2.py                 # 学习率 5e-4 实验
├── train3.py                 # batch size 64 实验
├── train4.py                 # Adam 优化器 + 增强数据增强实验
├── train5.py                 # GELU 激活函数实验
├── test.py                   # 测试集评估 + 模型对比
├── img.png / img_1.png / img_2.png   # 单张图片测试示例
├── processe_data/            # 处理后的数据集（图片 + CSV 索引）
├── models0 32 0.0001 50/     # 基线实验输出
├── models1 32 0.0005 50/     # 学习率实验输出
├── models2 64 0.0001 50/     # 批大小实验输出
├── models3 32 0.0001 50 adam/  # Adam 优化器实验输出
└── models4 32 0.0005 GELU 50/  # GELU 激活实验输出
```

每个 `models*` 目录下保存了对应实验的：

- `*_best.pth` / `*_final.pth`：最佳 / 最终模型权重；
- `*_history.png`：训练损失与准确率曲线；
- `*_confusion_matrix.png`：混淆矩阵；
- `model_comparison.csv`：模型对比结果。

---

## 快速开始

1. 安装依赖（见 [环境依赖](#环境依赖)）。
2. 准备 GTSRB 数据集，并修改预处理脚本中的 `RAW_DATA_PATH` 为实际路径。
3. 运行数据预处理，生成 `processe_data/` 下的 CSV 与图片。
4. 启动训练：

```bash
python train.py
```

5. 训练完成后评估：

```bash
python test.py
```

---

## 数据预处理

预处理流程（对应 `prepose.py` 与 `9.py` 中的逻辑）：

1. **类别归并**：将 GTSRB 的 43 个原始 `ClassId` 映射到 5 个大类。
2. **数据清洗**：过滤路径不存在、读取失败、模糊（拉普拉斯方差低于阈值）的图像。
3. **尺寸统一**：所有图像 resize 为 64×64。
4. **数据增强**（仅训练集）：
   - 亮度 / 对比度调整；
   - 随机旋转（±15°）；
   - 随机裁剪（56×56 后 resize 回 64×64）。
5. **类别平衡**：使用 `RandomOverSampler`（过采样少数类）+ `RandomUnderSampler`（欠采样多数类）平衡类别分布。
6. **划分**：训练集按 8:2 分层划分出训练 / 验证集（`train_final.csv` / `val_processed.csv`），并保留独立测试集 `test_processed.csv`。

`number.py` 用于统计并打印各数据集的类别样本数量，便于核对平衡结果。

---

## 模型训练

训练使用以下通用配置：

- 输入尺寸：64×64；
- 类别数：5；
- 损失函数：带类别权重的 `CrossEntropyLoss`；
- 学习率调度：`CosineAnnealingLR` 余弦退火；
- 训练轮数：50 epoch；
- 随机种子：42（保证可复现）。

默认训练脚本会依次训练 MobileNetViT（2 层 Transformer）以及 VGG16、ResNet18 等对比模型：

```bash
python train.py      # 基线
python train2.py     # lr = 5e-4
python train3.py     # batch_size = 64
python train4.py     # Adam 优化器 + 增强数据增强
python train5.py     # GELU 激活 + GELUResNet18
```

训练过程中会保存最佳 / 最终模型权重，并绘制训练损失与准确率曲线。

---

## 测试与评估

`test.py` 在测试集上对多个模型进行统一评估，输出：

- Accuracy、Precision、Recall、F1 Score；
- 分类报告（`classification_report`）；
- 混淆矩阵（保存为 PNG）；
- 模型参数量与平均推理耗时。

运行：

```bash
python test.py
```

评估结果会汇总为 `model_comparison.csv`，保存在对应实验目录下。

---

## 实验结果

以下结果为各实验在测试集上的 Accuracy / F1（数据来源：各 `models*` 目录下的 `model_comparison.csv`）：

| 实验目录 | 关键改动 | MobileNetViT | MobileNetViT3 | VGG16 | ResNet18 / GELU |
| --- | --- | --- | --- | --- | --- |
| models0 | 基线（lr=1e-4, batch=32, AdamW） | 0.9789 | — | 0.9924 | 0.9914 |
| models1 | lr=5e-4 | 0.9907 | — | 0.9959 | 0.9943 |
| models2 | batch=64 | 0.9727 | 0.9761 | 0.9954 | 0.9891 |
| models3 | Adam 优化器 + 增强增强 | 0.9727 | 0.9761 | 0.9954 | 0.9891 |
| models4 | GELU 激活 + lr=5e-4 | 0.9912 | 0.9928 | 0.9949 | 0.9940 |

模型参数量与推理耗时（单张 64×64 图像，GPU）：

| 模型 | 参数量 | 推理耗时 |
| --- | --- | --- |
| MobileNetViT（2 层） | 0.345 M | 约 1.6 ms |
| MobileNetViT3（3 层） | 0.379 M | 约 2.0 ms |
| ResNet18 / GELUResNet18 | 11.18 M | 约 1.8 ms |
| VGG16 | 39.91 M | 约 1.3 ms |

结论：**MobileNetViT 以约 0.35 M 的极小参数量达到了与 VGG16（39.9 M）相近的分类精度**，在精度与效率之间取得了良好平衡，验证了轻量化混合架构在交通标志识别任务上的有效性。

---

## 训练脚本与参数修改说明

项目中的多个 `train*.py` 文件对应不同的模型与参数修改实验：

| 文件 | 相对基线的修改 | 输出目录 |
| --- | --- | --- |
| `train.py` | 基线：batch=32、lr=1e-4、AdamW、基础增强 | models0 |
| `train2.py` | 学习率提高到 5e-4 | models1 |
| `train3.py` | batch size 提高到 64 | models2 |
| `train4.py` | 优化器改为 Adam，增强数据增强（旋转/平移/色彩抖动） | models3 |
| `train5.py` | 激活函数 ReLU → GELU，并加入 GELUResNet18 对比 | models4 |

模型结构层面的修改集中在 `model.py`：

- MobileNetViT 的 Transformer 层数可在 2 层 / 3 层之间切换（`num_transformer_layers`）；
- 深度可分离卷积、Transformer 编码器、类别感知注意力均为可替换 / 可扩展的独立模块；
- 对比模型包含 VGG16、ResNet18，以及将 ReLU 替换为 GELU 的 GELUResNet18。

---

## 常见问题

**Q1：运行时报数据集路径不存在？**

预处理脚本与训练脚本中的路径为开发机的绝对路径，迁移后请将 `RAW_DATA_PATH`、`Config.data_path` 等统一修改为实际路径。

**Q2：找不到 `train_processed.csv` / `test_processed.csv`？**

请先按顺序运行数据预处理脚本，生成 `processe_data/` 下的 CSV 与图片。注意不同脚本中目录名存在 `processe_data`、`processed_data`、`processedf_data` 等历史写法，需保持一致。

**Q3：没有 GPU 可以训练吗？**

可以，代码会自动检测 CUDA，无 GPU 时回退到 CPU，但训练和推理速度会明显下降。

**Q4：为什么不同实验的 MobileNetViT 精度有波动？**

精度受学习率、批大小、优化器、激活函数、数据增强强度等多因素影响，这也是本项目设计多组消融实验的目的。

---

## 参考与致谢

- [MobileNet](https://arxiv.org/abs/1704.04861)：深度可分离卷积
- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)：Transformer 图像建模
- [GTSRB](https://benchmark.ini.rub.de/)：德国交通标志识别基准数据集
- [PyTorch](https://pytorch.org/) / [torchvision](https://pytorch.org/vision/stable/index.html)
