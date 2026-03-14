import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import MobileNetViT, VGG16, GELUResNet18

# 设置随机种子，保证实验可复现
torch.manual_seed(42)
np.random.seed(42)


# 配置训练参数
class Config:
    data_path = "./processe_data"
    train_csv = os.path.join(data_path, "train_processed.csv")
    test_csv = os.path.join(data_path, "test_processed.csv")
    num_classes = 5
    img_size = 64
    batch_size = 32
    lr = 0.0005
    epochs = 50
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "models4 32 0.0005 GELU 50"
    os.makedirs(save_path, exist_ok=True)


# 自定义数据集
class TrafficSignDataset(Dataset):
    def __init__(self, csv_path, transform=None, is_train=True):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        # 读取图像（BGR转RGB）
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 应用变换
        if self.transform:
            img = self.transform(img)
        return img, label


# 数据变换
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 加载数据
def load_data():
    # 加载训练集并划分为训练集和验证集
    train_dataset = TrafficSignDataset(
        Config.train_csv,
        transform=train_transform,
        is_train=True
    )
    # 划分训练集（90%）和验证集（10%）
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)),
        test_size=0.1,
        random_state=42,
        stratify=train_dataset.df['label']
    )
    # 创建子集
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)

    # 测试集
    test_dataset = TrafficSignDataset(
        Config.test_csv,
        transform=test_transform,
        is_train=False
    )

    # 数据加载器
    train_loader = DataLoader(
        train_subset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )

    return train_loader, val_loader, test_loader


# 计算类别权重（解决类别不平衡）
def compute_class_weights(dataset):
    labels = dataset.df['label'].values
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights).to(Config.device)


# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, model_name):
    # 记录训练过程
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    # 最佳验证准确率
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # 训练阶段
        model.train()
        train_running_loss = 0.0
        train_preds = []
        train_labels = []

        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(Config.device)
            labels = labels.to(Config.device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            # 统计
            train_running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # 计算训练集指标
        train_epoch_loss = train_running_loss / len(train_loader.dataset)
        train_epoch_acc = accuracy_score(train_labels, train_preds)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(Config.device)
                labels = labels.to(Config.device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # 计算验证集指标
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = accuracy_score(val_labels, val_preds)

        # 记录历史
        history['train_loss'].append(train_epoch_loss)
        history['val_loss'].append(val_epoch_loss)
        history['train_acc'].append(train_epoch_acc)
        history['val_acc'].append(val_epoch_acc)

        print(f"Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

        # 学习率调度
        scheduler.step()

        # 保存最佳模型
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), os.path.join(Config.save_path, f"{model_name}_best.pth"))
            print(f"Saved best model with Val Acc: {best_val_acc:.4f}")

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(Config.save_path, f"{model_name}_final.pth"))

    # 绘制损失和准确率曲线
    plot_training_history(history, model_name)

    return model, history


# 绘制训练历史
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'{model_name} Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_path, f"{model_name}_history.png"))
    plt.close()


# 主函数：训练目标模型
def main():
    # 加载数据
    train_loader, val_loader, test_loader = load_data()

    # 初始化训练集完整数据集（用于计算类别权重）
    full_train_dataset = TrafficSignDataset(Config.train_csv, transform=train_transform)
    class_weights = compute_class_weights(full_train_dataset)

    # # 1. 训练改进的MobileNetViT（基准模型）
    # print("Training Improved MobileNetViT...")
    # model_mobilenetvit = MobileNetViT(
    #     num_classes=Config.num_classes,
    #     num_transformer_layers=2  # 基准参数
    # ).to(Config.device)
    #
    # # 损失函数：加权交叉熵
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # # 优化器：AdamW
    # optimizer = optim.AdamW(model_mobilenetvit.parameters(), lr=Config.lr, weight_decay=1e-4)
    # # 学习率调度：余弦退火
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr / 10)
    #
    # # 训练
    # train_model(
    #     model_mobilenetvit,
    #     train_loader,
    #     val_loader,
    #     criterion,
    #     optimizer,
    #     scheduler,
    #     Config.epochs,
    #     "mobilenetvit_baseline"
    # )

    # 2. 参数调优实验：示例（Transformer层数=3）
    print("Training MobileNetViT with 3 Transformer layers...")
    model_mobilenetvit_3layers = MobileNetViT(
        num_classes=Config.num_classes,
        num_transformer_layers=3
    ).to(Config.device)

    optimizer = optim.AdamW(model_mobilenetvit_3layers.parameters(), lr=Config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr / 10)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_model(
        model_mobilenetvit_3layers,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        Config.epochs,
        "mobilenetvit_3layers"
    )
    # 1. 训练改进的MobileNetViT（基准模型）
    print("Training Improved MobileNetViT...")
    model_mobilenetvit = MobileNetViT(
        num_classes=Config.num_classes,
        num_transformer_layers=2  # 基准参数
    ).to(Config.device)

    # 损失函数：加权交叉熵
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # 优化器：AdamW
    optimizer = optim.AdamW(model_mobilenetvit.parameters(), lr=Config.lr, weight_decay=1e-4)
    # 学习率调度：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr / 10)

    # 训练
    train_model(
        model_mobilenetvit,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        Config.epochs,
        "mobilenetvit_baseline"
    )

    # 3. 训练对比模型：VGG16
    print("Training VGG16...")
    model_vgg16 = VGG16(num_classes=Config.num_classes).to(Config.device)

    optimizer = optim.AdamW(model_vgg16.parameters(), lr=Config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr / 10)

    train_model(
        model_vgg16,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        Config.epochs,
        "vgg16"
    )

    # 4. 训练对比模型：ResNet18
    print("Training GELUResNet18...")
    model_resnet18 = GELUResNet18(num_classes=Config.num_classes).to(Config.device)

    optimizer = optim.AdamW(model_resnet18.parameters(), lr=Config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr / 10)

    train_model(
        model_resnet18,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        Config.epochs,
        "GELUresnet18"
    )


if __name__ == "__main__":
    main()
# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import pandas as pd
# import numpy as np
# import cv2
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns  # 新增：用于混淆矩阵可视化
# from model import MobileNetViT, VGG16, GELUResNet18
#
# # 设置随机种子，保证实验可复现
# torch.manual_seed(42)
# np.random.seed(42)
#
#
# # 配置训练参数
# class Config:
#     data_path = "./processe_data"
#     train_csv = os.path.join(data_path, "train_processed.csv")
#     test_csv = os.path.join(data_path, "test_processed.csv")
#     num_classes = 5
#     img_size = 64
#     batch_size = 32
#     lr = 0.0005
#     epochs = 50
#     num_workers = 4
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     save_path = "models4 32 0.0005 GELU 50"
#     os.makedirs(save_path, exist_ok=True)
#     # 新增：类别名称（根据你的5类交通标志自定义，如["限速", "禁止", "警告", "指示", "其他"]）
#     class_names = [f"Class {i}" for i in range(num_classes)]
#
#
# # 自定义数据集
# class TrafficSignDataset(Dataset):
#     def __init__(self, csv_path, transform=None, is_train=True):
#         self.df = pd.read_csv(csv_path)
#         self.transform = transform
#         self.is_train = is_train
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         img_path = self.df.iloc[idx]['path']
#         label = self.df.iloc[idx]['label']
#         # 读取图像（BGR转RGB）
#         img = cv2.imread(img_path)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         # 应用变换
#         if self.transform:
#             img = self.transform(img)
#         return img, label
#
#
# # 数据变换
# train_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# test_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
#
# # 加载数据
# def load_data():
#     # 加载训练集并划分为训练集和验证集
#     train_dataset = TrafficSignDataset(
#         Config.train_csv,
#         transform=train_transform,
#         is_train=True
#     )
#     # 划分训练集（90%）和验证集（10%）
#     train_indices, val_indices = train_test_split(
#         range(len(train_dataset)),
#         test_size=0.1,
#         random_state=42,
#         stratify=train_dataset.df['label']
#     )
#     # 创建子集
#     train_subset = torch.utils.data.Subset(train_dataset, train_indices)
#     val_subset = torch.utils.data.Subset(train_dataset, val_indices)
#
#     # 测试集
#     test_dataset = TrafficSignDataset(
#         Config.test_csv,
#         transform=test_transform,
#         is_train=False
#     )
#
#     # 数据加载器
#     train_loader = DataLoader(
#         train_subset,
#         batch_size=Config.batch_size,
#         shuffle=True,
#         num_workers=Config.num_workers
#     )
#     val_loader = DataLoader(
#         val_subset,
#         batch_size=Config.batch_size,
#         shuffle=False,
#         num_workers=Config.num_workers
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=Config.batch_size,
#         shuffle=False,
#         num_workers=Config.num_workers
#     )
#
#     return train_loader, val_loader, test_loader
#
#
# # 计算类别权重（解决类别不平衡）
# def compute_class_weights(dataset):
#     labels = dataset.df['label'].values
#     class_counts = np.bincount(labels)
#     total = len(labels)
#     weights = total / (len(class_counts) * class_counts)
#     return torch.FloatTensor(weights).to(Config.device)
#
#
# # 新增：生成混淆矩阵（原始+归一化）
# def plot_confusion_matrix(model, test_loader, model_name):
#     """
#     生成并保存混淆矩阵（原始数值 + 归一化百分比）
#     :param model: 训练好的模型
#     :param test_loader: 测试集DataLoader
#     :param model_name: 模型名称（用于保存文件名）
#     """
#     model.eval()
#     all_preds = []
#     all_labels = []
#
#     # 在测试集上推理
#     with torch.no_grad():
#         for images, labels in tqdm(test_loader, desc=f"Generating CM for {model_name}"):
#             images = images.to(Config.device)
#             outputs = model(images)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.numpy())
#
#     # 计算混淆矩阵
#     cm = confusion_matrix(all_labels, all_preds)
#     # 计算归一化混淆矩阵（按行归一化，即每个类别的预测占比）
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     # 绘制双面板混淆矩阵（原始 + 归一化）
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#
#     # 1. 原始数值混淆矩阵
#     sns.heatmap(
#         cm,
#         ax=ax1,
#         annot=True,
#         fmt='d',  # 显示整数
#         cmap='Blues',
#         xticklabels=Config.class_names,
#         yticklabels=Config.class_names,
#         cbar=True
#     )
#     ax1.set_title(f'{model_name} Confusion Matrix (Raw Counts)')
#     ax1.set_xlabel('Predicted Label')
#     ax1.set_ylabel('True Label')
#
#     # 2. 归一化混淆矩阵（百分比）
#     sns.heatmap(
#         cm_normalized,
#         ax=ax2,
#         annot=True,
#         fmt='.2f',  # 显示两位小数
#         cmap='Blues',
#         xticklabels=Config.class_names,
#         yticklabels=Config.class_names,
#         cbar=True
#     )
#     ax2.set_title(f'{model_name} Confusion Matrix (Normalized)')
#     ax2.set_xlabel('Predicted Label')
#     ax2.set_ylabel('True Label')
#
#     # 保存混淆矩阵
#     plt.tight_layout()
#     cm_save_path = os.path.join(Config.save_path, f"{model_name}_confusion_matrix.png")
#     plt.savefig(cm_save_path)
#     plt.close()
#
#     # 保存混淆矩阵数值（可选，便于后续分析）
#     np.save(os.path.join(Config.save_path, f"{model_name}_cm_raw.npy"), cm)
#     np.save(os.path.join(Config.save_path, f"{model_name}_cm_normalized.npy"), cm_normalized)
#
#     print(f"Confusion matrix saved to: {cm_save_path}")
#     return cm, cm_normalized
#
#
# # 训练函数（修改：训练完成后调用混淆矩阵生成）
# def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs, model_name):
#     # 记录训练过程
#     history = {
#         'train_loss': [],
#         'val_loss': [],
#         'train_acc': [],
#         'val_acc': []
#     }
#     # 最佳验证准确率
#     best_val_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch + 1}/{num_epochs}")
#         print("-" * 10)
#
#         # 训练阶段
#         model.train()
#         train_running_loss = 0.0
#         train_preds = []
#         train_labels = []
#
#         for images, labels in tqdm(train_loader, desc="Training"):
#             images = images.to(Config.device)
#             labels = labels.to(Config.device)
#
#             # 清零梯度
#             optimizer.zero_grad()
#
#             # 前向传播
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#
#             # 反向传播和优化
#             loss.backward()
#             optimizer.step()
#
#             # 统计
#             train_running_loss += loss.item() * images.size(0)
#             _, preds = torch.max(outputs, 1)
#             train_preds.extend(preds.cpu().numpy())
#             train_labels.extend(labels.cpu().numpy())
#
#         # 计算训练集指标
#         train_epoch_loss = train_running_loss / len(train_loader.dataset)
#         train_epoch_acc = accuracy_score(train_labels, train_preds)
#
#         # 验证阶段
#         model.eval()
#         val_running_loss = 0.0
#         val_preds = []
#         val_labels = []
#
#         with torch.no_grad():
#             for images, labels in tqdm(val_loader, desc="Validation"):
#                 images = images.to(Config.device)
#                 labels = labels.to(Config.device)
#
#                 outputs = model(images)
#                 loss = criterion(outputs, labels)
#
#                 val_running_loss += loss.item() * images.size(0)
#                 _, preds = torch.max(outputs, 1)
#                 val_preds.extend(preds.cpu().numpy())
#                 val_labels.extend(labels.cpu().numpy())
#
#         # 计算验证集指标
#         val_epoch_loss = val_running_loss / len(val_loader.dataset)
#         val_epoch_acc = accuracy_score(val_labels, val_preds)
#
#         # 记录历史
#         history['train_loss'].append(train_epoch_loss)
#         history['val_loss'].append(val_epoch_loss)
#         history['train_acc'].append(train_epoch_acc)
#         history['val_acc'].append(val_epoch_acc)
#
#         print(f"Train Loss: {train_epoch_loss:.4f} Acc: {train_epoch_acc:.4f}")
#         print(f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
#
#         # 学习率调度
#         scheduler.step()
#
#         # 保存最佳模型
#         if val_epoch_acc > best_val_acc:
#             best_val_acc = val_epoch_acc
#             torch.save(model.state_dict(), os.path.join(Config.save_path, f"{model_name}_best.pth"))
#             print(f"Saved best model with Val Acc: {best_val_acc:.4f}")
#
#     # 保存最终模型
#     torch.save(model.state_dict(), os.path.join(Config.save_path, f"{model_name}_final.pth"))
#
#     # 绘制损失和准确率曲线
#     plot_training_history(history, model_name)
#
#     # 新增：加载最佳模型并生成混淆矩阵
#     print(f"\nGenerating confusion matrix for {model_name} (best model)...")
#     model.load_state_dict(torch.load(os.path.join(Config.save_path, f"{model_name}_best.pth")))
#     plot_confusion_matrix(model, test_loader, model_name)
#
#     return model, history
#
#
# # 绘制训练历史
# def plot_training_history(history, model_name):
#     plt.figure(figsize=(12, 4))
#
#     # 损失曲线
#     plt.subplot(1, 2, 1)
#     plt.plot(history['train_loss'], label='Train Loss')
#     plt.plot(history['val_loss'], label='Val Loss')
#     plt.title(f'{model_name} Loss Curve')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#
#     # 准确率曲线
#     plt.subplot(1, 2, 2)
#     plt.plot(history['train_acc'], label='Train Acc')
#     plt.plot(history['val_acc'], label='Val Acc')
#     plt.title(f'{model_name} Accuracy Curve')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(Config.save_path, f"{model_name}_history.png"))
#     plt.close()
#
#
# # 主函数：训练目标模型
# def main():
#     # 加载数据
#     train_loader, val_loader, test_loader = load_data()
#
#     # 初始化训练集完整数据集（用于计算类别权重）
#     full_train_dataset = TrafficSignDataset(Config.train_csv, transform=train_transform)
#     class_weights = compute_class_weights(full_train_dataset)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)  # 统一定义损失函数，避免重复
#
#     # 2. 参数调优实验：示例（Transformer层数=3）
#     print("Training MobileNetViT with 3 Transformer layers...")
#     model_mobilenetvit_3layers = MobileNetViT(
#         num_classes=Config.num_classes,
#         num_transformer_layers=3
#     ).to(Config.device)
#
#     optimizer = optim.AdamW(model_mobilenetvit_3layers.parameters(), lr=Config.lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr / 10)
#
#     train_model(
#         model_mobilenetvit_3layers,
#         train_loader,
#         val_loader,
#         test_loader,  # 新增：传入test_loader用于生成混淆矩阵
#         criterion,
#         optimizer,
#         scheduler,
#         Config.epochs,
#         "mobilenetvit_3layers"
#     )
#
#     # 1. 训练改进的MobileNetViT（基准模型）
#     print("Training Improved MobileNetViT...")
#     model_mobilenetvit = MobileNetViT(
#         num_classes=Config.num_classes,
#         num_transformer_layers=2  # 基准参数
#     ).to(Config.device)
#
#     # 优化器：AdamW
#     optimizer = optim.AdamW(model_mobilenetvit.parameters(), lr=Config.lr, weight_decay=1e-4)
#     # 学习率调度：余弦退火
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr / 10)
#
#     # 训练
#     train_model(
#         model_mobilenetvit,
#         train_loader,
#         val_loader,
#         test_loader,  # 新增：传入test_loader
#         criterion,
#         optimizer,
#         scheduler,
#         Config.epochs,
#         "mobilenetvit_baseline"
#     )
#
#     # 3. 训练对比模型：VGG16
#     print("Training VGG16...")
#     model_vgg16 = VGG16(num_classes=Config.num_classes).to(Config.device)
#
#     optimizer = optim.AdamW(model_vgg16.parameters(), lr=Config.lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr / 10)
#
#     train_model(
#         model_vgg16,
#         train_loader,
#         val_loader,
#         test_loader,  # 新增：传入test_loader
#         criterion,
#         optimizer,
#         scheduler,
#         Config.epochs,
#         "vgg16"
#     )
#
#     # 4. 训练对比模型：ResNet18
#     print("Training GELUResNet18...")
#     model_resnet18 = GELUResNet18(num_classes=Config.num_classes).to(Config.device)
#
#     optimizer = optim.AdamW(model_resnet18.parameters(), lr=Config.lr, weight_decay=1e-4)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=Config.lr / 10)
#
#     train_model(
#         model_resnet18,
#         train_loader,
#         val_loader,
#         test_loader,  # 新增：传入test_loader
#         criterion,
#         optimizer,
#         scheduler,
#         Config.epochs,
#         "GELUresnet18"
#     )
#
#
# if __name__ == "__main__":
#     main()