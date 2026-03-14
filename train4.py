# base_tr.py  |  修复版：2025-06-25
# 主要改动：
# 1. 为每个模型单独创建 optimizer / scheduler，彻底解决“绑错参数”导致准确率 0.09 的 bug
# 2. 封装 build_optimizer / build_scheduler，代码更简洁
# 3. 保留 NAdam，但可随时换回 AdamW（见注释）
# 4. 所有路径、超参仍集中放在 Config，方便后续调实验

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
from model import MobileNetViT, VGG16, ResNet18

torch.manual_seed(42)
np.random.seed(42)


# -------------------- 配置 --------------------
class Config:
    data_path = "./processe_data"
    train_csv = os.path.join(data_path, "train_processed.csv")
    test_csv  = os.path.join(data_path, "test_processed.csv")
    num_classes = 5
    img_size = 64
    batch_size = 64
    lr = 0.0001
    epochs = 50
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = "models3 32 0.0001 50 adam"
    os.makedirs(save_path, exist_ok=True)


# -------------------- 数据集 --------------------
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
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label


# -------------------- 数据增强 --------------------
train_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.RandomRotation(15),  # 模拟标志旋转（±15°）
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 轻微平移（±10%）
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 适应光照变化

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# -------------------- 数据加载 --------------------
def load_data():
    train_dataset = TrafficSignDataset(Config.train_csv, train_transform)
    train_idx, val_idx = train_test_split(
        range(len(train_dataset)),
        test_size=0.1,
        random_state=42,
        stratify=train_dataset.df['label']
    )
    train_sub = torch.utils.data.Subset(train_dataset, train_idx)
    val_sub   = torch.utils.data.Subset(train_dataset, val_idx)

    test_dataset = TrafficSignDataset(Config.test_csv, test_transform)

    train_loader = DataLoader(train_sub, batch_size=Config.batch_size,
                              shuffle=True,  num_workers=Config.num_workers)
    val_loader   = DataLoader(val_sub,   batch_size=Config.batch_size,
                              shuffle=False, num_workers=Config.num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=Config.batch_size,
                              shuffle=False, num_workers=Config.num_workers)
    return train_loader, val_loader, test_loader


# -------------------- 类别权重 --------------------
def compute_class_weights(dataset):
    """计算类别权重，不平衡时对少数类赋予更高权重"""
    labels = dataset.df['label'].values
    counts = np.bincount(labels)
    weights = len(labels) / (len(counts) * counts)
    return torch.FloatTensor(weights).to(Config.device)


# -------------------- 优化器 / 调度器封装 --------------------
def build_optimizer(model, lr):
    """
    统一使用带权重衰减的Adam优化器
    :param model: 模型
    :param lr: 学习率（不同模型可传入不同值）
    """
    return optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),  # Adam默认动量参数
        eps=1e-08,
        weight_decay=1e-4  # 权重衰减（L2正则化），抑制过拟合
    )


def build_scheduler(optimizer,eta_min):
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=eta_min
    )


# -------------------- 训练 / 验证 --------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, preds, labels = 0.0, [], []
    for imgs, lbls in tqdm(loader, desc="Train", leave=False):
        imgs, lbls = imgs.to(Config.device), lbls.to(Config.device)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds.extend(outs.argmax(1).cpu().numpy())
        labels.extend(lbls.cpu().numpy())
    acc = accuracy_score(labels, preds)
    return running_loss / len(loader.dataset), acc


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss, preds, labels = 0.0, [], []
    for imgs, lbls in tqdm(loader, desc="Val", leave=False):
        imgs, lbls = imgs.to(Config.device), lbls.to(Config.device)
        outs = model(imgs)
        loss = criterion(outs, lbls)
        running_loss += loss.item() * imgs.size(0)
        preds.extend(outs.argmax(1).cpu().numpy())
        labels.extend(lbls.cpu().numpy())
    acc = accuracy_score(labels, preds)
    return running_loss / len(loader.dataset), acc


# -------------------- 完整训练流程 --------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, name):
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, Config.epochs + 1):
        print(f"Epoch {epoch}/{Config.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = validate(model, val_loader, criterion)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(Config.save_path, f"{name}_best.pth"))
            print(f"↑ New best val acc: {best_val_acc:.4f}")

    torch.save(model.state_dict(),
               os.path.join(Config.save_path, f"{name}_final.pth"))
    plot_history(history, name)
    return history


# -------------------- 绘图 --------------------
def plot_history(h, name):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h['train_loss'], label='train')
    plt.plot(h['val_loss'], label='val')
    plt.title(f"{name} Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(h['train_acc'], label='train')
    plt.plot(h['val_acc'], label='val')
    plt.title(f"{name} Acc"); plt.xlabel("Epoch"); plt.ylabel("Acc"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_path, f"{name}_history.png"))
    plt.close()


# -------------------- 主函数 --------------------
def main():
    train_loader, val_loader, test_loader = load_data()
    full_train = TrafficSignDataset(Config.train_csv, train_transform)
    weights = compute_class_weights(full_train)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # # 1. MobileNetViT baseline
    # print("\n>>> Training MobileNetViT (baseline) ...")
    # mnvit = MobileNetViT(num_classes=Config.num_classes, num_transformer_layers=2).to(Config.device)
    # opt = build_optimizer(mnvit, lr=0.0005)
    # sched = build_scheduler(opt)
    # train_model(mnvit, train_loader, val_loader, criterion, opt, sched, "mobilenetvit_baseline")
    #
    # # 2. MobileNetViT 3-transformer layers
    # print("\n>>> MobileNetViT (3 layers) ...")
    # mnvit3 = MobileNetViT(num_classes=Config.num_classes, num_transformer_layers=3).to(Config.device)
    # opt = build_optimizer(mnvit3, lr=0.0005)
    # sched = build_scheduler(opt)
    # train_model(mnvit3, train_loader, val_loader, criterion, opt, sched, "mobilenetvit_3layers")

    # 3. VGG16
    print("\n>>> VGG16 ...")
    vgg = VGG16(num_classes=Config.num_classes).to(Config.device)
    opt = build_optimizer(vgg, lr=Config.lr)
    sched = build_scheduler(opt,eta_min=Config.lr/10)
    train_model(vgg, train_loader, val_loader, criterion, opt, sched, "vgg16")

    # 1. MobileNetViT baseline
    print("\n>>> Training MobileNetViT (baseline) ...")
    mnvit = MobileNetViT(num_classes=Config.num_classes, num_transformer_layers=2).to(Config.device)
    opt = build_optimizer(mnvit, lr=Config.lr)
    sched = build_scheduler(opt,eta_min=Config.lr/10)
    train_model(mnvit, train_loader, val_loader, criterion, opt, sched, "mobilenetvit_baseline")

    # 2. MobileNetViT 3-transformer layers
    print("\n>>> MobileNetViT (3 layers) ...")
    mnvit3 = MobileNetViT(num_classes=Config.num_classes, num_transformer_layers=3).to(Config.device)
    opt = build_optimizer(mnvit3, lr=Config.lr)
    sched = build_scheduler(opt,eta_min=Config.lr/10)
    train_model(mnvit3, train_loader, val_loader, criterion, opt, sched, "mobilenetvit_3layers")

    # 4. ResNet18
    print("\n>>> ResNet18 ...")
    res18 = ResNet18(num_classes=Config.num_classes).to(Config.device)
    opt = build_optimizer(res18, lr=Config.lr)
    sched = build_scheduler(opt,eta_min=Config.lr/10)
    train_model(res18, train_loader, val_loader, criterion, opt, sched, "resnet18")


if __name__ == "__main__":
    main()