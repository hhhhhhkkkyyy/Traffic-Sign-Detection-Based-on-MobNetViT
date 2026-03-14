import os
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import MobileNetViT, VGG16, ResNet18
from train4 import TrafficSignDataset, Config


# 加载测试数据
def load_test_data():
    test_dataset = TrafficSignDataset(
        Config.test_csv,
        transform=test_transform,
        is_train=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )
    return test_loader


# 测试模型
def test_model(model, test_loader, model_name):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {model_name}"):
            images = images.to(Config.device)
            labels = labels.to(Config.device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评价指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n{model_name} Test Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 详细分类报告
    class_names = ["class0", "class1", "class2", "class3", "class4"]
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        digits=4
    ))

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.save_path, f"{model_name}_confusion_matrix.png"))
    plt.close()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'preds': all_preds,
        'labels': all_labels
    }


# 计算模型参数量和推理速度
def evaluate_model_efficiency(model, model_name):
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e6  # 转换为百万

    # 计算推理速度
    dummy_input = torch.randn(1, 3, Config.img_size, Config.img_size).to(Config.device)
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)

    # 计时
    import time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(dummy_input)
    end_time = time.time()

    avg_inference_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒

    print(f"\n{model_name} Efficiency:")
    print(f"Total Parameters: {total_params:.2f} M")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")

    return {
        'params': total_params,
        'inference_time': avg_inference_time
    }


# 主函数：测试所有模型并对比结果
def main():
    # 数据变换（与训练一致）
    global test_transform
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载测试数据
    test_loader = load_test_data()

    # 加载测试数据
    test_loader = load_test_data()

    # 1. 测试改进的MobileNetViT（最优参数）
    model_mobilenetvit = MobileNetViT(
        num_classes=Config.num_classes,
        num_transformer_layers=2
    ).to(Config.device)
    model_mobilenetvit.load_state_dict(torch.load(
        os.path.join(Config.save_path, r"D:\666\PyCharm 2023.3.2\pythonProject\dazuoye\models4 32 0.0005 GELU 50\mobilenetvit_baseline_best.pth"),
        map_location=Config.device
    ))
    mobilenetvit_results = test_model(model_mobilenetvit, test_loader, "MobileNetViT")
    mobilenetvit_efficiency = evaluate_model_efficiency(model_mobilenetvit, "MobileNetViT")

    # 2. 测试改进的MobileNetViT（最优参数）
    model_mobilenetvit = MobileNetViT(
        num_classes=Config.num_classes,
        num_transformer_layers=3
    ).to(Config.device)
    model_mobilenetvit.load_state_dict(torch.load(
        os.path.join(Config.save_path,
                     r"D:\666\PyCharm 2023.3.2\pythonProject\dazuoye\models4 32 0.0005 GELU 50\mobilenetvit_3layers_best.pth"),
        map_location=Config.device
    ))
    mobilenetvit3_results = test_model(model_mobilenetvit, test_loader, "MobileNetViT3")
    mobilenetvit3_efficiency = evaluate_model_efficiency(model_mobilenetvit, "MobileNetViT3")

    # 2. 测试VGG16
    model_vgg16 = VGG16(num_classes=Config.num_classes).to(Config.device)
    model_vgg16.load_state_dict(torch.load(
        os.path.join(Config.save_path, r""),
        map_location=Config.device
    ))
    vgg16_results = test_model(model_vgg16, test_loader, "VGG16")
    vgg16_efficiency = evaluate_model_efficiency(model_vgg16, "VGG16")

    # 3. 测试ResNet18
    model_resnet18 = ResNet18(num_classes=Config.num_classes).to(Config.device)
    model_resnet18.load_state_dict(torch.load(
        os.path.join(Config.save_path, r"D:\666\PyCharm 2023.3.2\pythonProject\dazuoye\models4 32 0.0005 GELU 50\vgg16_best.pth"),
        map_location=Config.device
    ))
    resnet18_results = test_model(model_resnet18, test_loader, "ResNet18")
    resnet18_efficiency = evaluate_model_efficiency(model_resnet18, "ResNet18")

    # 汇总结果并保存
    results_df = pd.DataFrame({
        'Model': ['MobileNetViT', 'MobileNetViT3','VGG16', 'ResNet18'],
        'Accuracy': [
            mobilenetvit_results['accuracy'],
            mobilenetvit3_results['accuracy'],
            vgg16_results['accuracy'],
            resnet18_results['accuracy']
        ],
        'F1 Score': [
            mobilenetvit_results['f1'],
            mobilenetvit3_results['f1'],
            vgg16_results['f1'],
            resnet18_results['f1']
        ],
        'Params (M)': [
            mobilenetvit_efficiency['params'],
            mobilenetvit3_efficiency['params'],
            vgg16_efficiency['params'],
            resnet18_efficiency['params']
        ],
        'Inference Time (ms)': [
            mobilenetvit_efficiency['inference_time'],
            mobilenetvit3_efficiency['inference_time'],
            vgg16_efficiency['inference_time'],
            resnet18_efficiency['inference_time']
        ]
    })

    results_df.to_csv(os.path.join(Config.save_path, "model_comparison.csv"), index=False)
    print("\n模型对比结果已保存至 model_comparison.csv")
    print(results_df)


if __name__ == "__main__":
    main()
