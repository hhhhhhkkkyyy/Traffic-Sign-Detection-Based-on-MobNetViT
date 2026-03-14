import os
import torch
import numpy as np
import pandas as pd
import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torchvision import transforms
from model import MobileNetViT, VGG16, ResNet18
from train5 import Config
from sklearn.metrics import precision_score, recall_score, f1_score

# ===================== 全局配置 =====================
CLASS_NAMES = ["class0", "class1", "class2", "class3", "class4"]
TEST_IMAGE_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\dazuoye\img_2.png"
MODEL_WEIGHTS = {
    "MobileNetViT": os.path.join(Config.save_path, r"mobilenetvit_baseline_best.pth"),
    "MobileNetViT3": os.path.join(Config.save_path, r"mobilenetvit_3layers_best.pth"),
    "VGG16": os.path.join(Config.save_path, r"vgg16_best.pth"),
    "ResNet18": os.path.join(Config.save_path, r"GELUresnet18_best.pth")
}
# 多标签阈值：概率大于该值则判定为该类别
MULTI_LABEL_THRESHOLD = 0.5



# ===================== 工具函数：适配权重参数名 =====================
def adapt_weight_prefix(state_dict, prefix="resnet."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            new_key = prefix + k
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


# ===================== 数据预处理 =====================
def preprocess_single_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图片：{image_path}")
    image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_for_model = cv2.resize(image_original, (Config.img_size, Config.img_size), interpolation=cv2.INTER_LANCZOS4)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_for_model).unsqueeze(0).to(Config.device)
    return image_original, image_tensor


# ===================== 加载模型 =====================
def load_all_models():
    models = {}

    # 1. MobileNetViT（2层）
    model_mobilenetvit = MobileNetViT(num_classes=Config.num_classes, num_transformer_layers=2).to(Config.device)
    model_mobilenetvit.load_state_dict(torch.load(MODEL_WEIGHTS["MobileNetViT"], map_location=Config.device))
    model_mobilenetvit.eval()
    models["MobileNetViT"] = model_mobilenetvit

    # 2. MobileNetViT（3层）
    model_mobilenetvit3 = MobileNetViT(num_classes=Config.num_classes, num_transformer_layers=3).to(Config.device)
    model_mobilenetvit3.load_state_dict(torch.load(MODEL_WEIGHTS["MobileNetViT3"], map_location=Config.device))
    model_mobilenetvit3.eval()
    models["MobileNetViT3"] = model_mobilenetvit3

    # 3. VGG16
    model_vgg16 = VGG16(num_classes=Config.num_classes).to(Config.device)
    model_vgg16.load_state_dict(torch.load(MODEL_WEIGHTS["VGG16"], map_location=Config.device))
    model_vgg16.eval()
    models["VGG16"] = model_vgg16

    # 4. ResNet18
    model_resnet18 = ResNet18(num_classes=Config.num_classes).to(Config.device)
    resnet_state_dict = torch.load(MODEL_WEIGHTS["ResNet18"], map_location=Config.device)
    resnet_state_dict_adapted = adapt_weight_prefix(resnet_state_dict, prefix="resnet.")
    model_resnet18.load_state_dict(resnet_state_dict_adapted, strict=True)
    model_resnet18.eval()
    models["ResNet18"] = model_resnet18

    print(f"✅ 成功加载 {len(models)} 个模型")
    return models


# ===================== 多标签预测逻辑 =====================
def multi_label_predict(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
        pred_labels = [CLASS_NAMES[i] for i, prob in enumerate(probs) if prob >= threshold]
        pred_indices = [i for i, prob in enumerate(probs) if prob >= threshold]
    return probs, pred_labels, pred_indices


# ===================== 单张图片测试（多标签适配） =====================
def test_single_image(models, image_original, image_tensor):
    single_results = {}
    efficiency_metrics = {}

    for model_name, model in models.items():
        print(f"\n{'=' * 60}")
        print(f"测试模型：{model_name}")
        print(f"{'=' * 60}")

        # 1. 推理速度测试
        import time
        with torch.no_grad():
            for _ in range(10):
                _ = model(image_tensor)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image_tensor)
        inference_time = (time.time() - start_time) * 1000

        # 2. 多标签预测
        probs, pred_labels, pred_indices = multi_label_predict(model, image_tensor, MULTI_LABEL_THRESHOLD)

        # 3. 输出结果
        print(f"预测类别（阈值={MULTI_LABEL_THRESHOLD}）：{pred_labels if pred_labels else '无'}")
        print("各类别概率：")
        for i, (cls_name, prob) in enumerate(zip(CLASS_NAMES, probs)):
            print(f"  {cls_name}: {prob:.4f} {'★' if prob >= MULTI_LABEL_THRESHOLD else ''}")

        # 4. 模型参数量
        total_params = sum(p.numel() for p in model.parameters()) / 1e6

        # 5. 存储结果
        single_results[model_name] = {
            "pred_labels": pred_labels,
            "pred_indices": pred_indices,
            "all_probs": probs.tolist(),
            "threshold": MULTI_LABEL_THRESHOLD
        }
        efficiency_metrics[model_name] = {
            "params (M)": total_params,
            "inference_time (ms)": inference_time
        }

    # ===================== 汇总结果 =====================
    results_data = []
    for model_name, res in single_results.items():
        row = {
            "Model": model_name,
            "Predicted Labels": ", ".join(res["pred_labels"]) if res["pred_labels"] else "无",
            "Params (M)": efficiency_metrics[model_name]["params (M)"],
            "Inference Time (ms)": efficiency_metrics[model_name]["inference_time (ms)"]
        }
        for i, cls in enumerate(CLASS_NAMES):
            row[f"{cls} Probability"] = res["all_probs"][i]
        results_data.append(row)

    results_df = pd.DataFrame(results_data)
    print(f"\n{'=' * 60}")
    print("所有模型多标签测试汇总")
    print(f"{'=' * 60}")
    print(results_df)

    # 保存结果
    results_df.to_csv(os.path.join(Config.save_path, "multi_label_single_test_results1.csv"), encoding='utf-8',
                      index=False)

    # ===================== 多标签可视化（修复插值设置） =====================
    plt.figure(figsize=(25, 18), dpi=100)

    # 子图1：原始图片（核心修复：将插值设置到imshow返回值）
    plt.subplot(2, 3, 1)
    # 保存imshow的返回对象，对其设置插值
    img_obj = plt.imshow(image_original)
    img_obj.set_interpolation('none')  # 正确设置图片插值方式
    plt.title("Original Test Image", fontsize=14)
    plt.axis('off')

    # 子图2-5：每个模型的多标签概率条形图
    model_list = list(single_results.keys())
    for idx, model_name in enumerate(model_list):
        ax = plt.subplot(2, 3, idx + 2)
        probs = single_results[model_name]["all_probs"]
        colors = ['#d62728' if p >= MULTI_LABEL_THRESHOLD else '#1f77b4' for p in probs]
        bars = ax.bar(CLASS_NAMES, probs, color=colors)
        # 添加阈值线
        ax.axhline(y=MULTI_LABEL_THRESHOLD, color='red', linestyle='--', alpha=0.7,
                   label=f"Threshold={MULTI_LABEL_THRESHOLD}")
        # 标注概率值
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{prob:.2f}", ha='center', va='bottom', fontsize=10)
        # 标注预测类别
        pred_labels = single_results[model_name]["pred_labels"]
        ax.set_title(f"{model_name}\nPred: {', '.join(pred_labels) if pred_labels else 'None'}", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.set_ylabel("Probability")

    # 子图6：模型效率对比（参数量+推理时间）
    ax = plt.subplot(2, 3, 6)
    x = np.arange(len(model_list))
    width = 0.35
    # 参数量
    params = [efficiency_metrics[m]["params (M)"] for m in model_list]
    ax.bar(x - width / 2, params, width, label='Params (M)', color='#2ca02c')
    # 推理时间（缩放至0-1区间便于对比）
    inf_times = [efficiency_metrics[m]["inference_time (ms)"] for m in model_list]
    inf_times_norm = [t / max(inf_times) for t in inf_times]
    ax.bar(x + width / 2, inf_times_norm, width, label='Inference Time (Norm)', color='#ff7f0e')
    # 标注
    ax.set_xticks(x)
    ax.set_xticklabels(model_list, rotation=15)
    ax.set_title("Model Efficiency", fontsize=12)
    ax.legend()
    ax.set_ylabel("Value (Normalized)")

    plt.tight_layout(pad=2.0)
    # 保存高分辨率图片
    plt.savefig(
        os.path.join(Config.save_path, "multi_label_visualization1.png"),
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.1
    )
    plt.show()

    return results_df


# ===================== 主函数 =====================
def main():
    print(f"📌 开始预处理图片：{TEST_IMAGE_PATH}")
    image_original, image_tensor = preprocess_single_image(TEST_IMAGE_PATH)

    print("\n📌 加载模型...")
    models = load_all_models()

    print("\n📌 开始多标签单张图片测试...")
    test_single_image(models, image_original, image_tensor)

    print("\n🎉 多标签测试完成！结果已保存至：", Config.save_path)


if __name__ == "__main__":
    main()
#
#
# # ===================== 工具函数：适配权重参数名 =====================
# def adapt_weight_prefix(state_dict, prefix="resnet."):
#     new_state_dict = {}
#     for k, v in state_dict.items():
#         if not k.startswith(prefix):
#             new_key = prefix + k
#         else:
#             new_key = k
#         new_state_dict[new_key] = v
#     return new_state_dict
#
#
# # ===================== 数据预处理 =====================
# def preprocess_single_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"无法读取图片：{image_path}")
#     image_original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 保留原图用于可视化
#     # 模型输入图缩放（不影响原图）
#     image_for_model = cv2.resize(image_original, (Config.img_size, Config.img_size), interpolation=cv2.INTER_LANCZOS4)
#
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     image_tensor = transform(image_for_model).unsqueeze(0).to(Config.device)
#     return image_original, image_tensor
#
#
# # ===================== 加载模型 =====================
# def load_all_models():
#     models = {}
#
#     # 1. MobileNetViT（2层）
#     model_mobilenetvit = MobileNetViT(num_classes=Config.num_classes, num_transformer_layers=2).to(Config.device)
#     model_mobilenetvit.load_state_dict(torch.load(MODEL_WEIGHTS["MobileNetViT"], map_location=Config.device))
#     model_mobilenetvit.eval()
#     models["MobileNetViT"] = model_mobilenetvit
#
#     # 2. MobileNetViT（3层）
#     model_mobilenetvit3 = MobileNetViT(num_classes=Config.num_classes, num_transformer_layers=3).to(Config.device)
#     model_mobilenetvit3.load_state_dict(torch.load(MODEL_WEIGHTS["MobileNetViT3"], map_location=Config.device))
#     model_mobilenetvit3.eval()
#     models["MobileNetViT3"] = model_mobilenetvit3
#
#     # 3. VGG16
#     model_vgg16 = VGG16(num_classes=Config.num_classes).to(Config.device)
#     model_vgg16.load_state_dict(torch.load(MODEL_WEIGHTS["VGG16"], map_location=Config.device))
#     model_vgg16.eval()
#     models["VGG16"] = model_vgg16
#
#     # 4. ResNet18
#     model_resnet18 = ResNet18(num_classes=Config.num_classes).to(Config.device)
#     resnet_state_dict = torch.load(MODEL_WEIGHTS["ResNet18"], map_location=Config.device)
#     resnet_state_dict_adapted = adapt_weight_prefix(resnet_state_dict, prefix="resnet.")
#     model_resnet18.load_state_dict(resnet_state_dict_adapted, strict=True)
#     model_resnet18.eval()
#     models["ResNet18"] = model_resnet18
#
#     print(f"✅ 成功加载 {len(models)} 个模型")
#     return models
#
#
# # ===================== 多标签预测逻辑 =====================
# def multi_label_predict(model, image_tensor, threshold=0.5):
#     """多标签预测：返回每个类别的概率 + 预测的类别列表"""
#     with torch.no_grad():
#         outputs = model(image_tensor)
#         # 多标签用sigmoid（单标签用softmax）
#         probs = torch.sigmoid(outputs).cpu().numpy()[0]
#         # 根据阈值判断是否属于该类别
#         pred_labels = [CLASS_NAMES[i] for i, prob in enumerate(probs) if prob >= threshold]
#         pred_indices = [i for i, prob in enumerate(probs) if prob >= threshold]
#     return probs, pred_labels, pred_indices
#
#
# # ===================== 单张图片测试（多标签适配） =====================
# def test_single_image(models, image_original, image_tensor):
#     single_results = {}
#     efficiency_metrics = {}
#
#     for model_name, model in models.items():
#         print(f"\n{'=' * 60}")
#         print(f"测试模型：{model_name}")
#         print(f"{'=' * 60}")
#
#         # 1. 推理速度测试
#         import time
#         with torch.no_grad():
#             for _ in range(10):
#                 _ = model(image_tensor)
#         start_time = time.time()
#         with torch.no_grad():
#             outputs = model(image_tensor)
#         inference_time = (time.time() - start_time) * 1000
#
#         # 2. 多标签预测
#         probs, pred_labels, pred_indices = multi_label_predict(model, image_tensor, MULTI_LABEL_THRESHOLD)
#
#         # 3. 输出结果
#         print(f"预测类别（阈值={MULTI_LABEL_THRESHOLD}）：{pred_labels if pred_labels else '无'}")
#         print("各类别概率：")
#         for i, (cls_name, prob) in enumerate(zip(CLASS_NAMES, probs)):
#             print(f"  {cls_name}: {prob:.4f} {'★' if prob >= MULTI_LABEL_THRESHOLD else ''}")
#
#         # 4. 模型参数量
#         total_params = sum(p.numel() for p in model.parameters()) / 1e6
#
#         # 5. 存储结果
#         single_results[model_name] = {
#             "pred_labels": pred_labels,
#             "pred_indices": pred_indices,
#             "all_probs": probs.tolist(),
#             "threshold": MULTI_LABEL_THRESHOLD
#         }
#         efficiency_metrics[model_name] = {
#             "params (M)": total_params,
#             "inference_time (ms)": inference_time
#         }
#
#     # ===================== 汇总结果 =====================
#     # 构建结果DataFrame
#     results_data = []
#     for model_name, res in single_results.items():
#         row = {
#             "Model": model_name,
#             "Predicted Labels": ", ".join(res["pred_labels"]) if res["pred_labels"] else "无",
#             "Params (M)": efficiency_metrics[model_name]["params (M)"],
#             "Inference Time (ms)": efficiency_metrics[model_name]["inference_time (ms)"]
#         }
#         # 添加每个类别的概率
#         for i, cls in enumerate(CLASS_NAMES):
#             row[f"{cls} Probability"] = res["all_probs"][i]
#         results_data.append(row)
#
#     results_df = pd.DataFrame(results_data)
#     print(f"\n{'=' * 60}")
#     print("所有模型多标签测试汇总")
#     print(f"{'=' * 60}")
#     print(results_df)
#
#     # 保存结果
#     results_df.to_csv(os.path.join(Config.save_path, "multi_label_single_test_results1.csv"), encoding='utf-8',
#                       index=False)
#
#     # ===================== 多标签可视化 =====================
#     plt.figure(figsize=(25, 18), dpi=100)
#
#     # 子图1：原始图片
#     plt.subplot(2, 3, 1)
#     plt.imshow(image_original)
#     plt.title("Original Test Image", fontsize=14)
#     plt.axis('off')
#     plt.gca().set_interpolation('none')
#
#     # 子图2-5：每个模型的多标签概率条形图
#     model_list = list(single_results.keys())
#     for idx, model_name in enumerate(model_list):
#         ax = plt.subplot(2, 3, idx + 2)
#         probs = single_results[model_name]["all_probs"]
#         # 绘制每个类别的概率，超过阈值标红
#         colors = ['#d62728' if p >= MULTI_LABEL_THRESHOLD else '#1f77b4' for p in probs]
#         bars = ax.bar(CLASS_NAMES, probs, color=colors)
#         # 添加阈值线
#         ax.axhline(y=MULTI_LABEL_THRESHOLD, color='red', linestyle='--', alpha=0.7,
#                    label=f"Threshold={MULTI_LABEL_THRESHOLD}")
#         # 标注概率值
#         for bar, prob in zip(bars, probs):
#             ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
#                     f"{prob:.2f}", ha='center', va='bottom', fontsize=10)
#         # 标注预测类别
#         pred_labels = single_results[model_name]["pred_labels"]
#         ax.set_title(f"{model_name}\nPred: {', '.join(pred_labels) if pred_labels else 'None'}", fontsize=12)
#         ax.set_ylim(0, 1.0)
#         ax.legend()
#         ax.set_ylabel("Probability")
#
#     # 子图6：模型效率对比（参数量+推理时间）
#     ax = plt.subplot(2, 3, 6)
#     x = np.arange(len(model_list))
#     width = 0.35
#     # 参数量
#     params = [efficiency_metrics[m]["params (M)"] for m in model_list]
#     ax.bar(x - width / 2, params, width, label='Params (M)', color='#2ca02c')
#     # 推理时间（缩放至0-1区间便于对比）
#     inf_times = [efficiency_metrics[m]["inference_time (ms)"] for m in model_list]
#     inf_times_norm = [t / max(inf_times) for t in inf_times]  # 归一化
#     ax.bar(x + width / 2, inf_times_norm, width, label='Inference Time (Norm)', color='#ff7f0e')
#     # 标注
#     ax.set_xticks(x)
#     ax.set_xticklabels(model_list, rotation=15)
#     ax.set_title("Model Efficiency", fontsize=12)
#     ax.legend()
#     ax.set_ylabel("Value (Normalized)")
#
#     plt.tight_layout(pad=2.0)
#     # 保存高分辨率图片
#     plt.savefig(
#         os.path.join(Config.save_path, "multi_label_visualization1.png"),
#         dpi=600,
#         bbox_inches='tight',
#         pad_inches=0.1
#     )
#     plt.show()
#
#     return results_df
#
#
# # ===================== 主函数 =====================
# def main():
#     print(f"📌 开始预处理图片：{TEST_IMAGE_PATH}")
#     image_original, image_tensor = preprocess_single_image(TEST_IMAGE_PATH)
#
#     print("\n📌 加载模型...")
#     models = load_all_models()
#
#     print("\n📌 开始多标签单张图片测试...")
#     test_single_image(models, image_original, image_tensor)
#
#     print("\n🎉 多标签测试完成！结果已保存至：", Config.save_path)
#
#
# if __name__ == "__main__":
#     main()