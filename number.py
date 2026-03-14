# import os
# import cv2
# import pandas as pd
# import numpy as np
# from collections import defaultdict
# import random
# from sklearn.model_selection import train_test_split
#
# # -------------------------- 1. 数据路径配置（适配大小写与路径拼接） --------------------------
# # 原始数据根目录（需根据实际环境修改）
# RAW_DATA_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\gtsrb"
# # 训练集/测试集图像根目录（统一小写，与CSV路径区分）
# TRAIN_BASE_PATH = os.path.join(RAW_DATA_PATH, "train")
# TEST_BASE_PATH = os.path.join(RAW_DATA_PATH, "test")
# # 原始CSV文件路径
# TRAIN_CSV = os.path.join(RAW_DATA_PATH, "train.csv")
# TEST_CSV = os.path.join(RAW_DATA_PATH, "test.csv")
#
# # 处理后数据输出路径（自动创建目录）
# OUTPUT_PATH = "./processe_data"
# os.makedirs(os.path.join(OUTPUT_PATH, "train"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_PATH, "test"), exist_ok=True)
#
# # -------------------------- 2. 类别映射（原始小类→自定义大类） --------------------------
# class_mapping = {
#     # 大类0：限速类（原始类别0-8）
#     0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
#     # 大类1：禁止类（原始类别9-16、38-40、42）
#     9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 38: 1, 39: 1, 40: 1, 42: 1,
#     # 大类2：指示类（原始类别17-22、24、28-30）
#     17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 24: 2, 28: 2, 29: 2, 30: 2,
#     # 大类3：警告类（原始类别25-27、31-32、36）
#     25: 3, 26: 3, 27: 3, 31: 3, 32: 3, 36: 3,
#     # 大类4：其他类（原始类别33-35）
#     33: 4, 34: 4, 35: 4
# }
#
#
# # -------------------------- 3. 工具函数（图像清晰度判断、增强、数据平衡） --------------------------
# def is_blurry(image, threshold=50):
#     """计算拉普拉斯方差判断图像清晰度：模糊返回True，清晰返回False"""
#     if len(image.shape) == 3:  # 彩色图像转灰度
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:  # 已为灰度图像
#         gray = image
#     laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return laplacian_var < threshold
#
#
# def augment_image(image):
#     """训练集图像增强：返回增强后的图像列表（含原始图像）"""
#     augmented_imgs = [image]  # 先添加原始图像
#     rows, cols = image.shape[:2]
#
#     # 1. 亮度/对比度调整（模拟不同光照）
#     alpha = np.random.uniform(0.8, 1.2)  # 对比度系数
#     beta = np.random.uniform(-20, 20)  # 亮度偏移
#     img_adj = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
#     augmented_imgs.append(img_adj)
#
#     # 2. 随机旋转（-15°~15°，白色背景填充避免黑边）
#     angle = np.random.uniform(-15, 15)
#     rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
#     img_rot = cv2.warpAffine(image, rot_matrix, (cols, rows), borderValue=(255, 255, 255))
#     augmented_imgs.append(img_rot)
#
#     # 3. 随机裁剪（56×56→resize回64×64，增强鲁棒性）
#     crop_size = 56
#     x = np.random.randint(0, max(1, cols - crop_size))
#     y = np.random.randint(0, max(1, rows - crop_size))
#     img_crop = image[y:y + crop_size, x:x + crop_size]
#     img_crop = cv2.resize(img_crop, (64, 64))
#     augmented_imgs.append(img_crop)
#
#     return augmented_imgs
#
#
# def balance_data(processed_data):
#     """欠采样平衡类别：按最少类别数量裁剪所有类别数据"""
#     # 按标签分组
#     grouped_data = defaultdict(list)
#     for item in processed_data:
#         grouped_data[item['label']].append(item)
#
#     # 确定最少类别样本数
#     min_class_count = min(len(group) for group in grouped_data.values())
#
#     # 欠采样各类别
#     balanced_data = []
#     for label, items in grouped_data.items():
#         balanced_data.extend(random.sample(items, min_class_count))
#
#     return balanced_data
#
#
# # -------------------------- 4. 训练集处理（路径转换+增强+平衡） --------------------------
# def process_train_data():
#     print("=" * 50)
#     print("开始处理训练集...")
#
#     # 读取原始CSV
#     try:
#         df = pd.read_csv(TRAIN_CSV)
#         print(f"成功读取train.csv，共{len(df)}条原始记录")
#     except Exception as e:
#         print(f"读取train.csv失败：{str(e)}")
#         return
#
#     processed_data = []  # 存储（路径, 标签）字典
#     skip_count = 0  # 统计跳过的无效记录
#
#     # 逐行处理数据
#     for idx, row in df.iterrows():
#         # 每处理1000条打印进度
#         if idx % 1000 == 0 and idx != 0:
#             print(f"已处理{idx}条 | 成功{len(processed_data)}条 | 跳过{skip_count}条")
#
#         # 1. 过滤不在映射中的类别
#         class_id = row['ClassId']
#         if class_id not in class_mapping:
#             skip_count += 1
#             continue
#
#         # 2. 转换CSV路径为实际路径（Train→train，适配目录大小写）
#         csv_path = row['Path']
#         img_relative_path = csv_path.replace("Train/", "train/")
#         img_full_path = os.path.join(RAW_DATA_PATH, img_relative_path)
#
#         # 3. 检查图像文件是否存在
#         if not os.path.exists(img_full_path):
#             skip_count += 1
#             continue
#
#         # 4. 读取并验证图像有效性
#         img = cv2.imread(img_full_path)
#         if img is None:  # 读取失败（损坏或格式错误）
#             skip_count += 1
#             continue
#
#         # 5. 过滤模糊图像
#         if is_blurry(img):
#             skip_count += 1
#             continue
#
#         # 6. 统一尺寸为64×64
#         img = cv2.resize(img, (64, 64))
#
#         # 7. 获取大类标签
#         large_class = class_mapping[class_id]
#
#         # 8. 保存原始图像并记录
#         orig_save_name = f"class{large_class}_idx{idx}_orig.jpg"
#         orig_save_path = os.path.join(OUTPUT_PATH, "train", orig_save_name)
#         cv2.imwrite(orig_save_path, img)
#         processed_data.append({
#             'path': orig_save_path,
#             'label': large_class
#         })
#
#         # 9. 生成增强图像（跳过原始图像，避免重复）
#         augmented_imgs = augment_image(img)
#         for aug_idx, aug_img in enumerate(augmented_imgs[1:]):  # [1:]排除原始图
#             aug_save_name = f"class{large_class}_idx{idx}_aug{aug_idx + 1}.jpg"
#             aug_save_path = os.path.join(OUTPUT_PATH, "train", aug_save_name)
#             cv2.imwrite(aug_save_path, aug_img)
#             processed_data.append({
#                 'path': aug_save_path,
#                 'label': large_class
#             })
#
#     # 10. 平衡类别数据
#     if len(processed_data) > 0:
#         balanced_data = balance_data(processed_data)
#         # 保存平衡后的CSV
#         balanced_df = pd.DataFrame(balanced_data)
#         balanced_df.to_csv(os.path.join(OUTPUT_PATH, "train_processed.csv"), index=False)
#
#         # 打印处理结果
#         print(f"\n训练集处理完成！")
#         print(f"原始记录数：{len(df)}")
#         print(f"跳过记录数：{skip_count}")
#         print(f"增强后总数：{len(processed_data)}")
#         print(f"平衡后总数：{len(balanced_data)}")
#         print(f"结果CSV路径：{os.path.join(OUTPUT_PATH, 'train_processed.csv')}")
#     else:
#         print("警告：训练集无有效数据生成！请检查路径或类别映射")
#     print("=" * 50)
#
#
# # -------------------------- 5. 测试集处理（无增强，仅清洗与格式统一） --------------------------
# def process_test_data():
#     print("\n" + "=" * 50)
#     print("开始处理测试集...")
#
#     # 读取原始CSV
#     try:
#         df = pd.read_csv(TEST_CSV)
#         print(f"成功读取test.csv，共{len(df)}条原始记录")
#     except Exception as e:
#         print(f"读取test.csv失败：{str(e)}")
#         return
#
#     processed_data = []
#     skip_count = 0
#
#     # 逐行处理数据
#     for idx, row in df.iterrows():
#         # 每处理500条打印进度
#         if idx % 500 == 0 and idx != 0:
#             print(f"已处理{idx}条 | 成功{len(processed_data)}条 | 跳过{skip_count}条")
#
#         # 1. 过滤不在映射中的类别
#         class_id = row['ClassId']
#         if class_id not in class_mapping:
#             skip_count += 1
#             continue
#
#         # 2. 转换CSV路径为实际路径（Test→test）
#         csv_path = row['Path']
#         img_relative_path = csv_path.replace("Test/", "test/")
#         img_full_path = os.path.join(RAW_DATA_PATH, img_relative_path)
#
#         # 3. 检查图像文件是否存在
#         if not os.path.exists(img_full_path):
#             skip_count += 1
#             continue
#
#         # 4. 读取并验证图像有效性
#         img = cv2.imread(img_full_path)
#         if img is None:
#             skip_count += 1
#             continue
#
#         # 5. 过滤模糊图像
#         if is_blurry(img):
#             skip_count += 1
#             continue
#
#         # 6. 统一尺寸为64×64
#         img = cv2.resize(img, (64, 64))
#
#         # 7. 获取大类标签
#         large_class = class_mapping[class_id]
#
#         # 8. 保存测试集图像（无增强，避免数据泄露）
#         test_save_name = f"class{large_class}_idx{idx}_test.jpg"
#         test_save_path = os.path.join(OUTPUT_PATH, "test", test_save_name)
#         cv2.imwrite(test_save_path, img)
#         processed_data.append({
#             'path': test_save_path,
#             'label': large_class
#         })
#
#     # 保存测试集结果CSV
#     if len(processed_data) > 0:
#         processed_df = pd.DataFrame(processed_data)
#         processed_df.to_csv(os.path.join(OUTPUT_PATH, "test_processed.csv"), index=False)
#
#         print(f"\n测试集处理完成！")
#         print(f"原始记录数：{len(df)}")
#         print(f"跳过记录数：{skip_count}")
#         print(f"处理后总数：{len(processed_data)}")
#         print(f"结果CSV路径：{os.path.join(OUTPUT_PATH, 'test_processed.csv')}")
#     else:
#         print("警告：测试集无有效数据生成！请检查路径或类别映射")
#     print("=" * 50)
#
#
# # -------------------------- 6. 主函数（参数校验+执行处理流程） --------------------------
# if __name__ == "__main__":
#     # 先校验原始数据路径
#     path_checks = [
#         (RAW_DATA_PATH, "原始数据根目录"),
#         (TRAIN_CSV, "train.csv文件"),
#         (TEST_CSV, "test.csv文件"),
#         (TRAIN_BASE_PATH, "train图像目录"),
#         (TEST_BASE_PATH, "test图像目录")
#     ]
#
#     valid = True
#     for path, desc in path_checks:
#         if not os.path.exists(path):
#             print(f"错误：{desc}不存在 → {path}")
#             valid = False
#
#     # 路径校验通过则执行处理
#     if valid:
#         process_train_data()
#         process_test_data()
#         print(f"\n所有数据处理流程结束！")
#         print(f"最终结果目录：{os.path.abspath(OUTPUT_PATH)}")
#         print("提示：后续训练时，需将Config.data_path指向上述目录")

import os
import cv2
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# -------------------------- 1. 路径配置 --------------------------
RAW_DATA_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\gtsrb"
TRAIN_BASE_PATH = os.path.join(RAW_DATA_PATH, "train")
TEST_BASE_PATH = os.path.join(RAW_DATA_PATH, "test")
TRAIN_CSV = os.path.join(RAW_DATA_PATH, "train.csv")
TEST_CSV = os.path.join(RAW_DATA_PATH, "test.csv")
OUTPUT_PATH = "./processedf_data"
os.makedirs(os.path.join(OUTPUT_PATH, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "test"), exist_ok=True)

# -------------------------- 2. 类别映射 --------------------------
class_mapping = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0,
    9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 38: 1, 39: 1, 40: 1, 42: 1,
    17: 2, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 24: 2, 28: 2, 29: 2, 30: 2,
    25: 3, 26: 3, 27: 3, 31: 3, 32: 3, 36: 3,
    33: 4, 34: 4, 35: 4
}


# -------------------------- 3. 工具函数 --------------------------
def is_blurry(image, threshold=50):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian < threshold


def augment_image(image):
    augmented = []
    augmented.append(image)
    alpha = np.random.uniform(0.8, 1.2)
    beta = np.random.uniform(-20, 20)
    img_adj = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    augmented.append(img_adj)
    rows, cols = image.shape[:2]
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    img_rot = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
    augmented.append(img_rot)
    crop_size = 56
    x = np.random.randint(0, max(1, image.shape[1] - crop_size))
    y = np.random.randint(0, max(1, image.shape[0] - crop_size))
    img_crop = image[y:y + crop_size, x:x + crop_size]
    img_crop = cv2.resize(img_crop, (64, 64))
    augmented.append(img_crop)
    return augmented


# -------------------------- 4. 平衡数据集通用函数 --------------------------
def balance_dataset(processed_data, output_csv_path, dataset_name):
    """对处理后的数据集进行平衡（过采样+欠采样）"""
    if not processed_data:
        print(f"警告：{dataset_name}无有效数据，跳过平衡步骤")
        return

    # 提取标签和路径
    labels = [d['label'] for d in processed_data]
    paths = [d['path'] for d in processed_data]

    # 统计原始分布
    print(f"\n{dataset_name}原始类别分布：")
    print(Counter(labels))

    # 过采样（少数类重复）+ 欠采样（多数类删减）
    ros = RandomOverSampler(random_state=42)  # 过采样器
    rus = RandomUnderSampler(random_state=42)  # 欠采样器

    # 转换为numpy数组（过采样要求二维输入）
    paths_np = np.array(paths).reshape(-1, 1)
    labels_np = np.array(labels)

    # 先过采样到多数类水平，再过欠采样到平衡
    paths_resampled, labels_resampled = ros.fit_resample(paths_np, labels_np)
    paths_balanced, labels_balanced = rus.fit_resample(paths_resampled, labels_resampled)

    # 生成平衡后的数据列表
    balanced_data = [
        {'path': path[0], 'label': label}
        for path, label in zip(paths_balanced, labels_balanced)
    ]

    # 保存平衡后的CSV
    balanced_df = pd.DataFrame(balanced_data)
    balanced_df.to_csv(output_csv_path, index=False)

    # 打印平衡后分布
    print(f"{dataset_name}平衡后类别分布：")
    print(Counter(labels_balanced))
    print(f"平衡后总样本数：{len(balanced_data)}")
    print(f"平衡后CSV路径：{output_csv_path}\n")


# -------------------------- 5. 处理训练集（含平衡） --------------------------
def process_train_data():
    print("=" * 50)
    print("开始处理训练集...")
    try:
        df = pd.read_csv(TRAIN_CSV)
        print(f"成功读取train.csv，共{len(df)}条原始记录")
    except Exception as e:
        print(f"读取train.csv失败：{str(e)}")
        return

    processed_data = []
    skip_count = 0

    for idx, row in df.iterrows():
        if idx % 1000 == 0 and idx != 0:
            print(f"已处理{idx}条，成功{len(processed_data)}条，跳过{skip_count}条")

        class_id = row['ClassId']
        if class_id not in class_mapping:
            skip_count += 1
            continue

        csv_path = row['Path']
        img_relative_path = csv_path.replace("Train/", "train/")
        img_full_path = os.path.join(RAW_DATA_PATH, img_relative_path)

        if not os.path.exists(img_full_path):
            skip_count += 1
            continue

        img = cv2.imread(img_full_path)
        if img is None:
            skip_count += 1
            continue

        if is_blurry(img):
            skip_count += 1
            continue

        img = cv2.resize(img, (64, 64))
        large_class = class_mapping[class_id]

        # 保存原始图像
        orig_save_name = f"class{large_class}_idx{idx}_orig.jpg"
        orig_save_path = os.path.join(OUTPUT_PATH, "train", orig_save_name)
        cv2.imwrite(orig_save_path, img)
        processed_data.append({'path': orig_save_path, 'label': large_class})

        # 数据增强（仅训练集）
        augmented_imgs = augment_image(img)
        for aug_idx, aug_img in enumerate(augmented_imgs[1:]):
            aug_save_name = f"class{large_class}_idx{idx}_aug{aug_idx + 1}.jpg"
            aug_save_path = os.path.join(OUTPUT_PATH, "train", aug_save_name)
            cv2.imwrite(aug_save_path, aug_img)
            processed_data.append({'path': aug_save_path, 'label': large_class})

    # 平衡训练集
    balance_dataset(
        processed_data,
        os.path.join(OUTPUT_PATH, "train_balanced.csv"),
        "训练集"
    )
    print("=" * 50)


# -------------------------- 6. 处理测试集（含平衡） --------------------------
def process_test_data():
    print("\n" + "=" * 50)
    print("开始处理测试集...")
    try:
        df = pd.read_csv(TEST_CSV)
        print(f"成功读取test.csv，共{len(df)}条原始记录")
    except Exception as e:
        print(f"读取test.csv失败：{str(e)}")
        return

    processed_data = []
    skip_count = 0

    for idx, row in df.iterrows():
        if idx % 500 == 0 and idx != 0:
            print(f"已处理{idx}条，成功{len(processed_data)}条，跳过{skip_count}条")

        class_id = row['ClassId']
        if class_id not in class_mapping:
            skip_count += 1
            continue

        csv_path = row['Path']
        img_relative_path = csv_path.replace("Test/", "test/")
        img_full_path = os.path.join(RAW_DATA_PATH, img_relative_path)

        if not os.path.exists(img_full_path):
            skip_count += 1
            continue

        img = cv2.imread(img_full_path)
        if img is None:
            skip_count += 1
            continue

        if is_blurry(img):
            skip_count += 1
            continue

        img = cv2.resize(img, (64, 64))
        large_class = class_mapping[class_id]

        # 保存测试集图像（无增强，避免数据泄露）
        test_save_name = f"class{large_class}_idx{idx}_test.jpg"
        test_save_path = os.path.join(OUTPUT_PATH, "test", test_save_name)
        cv2.imwrite(test_save_path, img)
        processed_data.append({'path': test_save_path, 'label': large_class})

    # 平衡测试集（与训练集逻辑一致，但不做增强，仅通过复制原始样本平衡）
    balance_dataset(
        processed_data,
        os.path.join(OUTPUT_PATH, "test_balanced.csv"),
        "测试集"
    )
    print("=" * 50)


# -------------------------- 7. 主函数 --------------------------
if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_PATH):
        print(f"错误：原始数据根目录不存在 → {RAW_DATA_PATH}")
    elif not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"错误：train.csv或test.csv不存在，请检查路径")
    elif not os.path.exists(TRAIN_BASE_PATH) or not os.path.exists(TEST_BASE_PATH):
        print(f"错误：原始图像目录（train/test）不存在，请检查路径")
    else:
        process_train_data()
        process_test_data()
        print("\n所有数据处理流程结束！")
        print(f"处理后的数据位于：{os.path.abspath(OUTPUT_PATH)}")
        print("下一步：训练和测试时，请使用 train_balanced.csv 和 test_balanced.csv")
# import pandas as pd
# import os
# from collections import Counter
#
#
# def count_class_samples(csv_path):
#     """统计CSV文件中每个类别的样本数量"""
#     if not os.path.exists(csv_path):
#         print(f"错误：文件不存在 → {csv_path}")
#         return None
#
#     # 读取CSV文件
#     df = pd.read_csv(csv_path)
#
#     # 检查是否包含'label'列
#     if 'label' not in df.columns:
#         print("错误：CSV文件中缺少'label'列")
#         return None
#
#     # 统计每个类别的数量
#     class_counts = Counter(df['label'])
#
#     # 按类别ID排序（确保0,1,2,3,4顺序输出）
#     sorted_counts = sorted(class_counts.items(), key=lambda x: x[0])
#
#     return sorted_counts
#
#
# def print_class_distribution(counts, dataset_name):
#     """打印类别分布统计结果"""
#     if not counts:
#         return
#
#     print(f"\n===== {dataset_name} 类别分布 =====")
#     total = 0
#     for class_id, count in counts:
#         print(f"类别 {class_id}：{count} 张图片")
#         total += count
#     print(f"-------------------------")
#     print(f"总样本数：{total} 张图片")
#     print(f"=========================\n")
#
#
# if __name__ == "__main__":
#     # 请根据实际文件路径修改
#     train_csv = "./processed_data/train_balanced.csv"  # 平衡后的训练集
#     test_csv = "./processed_data/test_processed.csv"  # 处理后的测试集
#
#     # 统计并打印训练集类别分布
#     train_counts = count_class_samples(train_csv)
#     print_class_distribution(train_counts, "训练集")
#
#     # 统计并打印测试集类别分布
#     test_counts = count_class_samples(test_csv)
#     print_class_distribution(test_counts, "测试集")