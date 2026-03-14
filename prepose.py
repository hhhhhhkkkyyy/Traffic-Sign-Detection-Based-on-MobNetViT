# import os
# import cv2
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import os
# import cv2
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
# from collections import Counter
#
# # -------------------------- 1. 关键修改：适配CSV路径格式（大小写+路径拼接） --------------------------
# # 数据路径配置（注意：CSV中Path是Train/Test，代码中文件夹是train/test，需统一）
# RAW_DATA_PATH = r"D:\666\PyCharm 2023.3.2\pythonProject\gtsrb"
# # 训练集：CSV中Path是"Train/类别/文件名"，需拼接为"gtsrb/train/类别/文件名"（将Train替换为train）
# TRAIN_BASE_PATH = os.path.join(RAW_DATA_PATH, "train")  # 原始训练图像根目录（gtsrb/train）
# # 测试集：CSV中Path是"Test/文件名"，需拼接为"gtsrb/test/文件名"（将Test替换为test）
# TEST_BASE_PATH = os.path.join(RAW_DATA_PATH, "test")  # 原始测试图像根目录（gtsrb/test）
#
# # 原始CSV文件路径
# TRAIN_CSV = os.path.join(RAW_DATA_PATH, "train.csv")
# TEST_CSV = os.path.join(RAW_DATA_PATH, "test.csv")
#
# # 处理后的数据输出路径
# OUTPUT_PATH = "./processed_data"
# os.makedirs(os.path.join(OUTPUT_PATH, "train"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_PATH, "test"), exist_ok=True)
#
# # -------------------------- 2. 类别映射（保持不变，确保覆盖CSV中的ClassId） --------------------------
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
# # -------------------------- 3. 工具函数（添加日志打印，便于排查） --------------------------
# def is_blurry(image, threshold=50):
#     """计算拉普拉斯方差判断图像清晰度（模糊返回True，清晰返回False）"""
#     if len(image.shape) == 3:  # 彩色图像转灰度
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:  # 已为灰度图像
#         gray = image
#     laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return laplacian < threshold
#
#
# def augment_image(image):
#     """对训练集图像进行增强，返回增强后的图像列表（含原始图像）"""
#     augmented = []
#     # 1. 原始图像（必保留）
#     augmented.append(image)
#     # 2. 亮度/对比度调整（模拟不同光照）
#     alpha = np.random.uniform(0.8, 1.2)  # 对比度系数（0.8=降低，1.2=提高）
#     beta = np.random.uniform(-20, 20)  # 亮度偏移（-20=变暗，20=变亮）
#     img_adj = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
#     augmented.append(img_adj)
#     # 3. 随机旋转（-15°~15°，避免标志方向敏感）
#     rows, cols = image.shape[:2]
#     angle = np.random.uniform(-15, 15)
#     # 旋转矩阵（以图像中心为原点，旋转后填充白色背景，避免黑边）
#     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
#     img_rot = cv2.warpAffine(image, M, (cols, rows), borderValue=(255, 255, 255))
#     augmented.append(img_rot)
#     # 4. 随机裁剪（56×56→resize回64×64，增强局部特征鲁棒性）
#     crop_size = 56
#     # 确保裁剪范围不超出图像（避免索引错误）
#     x = np.random.randint(0, max(1, image.shape[1] - crop_size))
#     y = np.random.randint(0, max(1, image.shape[0] - crop_size))
#     img_crop = image[y:y + crop_size, x:x + crop_size]
#     img_crop = cv2.resize(img_crop, (64, 64))  # 恢复为64×64
#     augmented.append(img_crop)
#     return augmented
#
#
# # -------------------------- 4. 处理训练集（核心修改：路径转换+日志打印） --------------------------
# def process_train_data():
#     print("=" * 50)
#     print("开始处理训练集...")
#     try:
#         df = pd.read_csv(TRAIN_CSV)
#         print(f"成功读取train.csv，共{len(df)}条原始记录")
#     except Exception as e:
#         print(f"读取train.csv失败：{str(e)}")
#         return
#
#     processed_data = []
#     skip_count = 0
#
#     for idx, row in df.iterrows():
#         if idx % 1000 == 0 and idx != 0:
#             print(f"已处理{idx}条训练集记录，成功{len(processed_data)}条，跳过{skip_count}条")
#
#         class_id = row['ClassId']
#         if class_id not in class_mapping:
#             skip_count += 1
#             continue
#
#         csv_path = row['Path']
#         img_relative_path = csv_path.replace("Train/", "train/")
#         img_full_path = os.path.join(RAW_DATA_PATH, img_relative_path)
#
#         if not os.path.exists(img_full_path):
#             skip_count += 1
#             continue
#
#         img = cv2.imread(img_full_path)
#         if img is None:
#             skip_count += 1
#             continue
#
#         if is_blurry(img):
#             skip_count += 1
#             continue
#
#         img = cv2.resize(img, (64, 64))
#         large_class = class_mapping[class_id]
#
#         orig_save_name = f"class{large_class}_idx{idx}_orig.jpg"
#         orig_save_path = os.path.join(OUTPUT_PATH, "train", orig_save_name)
#         cv2.imwrite(orig_save_path, img)
#         processed_data.append({
#             'path': orig_save_path,
#             'label': large_class
#         })
#
#         augmented_imgs = augment_image(img)
#         for aug_idx, aug_img in enumerate(augmented_imgs[1:]):
#             aug_save_name = f"class{large_class}_idx{idx}_aug{aug_idx + 1}.jpg"
#             aug_save_path = os.path.join(OUTPUT_PATH, "train", aug_save_name)
#             cv2.imwrite(aug_save_path, aug_img)
#             processed_data.append({
#                 'path': aug_save_path,
#                 'label': large_class
#             })
#
#     # -------------------------- 数据平衡核心逻辑 --------------------------
#     if len(processed_data) > 0:
#         # 1. 提取标签和路径
#         labels = [d['label'] for d in processed_data]
#         paths = [d['path'] for d in processed_data]
#
#         # 2. 统计原始类别分布
#         print("\n原始类别分布：")
#         print(Counter(labels))
#
#         # 3. 过采样（少数类重复生成样本） + 欠采样（多数类随机删除样本）
#         # 目标：所有类别样本数接近最多类的数量（可根据需求调整）
#         ros = RandomOverSampler(random_state=42)
#         rus = RandomUnderSampler(random_state=42)
#
#         # 先过采样（转成numpy数组，过采样需要二维输入）
#         paths_np = np.array(paths).reshape(-1, 1)
#         labels_np = np.array(labels)
#         paths_resampled, labels_resampled = ros.fit_resample(paths_np, labels_np)
#
#         # 再过欠采样（使所有类别数量一致）
#         paths_balanced, labels_balanced = rus.fit_resample(paths_resampled, labels_resampled)
#
#         # 4. 重新生成平衡后的数据集（仅保留路径，图像已保存，无需重复增强）
#         balanced_data = []
#         for path, label in zip(paths_balanced.flatten(), labels_balanced):
#             balanced_data.append({
#                 'path': path,
#                 'label': label
#             })
#
#         # 5. 保存平衡后的CSV
#         balanced_df = pd.DataFrame(balanced_data)
#         balanced_df.to_csv(os.path.join(OUTPUT_PATH, "train_balanced.csv"), index=False)
#
#         print("\n平衡后类别分布：")
#         print(Counter(labels_balanced))
#         print(f"处理后记录数（平衡后）：{len(balanced_data)}")
#         print(f"平衡后的CSV保存路径：{os.path.join(OUTPUT_PATH, 'train_balanced.csv')}")
#     else:
#         print("警告：训练集无有效数据生成！请检查路径或类别映射")
#     print("=" * 50)
#
#
# # -------------------------- 5. 处理测试集（保持不变） --------------------------
# def process_test_data():
#     print("\n" + "=" * 50)
#     print("开始处理测试集...")
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
#     for idx, row in df.iterrows():
#         if idx % 500 == 0 and idx != 0:
#             print(f"已处理{idx}条测试集记录，成功{len(processed_data)}条，跳过{skip_count}条")
#
#         class_id = row['ClassId']
#         if class_id not in class_mapping:
#             skip_count += 1
#             continue
#
#         csv_path = row['Path']
#         img_relative_path = csv_path.replace("Test/", "test/")
#         img_full_path = os.path.join(RAW_DATA_PATH, img_relative_path)
#
#         if not os.path.exists(img_full_path):
#             skip_count += 1
#             continue
#
#         img = cv2.imread(img_full_path)
#         if img is None:
#             skip_count += 1
#             continue
#
#         if is_blurry(img):
#             skip_count += 1
#             continue
#
#         img = cv2.resize(img, (64, 64))
#         large_class = class_mapping[class_id]
#
#         test_save_name = f"class{large_class}_idx{idx}_test.jpg"
#         test_save_path = os.path.join(OUTPUT_PATH, "test", test_save_name)
#         cv2.imwrite(test_save_path, img)
#         processed_data.append({
#             'path': test_save_path,
#             'label': large_class
#         })
#
#     if len(processed_data) > 0:
#         processed_df = pd.DataFrame(processed_data)
#         processed_df.to_csv(os.path.join(OUTPUT_PATH, "test_processed.csv"), index=False)
#         print(f"测试集处理完成！")
#         print(f"原始记录数：{len(df)}")
#         print(f"跳过记录数：{skip_count}")
#         print(f"处理后记录数：{len(processed_data)}")
#         print(f"处理后的CSV保存路径：{os.path.join(OUTPUT_PATH, 'test_processed.csv')}")
#     else:
#         print("警告：测试集无有效数据生成！请检查路径或类别映射")
#     print("=" * 50)
#
#
# # -------------------------- 6. 主函数（执行处理） --------------------------
# if __name__ == "__main__":
#     if not os.path.exists(RAW_DATA_PATH):
#         print(f"错误：原始数据根目录不存在 → {RAW_DATA_PATH}")
#     elif not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
#         print(f"错误：train.csv或test.csv不存在，请检查路径")
#     elif not os.path.exists(TRAIN_BASE_PATH) or not os.path.exists(TEST_BASE_PATH):
#         print(f"错误：原始图像目录（train/test）不存在，请检查路径")
#     else:
#         process_train_data()
#         process_test_data()
#         print("\n所有数据处理流程结束！")
#         print(f"处理后的数据位于：{os.path.abspath(OUTPUT_PATH)}")
#         print("下一步：运行train.py时，使用train_balanced.csv作为训练集")
# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split
#
# # -------------------------- 配置参数（需根据你的实际路径修改） --------------------------
# # 平衡后训练集CSV路径（即process_train_data函数输出的train_balanced.csv路径）
# BALANCED_TRAIN_CSV = r"D:\666\PyCharm 2023.3.2\pythonProject\dazuoye\processe_data\train_processed.csv"
# # 拆分后文件的输出路径（建议与原训练集、测试集输出路径一致）
# OUTPUT_PATH = "./processe_data"
#
#
# # -------------------------- 核心拆分逻辑 --------------------------
# def split_train_val():
#     print("=" * 50)
#     print("开始从训练集拆分验证集（8:2比例）...")
#
#     # 1. 读取平衡后的训练集数据
#     if not os.path.exists(BALANCED_TRAIN_CSV):
#         print(f"错误：未找到平衡后训练集文件，请检查路径：{BALANCED_TRAIN_CSV}")
#         return
#
#     try:
#         balanced_df = pd.read_csv(BALANCED_TRAIN_CSV)
#         print(f"成功读取平衡后训练集，共{len(balanced_df)}条记录")
#         print("原始训练集类别分布：")
#         print(balanced_df['label'].value_counts().sort_index())
#     except Exception as e:
#         print(f"读取平衡后训练集失败：{str(e)}")
#         return
#
#     # 2. 按8:2拆分训练集与验证集
#     # stratify=balanced_df['label']：按标签分层拆分，确保验证集类别分布与原训练集一致
#     train_df, val_df = train_test_split(
#         balanced_df,
#         test_size=0.2,  # 验证集占比20%
#         random_state=42,  # 随机种子，保证拆分结果可复现
#         stratify=balanced_df['label']  # 关键参数：维持类别分布平衡
#     )
#
#     # 3. 保存拆分后的文件
#     # 确保输出文件夹存在
#     os.makedirs(OUTPUT_PATH, exist_ok=True)
#
#     # 保存最终训练集（80%）
#     train_final_path = os.path.join(OUTPUT_PATH, "train_final.csv")
#     train_df.to_csv(train_final_path, index=False)
#
#     # 保存验证集（20%）
#     val_processed_path = os.path.join(OUTPUT_PATH, "val_processed.csv")
#     val_df.to_csv(val_processed_path, index=False)
#
#     # 4. 输出拆分结果
#     print("\n拆分完成！")
#     print(f"最终训练集：{len(train_df)}条记录，保存路径：{train_final_path}")
#     print(f"验证集：{len(val_df)}条记录，保存路径：{val_processed_path}")
#
#     print("\n拆分后类别分布对比：")
#     print("-" * 30)
#     print("最终训练集类别分布：")
#     print(train_df['label'].value_counts().sort_index())
#     print("-" * 30)
#     print("验证集类别分布：")
#     print(val_df['label'].value_counts().sort_index())
#     print("=" * 50)
#
#
# # -------------------------- 执行拆分 --------------------------
# if __name__ == "__main__":
#     split_train_val()


import os
import pandas as pd
from collections import Counter

# -------------------------- 配置路径 --------------------------
OUTPUT_PATH = "./processe_data"  # 与预处理脚本输出路径一致
TRAIN_CSV = os.path.join(OUTPUT_PATH, "train_final.csv")
TEST_CSV = os.path.join(OUTPUT_PATH, "val_processed.csv")

# -------------------------- 统计函数 --------------------------
def count_images(csv_path, dataset_name):
    if not os.path.exists(csv_path):
        print(f"错误：{dataset_name}的CSV文件不存在 → {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        print(f"错误：{dataset_name}的CSV缺少label列")
        return

    label_counts = Counter(df['label'])
    print(f"\n{dataset_name} 每个类别的图片数量：")
    for label in sorted(label_counts.keys()):
        print(f"  类别 {label}: {label_counts[label]} 张")
    print(f"  总计: {len(df)} 张")

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    count_images(TRAIN_CSV, "训练集")
    count_images(TEST_CSV, "验证集")
