import os
import random
import shutil

image_dir = "data/data"

new_train_image_dir = "data/train/images"
new_test_image_dir = "data/val/images"
train_label_file = "data/train/labels.txt"
test_label_file = "data/val/labels.txt"


label_map = {'cat': 0, 'dog':1}

# 划分数据集
all_images = [f for f in os.listdir(image_dir)]
random.seed(42)
random.shuffle(all_images)

split_idx = int(0.8 * len(all_images))
train_images = all_images[:split_idx]
test_images = all_images[split_idx:]

# 复制 + 重命名 + 生成训练集标签
with open(train_label_file, "w") as f:
    for i, filename in enumerate(train_images):
        ext = os.path.splitext(filename)[1]  # 保留后缀
        new_name = f"train_{i}{ext}"
        shutil.copy(os.path.join(image_dir, filename), os.path.join(new_train_image_dir, new_name))

        for key in label_map:
            if filename.startswith(key):
                f.write(f"{new_name} {label_map[key]}\n")
                break

# 复制 + 重命名 + 生成测试集标签
with open(test_label_file, "w") as f:
    for i, filename in enumerate(test_images):
        ext = os.path.splitext(filename)[1]
        new_name = f"test_{i}{ext}"
        shutil.copy(os.path.join(image_dir, filename), os.path.join(new_test_image_dir, new_name))

        for key in label_map:
            if filename.startswith(key):
                f.write(f"{new_name} {label_map[key]}\n")
                break

print(f"训练集：{len(train_images)} 张图片")
print(f"测试集：{len(test_images)} 张图片")
print("Done!")
