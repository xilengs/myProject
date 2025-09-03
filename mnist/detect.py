import torch
import cv2
import numpy as np
import os
import argparse
from mnist_train import MNISTDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 读取训练好的参数
checkpoint = torch.load('model/model.pth', map_location=device)
weights = checkpoint['weights']
biases = checkpoint['biases']
"""
print("weights:")
for i, W in enumerate(weights):
    print(f"Layer {i}: shape {W.shape}")
print("biases:")
for i, b in enumerate(biases):
    print(f"Layer: {i}: shape: {b.shape}")
"""

# 前向传播函数
def forward(x, weights, biases):
    a = x
    for W, b in zip(weights[:-1], biases[:-1]):
        a = torch.clamp(a @ W + b, min=0)
    logits = a @ weights[-1] + biases[-1]
    y_pred = logits.argmax(dim=1)
    return y_pred

def get_new_detect_folder(base_bolder):
    i = 1
    while True:
        new_folder = os.path.join(base_bolder, f"detect_{i}")
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            return new_folder
        i += 1

# 处理图片并预测
def predict_image(img_path, save_dir):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28,28))
    x = torch.tensor(img_resized, dtype=torch.float32).view(1,-1)
    x = x / 255.0
    x = (x-0.1307) / 0.3081
    x = x.to(device)

    pred = forward(x, weights, biases).item()

    # 图片太小，放不下太多文字，在文件名上保存预测结果
    """
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_out, f"Pred:{pred}", (10, 30),
             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # 保存结果
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, img_out)
    print(f"图片{img_path} 预测结果:{pred}, 已保存到{save_path}")
    """

    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    save_name = f"{name}_{pred}{ext}"
    save_path = os.path.join(save_dir,save_name)
    cv2.imwrite(save_path, img)
    print(f"已保存: {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='detect',
        description='detect handwritten digits'
    )
    
    parser.add_argument('--img_folder', default='data/mnist_test_png', help='the path of photo')
    parser.add_argument('--base_output', default='output', help='the path for storing the results')

    args = parser.parse_args()

    save_folder = get_new_detect_folder(args.base_output)
    for file in os.listdir(args.img_folder):
        img_path = os.path.join(args.img_folder, file)
        predict_image(img_path, save_folder)
    
    print(f"Done!")