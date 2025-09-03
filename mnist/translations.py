from PIL import Image
import os
import numpy as np
import struct

def read_idx_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images

def read_idx_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def save_images(images, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        img_path = os.path.join(out_dir, f"img_{i}.png")
        Image.fromarray(img).save(img_path)

if __name__ == "__main__":
    # 训练集
    images = read_idx_images("data/train-images.idx3-ubyte")
    labels = read_idx_labels("data/train-labels.idx1-ubyte")  # 如果后续需要标签，可以单独保存
    save_images(images, "data/mnist_train_png")

    # 测试集
    images = read_idx_images("data/t10k-images.idx3-ubyte")
    labels = read_idx_labels("data/t10k-labels.idx1-ubyte")
    save_images(images, "data/mnist_test_png")
