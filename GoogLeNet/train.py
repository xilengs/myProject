import copy

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from GoogLeNet import GoogLeNet
import os
from torch import optim, nn
import time
import pandas as pd


class GoogLeNetDataset(Dataset):
    def __init__(self, img_dir, label_txt, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.transform = transform
        self.img_label = []

        with open(label_txt, 'r') as f:
            for line in f:
                filename, label = line.strip().split()
                self.img_label.append((filename, int(label)))

    def __len__(self):
        return len(self.img_label)

    def __getitem__(self, idx):
        filename, label = self.img_label[idx]
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

def train_val_data_load(train_img_dir, train_label_txt, val_img_dir, val_label_txt, batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_dataset = GoogLeNetDataset(train_img_dir, train_label_txt, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = GoogLeNetDataset(val_img_dir, val_label_txt, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader

def train_model_process(train_loader, val_loader, num_classes=2, load_model=None, save_path="model", lr=0.001, epochs=10):
    # 传入load_model(模型参数路径)以继续训练
    # 否则重新定义模型并训练
    model = GoogLeNet(num_classes)
    if load_model:
        state_dict = torch.load(load_model)
        model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict()) # 最佳模型参数
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    since = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        train_loss = 0.0
        train_correct = 0

        val_loss = 0.0
        val_correct = 0

        train_num = 0
        val_num = 0

        # 训练阶段
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            model.train()

            outputs, aux1, aux2 = model(images)
            pre_lab = torch.argmax(outputs, dim=1)
            # 训练阶段需要两个辅助分类器计算loss
            loss = criterion(outputs, labels) + 0.3 * (criterion(aux1, labels) + criterion(aux2, labels))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 参数更新

            train_loss += loss.item() * images.size(0)
            train_correct += (pre_lab == labels.data).sum().item()
            train_num += labels.size(0)

            print(f"Epoch {epoch+1}/{epochs}, step{step+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc:{torch.sum((pre_lab == labels.data) / batch_size):.4f}")

        # 验证阶段
        for step, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            model.eval()

            with torch.no_grad():
                # 不需要辅助分类器
                outputs, _, _ = model(images)
                pre_lab = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (pre_lab == labels.data).sum().item()
                val_num += labels.size(0)
                print(
                    f"Epoch {epoch + 1}/{epochs}, step{step + 1}/{len(val_loader)}, Loss: {loss.item():.4f}, Acc:{torch.sum((pre_lab == labels.data) / batch_size):.4f}")

        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        train_acc = float(train_correct) / train_num
        val_acc = float(val_correct) / val_num
        train_acc_all.append(train_acc)
        val_acc_all.append(val_acc)

        print(f"Train Loss: {train_loss / train_num:.4f}, train Acc: {train_acc:.4f},"
              f"Val Loss: {val_loss / val_num:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n"
          f"Best Val Acc: {best_acc:.4f}")

    i = 1
    while True:
        dir_path = os.path.join(save_path, f"run{i}")
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
            break
        i += 1

    torch.save(best_model_wts, os.path.join(dir_path, "google_net_best_model.pth"))
    torch.save(model.state_dict(), os.path.join(dir_path, "google_net_last_model.pth"))
    train_process = pd.DataFrame(
        data={
            "epoch": range(1, epochs+1),
            "train_loss_all": train_loss_all,
            "val_loss_all": val_loss_all,
            "train_acc_all": train_acc_all,
            "val_acc_all": val_acc_all,
        }
    )

    return train_process

def matplot_acc_loss(train_process, save_path="model"):
    """
    for key in ["epoch", "train_loss_all", "val_loss_all", "train_acc_all", "val_acc_all"]:
        print(key, len(train_process[key]), train_process[key][:5])
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_process["epoch"], train_process["train_loss_all"], label="Train Loss")
    plt.plot(train_process["epoch"], train_process["val_loss_all"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process["train_acc_all"], label="Train Acc")
    plt.plot(train_process["epoch"],train_process["val_acc_all"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()

    plt.tight_layout() # 自动调整
    # plt.ion()

    i = 1
    while True:
        dir_path = os.path.join(save_path, f"run{i}")
        if not os.path.isdir(dir_path):
            dir_path = os.path.join(save_path, f"run{i-1}")
            break
        i += 1
    plt.savefig(os.path.join(dir_path, "google_net_output.png"))

    plt.show()



if __name__ == '__main__':
    train_image_dir = "../data/DogCat/train/images"
    train_label_txt = "../data/DogCat/train/labels.txt"
    test_image_dir = "../data/DogCat/val/images"
    test_label_txt = "../data/DogCat/val/labels.txt"
    batch_size = 128
    epochs = 20
    # lr = 0.001
    lr = 0.0005
    save_path = "model"
    load_model = "model/run3/google_net_best_model.pth"

    train_loader, val_loader = train_val_data_load(train_image_dir, train_label_txt, test_image_dir, test_label_txt, batch_size)
    # result = train_model_process(train_loader, val_loader, 2, save_path, lr, epochs)
    result = train_model_process(train_loader, val_loader, 2, load_model, save_path, lr, epochs)
    matplot_acc_loss(result, save_path)
