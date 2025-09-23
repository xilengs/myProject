import torch
from NIN import NiN
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import time
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import copy
import os

# 使用torchvision.datasets.CIFAR10直接读取数据
def train_val_data_load(batch_size=64, data_dir="../data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    class NinDataset(Dataset):
        def __init__(self):
            super().__init__()

    val_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return classes, train_loader, val_loader

def train(train_loader, val_loader, save_path='model', num_class=10, init_weights=None, batch_size=64, lr=0.001, epochs=20):
    model = NiN(num_class)
    if init_weights:
        state_dict = torch.load(init_weights)
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

            outputs = model(images)
            pre_lab = torch.argmax(outputs, dim=1)
            # 训练阶段需要两个辅助分类器计算loss
            loss = criterion(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 参数更新

            train_loss += loss.item() * images.size(0)
            train_correct += (pre_lab == labels.data).sum().item()
            train_num += labels.size(0)

            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs}, step{step+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc:{torch.sum((pre_lab == labels.data) / batch_size):.4f}")

        # 验证阶段
        for step, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            model.eval()

            with torch.no_grad():
                # 不需要辅助分类器
                outputs = model(images)
                pre_lab = torch.argmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (pre_lab == labels.data).sum().item()
                val_num += labels.size(0)

                if step % 50 == 0:
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

    torch.save(best_model_wts, os.path.join(dir_path, "nin_best_model.pth"))
    torch.save(model.state_dict(), os.path.join(dir_path, "nin_last_model.pth"))
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
    plt.savefig(os.path.join(dir_path, "nin_output.png"))

    plt.show()

if __name__ == '__main__':
    batch_size = 64
    epochs = 20
    # lr = 0.001
    lr = 0.0005
    save_path = "model"
    # init_weight = None
    init_weight = "model/run2/nin_best_model.pth"

    classes, train_loader, val_loader = train_val_data_load(batch_size=batch_size)
    # result = train_model_process(train_loader, val_loader, 2, save_path, lr, epochs)
    result = train(train_loader, val_loader, save_path, num_class=len(classes), init_weights=init_weight, lr=lr, epochs=epochs)
    matplot_acc_loss(result, save_path)
