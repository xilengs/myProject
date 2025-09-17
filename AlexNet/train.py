import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

num_class = 2

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.BatchNorm2d(96),
            # nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),
            # 重叠池化
            nn.MaxPool2d(kernel_size=3, stride=2),
            # LRN
            # nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),

            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            # nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),

            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

class AlexDataset(Dataset):
    def __init__(self, image_dir, label_txt, transform=None):
        super().__init__()
        self.images_dir = image_dir
        self.transform = transform
        self.image_label = []

        with open(label_txt, 'r') as f:
            for line in f:
                filename, label = line.strip().split()
                self.image_label.append((filename,int(label)))

    def __len__(self):
        return len(self.image_label)

    def __getitem__(self, idx):
        filename, label = self.image_label[idx]
        image_path = os.path.join(self.images_dir, filename)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            # image = np.array(image)
            # augmented = self.transform(image=image)
            # image = augmented['image']
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    train_image_dir = "data/train/images"
    train_label_txt = "data/train/labels.txt"
    test_image_dir = "data/val/images"
    test_label_txt = "data/val/labels.txt"
    batch_size = 128
    epochs = 99
    lr = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # transforms强化
    train_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.1
        ),
        transforms.RandomRotation(degrees=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Albumentations强化
    train_A_transform = A.Compose([
        A.Resize(227, 227),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.1
        ),
        A.Rotate(limit=50),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    test_A_transform = A.Compose([
        A.Resize(227, 227),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_dataset = AlexDataset(train_image_dir, train_label_txt, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = AlexDataset(test_image_dir, test_label_txt, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = AlexNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        print(f"\nEpoch {epoch+1}/{epochs}")
        running_loss = 0.0
        correct = 0
        total = 0

        for step, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device)

            outputs = model(image)
            loss = criterion(outputs, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            if (step+1) % 50 == 0:
                print(f"step [{step+1}/{len(train_loader)}],"
                      f"Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

        # ----- 验证 ----
        # 每个epoch验证一次
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for image, label in test_loader:
                image, label = image.to(device), label.to(device)

                outputs = model(image)
                loss = criterion(outputs, label)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += label.size(0)
                val_correct += (predicted == label).sum().item()

        val_loss /= len(test_loader)
        val_acc = 100 * val_correct / val_total

        print(f"Train Loss: {val_loss:.4f}, val Acc: {val_acc:.2f}%")