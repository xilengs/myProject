import copy
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from GoogLeNet import GoogLeNet
import os
from torch import optim, nn
import time


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

def train_model_process(model, train_loader, val_loader, lr=0.001, epochs=10):
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

        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            model.train()



if __name__ == '__main__':
    train_image_dir = "./data/DogCat/train/images"
    train_label_txt = "./data/DogCat/train/labels.txt"
    test_image_dir = "./data/DogCat/val/images"
    test_label_txt = "./data/DogCat/val/labels.txt"
    batch_size = 128
    epochs = 99
    lr = 0.001

    train_loader, val_loader = train_val_data_load(train_image_dir, train_label_txt, test_image_dir, test_label_txt, batch_size)

