import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random
from torchvision import transforms

def verify_images(image_folder):
    classes = ['Cat', 'Dog']
    class_to_idx = {'Cat': 0, 'Dog': 1}
    samples = []
    for cls_name in classes:
        cls_dir = os.path.join(image_folder, cls_name)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            path = os.path.join(cls_dir, fname)
            try:
                with Image.open(path) as img:
                    img.verify()
                samples.append((path, class_to_idx[cls_name]))
            except Exception:
                print(f"Warning: Skipping corrupted image {path}")
    return samples

class ImageDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img=img.convert('RGB')
            if self.transform:
                img = self.transform(img)
        return img, label

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 1x1卷积
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
def evaluate(model, test_dataloader):
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(inputs)
            preds = (outputs > 0.5) .float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    
    val_acc = val_correct / val_total
    return val_acc


if __name__ == '__main__':
    data_dir = r'DogCatClassify\data\PetImages'
    batch_size = 64
    img_size = 128
    epochs = 10
    lr = 0.001
    print_step = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_samples = verify_images(data_dir)
    random.seed(42)
    random.shuffle(all_samples)
    train_size = int(len(all_samples) * 0.8)
    train_samples = all_samples[:train_size]
    valid_samples = all_samples[train_size:]

    train_transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomCrop(size=(img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.1
        ),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = ImageDataset(train_samples, train_transform)
    valid_dataset = ImageDataset(valid_samples, valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = CNNModel().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        model.train()
        running_loss = 0.0

        for step, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (step+1) % print_step == 0:
                avg_loss = running_loss / print_step
                print(f' Step[{step+1}] - loss: {avg_loss:.4f}')
                running_loss = 0.0
            
        val_acc = evaluate(model, valid_dataloader)
        print(f'Validation Accuracy after epoch {epoch+1}: {val_acc:.4f}')