#
# 文件名: train.py
# -----------------
# 导入模型并使用 CIFAR-10 进行训练和测试
#

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm
import sys

try:
    from vit import create_vit_model
except ImportError:
    print("错误: 找不到 vit.py 文件。")
    print("请确保 vit.py 和 train.py 在同一个文件夹中。")
    sys.exit(1)


# --- 辅助函数 ---

def get_dataloaders(batch_size, image_size=224, data_root='../data'):
    """准备 CIFAR-10 数据加载器"""
    print(f"准备数据集，路径: {data_root}")
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    # 使用本地路径，并设置 download=False
    try:
        train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=False,
                                                     transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False,
                                                    transform=test_transform)
    except Exception as e:
        print(f"错误: 无法从 {data_root} 加载数据集。")
        print(f"请确保该路径 '{data_root}' 正确且包含 CIFAR-10 数据。")
        print(f"原始错误: {e}")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("数据集加载成功。")
    return train_loader, test_loader

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    """执行一个训练轮次"""
    model.train()
    total_loss = 0
    loop = tqdm(loader, desc='Training')
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    return total_loss / len(loader)

@torch.no_grad()
def test_model(model, loader, criterion, device):
    """在测试集上评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    loop = tqdm(loader, desc='Testing ')
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# --- 主执行函数 ---

def main():
    # 1. 超参数设置
    EPOCHS = 5
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    NUM_CLASSES = 10
    DIM = 768
    DEPTH = 12
    HEADS = 12
    MLP_DIM = 3072
    DIM_HEAD = 64
    BATCH_SIZE = 32      # 如果 GPU 显存不足 (例如 6-8GB)，请减小
    LEARNING_RATE = 3e-5
    WEIGHT_DECAY = 0.05
    
    # 检查 GPU
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    else:
        print("警告: 未检测到 CUDA。将使用 CPU 训练，速度会非常慢。", file=sys.stderr)
        DEVICE = 'cpu'

    # 2. 获取数据
    train_loader, test_loader = get_dataloaders(BATCH_SIZE, IMAGE_SIZE)
    
    # 3. 创建模型
    #    (使用 vit_base_patch16_224 的默认参数)
    model = create_vit_model(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_dim=MLP_DIM,
        dim_head=DIM_HEAD,
        pretrained=True, # 加载预训练权重
        timm_model_name='vit_base_patch16_224'
    )
    model.to(DEVICE)
    
    # 4. 设置优化器、损失函数和混合精度
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda'))
    
    # 5. 执行训练循环
    print(f"开始在 {DEVICE} 上训练...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        test_loss, test_acc = test_model(model, test_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    print("训练完成！")

if __name__ == "__main__":
    main()