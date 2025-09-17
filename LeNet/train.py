import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义模型
class LeNet(nn.Module):
    def __init__(self):
        # 从python 3开始，可以简化为：super().__init__()
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 =nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)
	
    def forward(self, x):
        # C1 + 激活
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = F.tanh(self.conv3(x))
        # 展平
        x = x.view(-1, 120)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
    
# 定义数据集
class MNISTDataset(Dataset):
    def __init__(self, file_path):
        # 可以不写，因为Dataset.__init__原本几乎是空的
        super().__init__()
        self.images, self.labels = self._read_file(file_path)
    
    def _read_file(self, file_path):
        images = []
        labels = []
        with open(file_path, 'r') as f:
            # 跳过第一行标题
            next(f)
            for line in f:
                items = line.strip().split(",")
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
        return images, labels
    
    def __getitem__(self, index):
        # 把数据转换为28 * 28的tensor
        image = torch.tensor(self.images[index], dtype=torch.float32).view(28, 28)
        # 增加通道，变成 1x28x28
        image = image.unsqueeze(0)
        image = image / 255.0
        image = (image - 0.1307) / 0.3081
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label
    
    def __len__(self):
        return len(self.images)
    
# 定义数据
batch_size = 64
learning_rate = 0.001
epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型、损失函数、优化器
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 保存模型
# save_all默认为False，只保存参数，为True保存整个模型
def save_model(model, save_path, save_all=False):
    i = 1
    while True:
        dir_path = os.path.join(save_path, f"exp{i}")
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
            break
        i += 1
    
    if save_all:
        path = os.path.join(dir_path, "model_all.pth")
        torch.save(model, path)
    else:
        path = os.path.join(dir_path, "model_para.pth")
        torch.save(model.state_dict(), path)
    print(f"模型已保存到{path}")

def train(save_path='model', save_all=False):
    # 数据加载
    train_dataset = MNISTDataset(r'./mnist/data/mnist_train.csv')
    train_dataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_dataLoader:
            # 对图像进行补零，扩充到1x32x32
            images = F.pad(images, pad=(2,2,2,2), mode='constant', value=0)
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataLoader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
    save_model(model=model, save_path=save_path)

def test():
    test_dataset = MNISTDataset(r'./mnist/data/mnist_test.csv')
    test_dataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataLoader:
            images = F.pad(images, pad=(2,2,2,2), mode='constant', value=0)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == '__main__':
    save_path = 'model'
    train(save_path=save_path)
    test()     