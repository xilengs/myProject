# 训练20轮只有90.01%的准确率，效果比手动写的差一点
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 定义模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.model(x)
    
# 定义数据集
class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images, self.labels = self._read_file(file_path)
    
    def _read_file(self, file_path):
        images = []
        labels = []
        with open(file_path, 'r') as f:
            next(f)
            for line in f:
                items = line.strip().split(",")
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
            return images, labels
    
    def __getitem__(self, index):
        image = torch.tensor(self.images[index], dtype=torch.float32).view(-1)
        image = image / 255.0
        image = (image - 0.1307) / 0.3081
        label = torch.tensor(self.labels[index], dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.images)

batch_size = 64
learning_rate = 0.001
num_epochs = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据加载
train_dataset = MNISTDataset(r'data/mnist_train.csv')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset(r'data/mnist_test.csv')
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=True)

# 模型、损失函数、优化器
model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练
def train():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad() # 清理梯度
            loss.backward()     # 反向传播
            optimizer.step()    # 更新参数

            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 测试
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    train()
    test()