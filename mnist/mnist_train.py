import torch
from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self, file_path):
        self.images, self.labels = self._read_file(file_path)

    def _read_file(self, file_path):
        images = []
        labels = []
        with open(file_path, 'r') as f:
            next(f) # 跳过标题行
            for line in f:
                line = line.rstrip('\n')
                items = line.split(',')
                images.append([float(x) for x in items[1:]])
                labels.append(int(items[0]))
        return images, labels
    
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = torch.tensor(image)
        # 参数为什么这么设置？
        # 基于整个MNIST数据集统计出来的均值和标准差
        image = image / 255.0
        image = (image - 0.1307) / 0.3081
        label = torch.tensor(label)
        return image, label
    
    def __len__(self):
        return len(self.images)
    
# 定义训练集和测试集的DataLoader
batch_size = 64
train_dataset = MNISTDataset("data/mnist_train.csv")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = MNISTDataset('data/mnist_test.csv')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# 定义神经网络
layer_sizes = [28*28, 128, 128, 128, 64, 10]
weights = []
biases = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# in_size 上一层的神经元数
# out_size 当前层的神经元数
for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
    W = torch.randn(in_size, out_size, device=device) * torch.sqrt(torch.tensor(2/in_size))
    b = torch.zeros(out_size, device=device)
    weights.append(W)
    biases.append(b)

# 函数定义
# ReLU
def relu(x):
    return torch.clamp(x, min=0)

# 计算ReLU函数的导数，大于0为1，小于0为0
def relu_grad(x):
    return (x>0).float()

def softmax(x):
    # 防止输入的值过大
    x_exp = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return x_exp / x_exp.sum(dim=1, keepdim=True)

# 交叉熵损失函数
def cross_entropy(pred, labels):
    N = pred.shape[0]
    one_hot = torch.zeros_like(pred)
    one_hot[torch.arange(N), labels] = 1
    loss = -(one_hot * torch.log(pred + 1e-8)).sum() / N
    return loss, one_hot

num_epochs = 15
learning_rate = 0.001

# 前向传播，可以返回输出和类别，也可以只返回类别
"""
def forward(x, weights, biases, return_probs=False):
    activations = [x]
    for W, b in zip(weights[:-1], biases[:-1]):
        z = activations[-1] @ W + b
        a = relu(z)
        activations.append(a)
    z_out = activations[-1] @ weights[-1] + biases[-1]

    if return_probs:
        y_pred = softmax(z_out)
        return y_pred, z_out
    else:
        y_pred = z_out.argmax(dim=1)
        return y_pred
"""

def train():
# 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            x = images.to(device)
            # print(x)
            y = labels.to(device)
            N = x.shape[0]
            # print(N)

            # 前向传播
            activations = [x]
            pre_acts = []
            for W, b in zip(weights[:-1], biases[:-1]):
                z = activations[-1] @ W + b
                pre_acts.append(z)
                a = relu(z)
                activations.append(a)
            # 输出层
            z_out = activations[-1] @ weights[-1] + biases[-1]
            pre_acts.append(z_out)
            y_pred = softmax(z_out)

            # 损失
            # cross_entropy交叉熵损失函数
            loss, one_hot = cross_entropy(y_pred, y)
            total_loss += loss.item()

            # 反向传播
            grads_W = [None] * len(weights)
            grads_b = [None] * len(biases)
            # 输出层梯度
            dL_dz = (y_pred - one_hot) / N # [n, output]
            grads_W[-1] = activations[-1].t() @ dL_dz
            grads_b[-1] = dL_dz.sum(dim=0)
            # 隐层梯度
            for i in range(len(weights)-2, -1, -1):
                dL_dz = dL_dz @ weights[i+1].t() * relu_grad(pre_acts[i])
                grads_W[i] = activations[i].t() @ dL_dz
                grads_b[i] = dL_dz.sum(dim=0)
            
            # 更新参数
            with torch.no_grad():
                for i in range(len(weights)):
                    weights[i] -= learning_rate * grads_W[i]
                    biases[i] -= learning_rate * grads_b[i]

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


    # 保存weights和biases
    torch.save({
        "weights": weights,
        "biases": biases
    }, "model/model.pth")


# 测试集评估
def test():
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            x = images.view(-1, layer_sizes[0]).to(device)
            y = labels.to(device)
            a = x
            for W, b in zip(weights[:-1], biases[:-1]):
                a = relu(a @ W + b)
            logits = a @ weights[-1] + biases[-1]
            preds = logits.argmax(dim=1)
            correct += (preds==y).sum().item()
            total += y.size(0)
        print(f"Test Accuracy: {correct/total*100:.2f}%")

if __name__ == '__main__':
    train()
    test()