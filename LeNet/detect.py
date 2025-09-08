from train import LeNet, MNISTDataset
import os
import torch
from torch.utils.data import DataLoader
import csv
import torch.nn.functional as F
import argparse

# 获取模型
# read_all_model默认为False，即为读取模型参数；True为读取整个模型
def get_model(model_path, read_all_model=False):
    if not os.path.isdir(os.path.join(model_path, "exp1")):
        raise ValueError(f'the address {model_path} has not model info')
    
    i = 2
    while True:
        if not os.path.isdir(os.path.join(model_path, f"exp{i}")):
            model_path = os.path.join(model_path, f"exp{i-1}")
            break

    if read_all_model:
        model_path = os.path.join(model_path, 'model_all.pth')
        model = torch.load(model_path)
    else:
        model = LeNet()
        model_path = os.path.join(model_path, 'model_para.pth')
        model.load_state_dict(torch.load(model_path))
    
    return model

def get_out_path(save_path):        
    i = 1
    while True:
        out_path = os.path.join(save_path, f"run{i}")
        if not os.path.isdir(out_path):
            os.makedirs(out_path)
            break
        i += 1
    return out_path

def detect(model_path, data_path, out_path='out', read_all_model=False, batch_size=64):
    model = get_model(model_path=model_path, read_all_model=read_all_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    data_dataset = MNISTDataset(data_path)
    data_dataLoader = DataLoader(data_dataset, batch_size=batch_size, shuffle=False)
    file_path = os.path.join(get_out_path(out_path), "prediction.csv")
    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label"])
        index = 0
        for images, _ in data_dataLoader:
            images = F.pad(images, pad=(2,2,2,2), mode='constant', value=0)
            images = images.to(device)

            output = model(images)
            pred = torch.argmax(output, dim=1)

            for p in pred:
                writer.writerow([index, p.item()])
                index += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='detect',
        description='detect handwritten digits using LeNet'
    )

    parser.add_argument('--model_path', default='model', help='the model address')
    parser.add_argument('--data_path', default='../mnist/data/mnist_test.csv', help='the data address')
    parser.add_argument('--out_path', default='out', help='the address saving the results')
    parser.add_argument('--read_all_model', action='store_true', help='read all model or just model params')
    parser.add_argument('--batch_size', default=64, type=int, help='the amount data read at one time')

    args = parser.parse_args()
    
    detect(args.model_path, args.data_path, args.out_path, args.read_all_model, args.batch_size)

    print('Done!')
    