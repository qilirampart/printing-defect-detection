import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# 1. 数据集类：用于加载图像并进行预处理（包括 Canny 边缘检测）
class SteelDefectDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        初始化数据集
        :param image_dir: 图像所在的主目录（train/valid/test）
        :param transform: 数据预处理操作（如图像转换）
        """
        self.image_dir = image_dir
        self.transform = transform
        self.classes = ['crazing', 'inclusion', 'patches', 'pitted', 'rolled', 'scratches']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}  # 类到标签的映射
        self.image_paths = []
        self.labels = []

        # 遍历每个类别的文件夹
        for label_name in self.classes:
            class_folder = os.path.join(self.image_dir, label_name)
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                if img_path.endswith('.bmp'):  # 仅处理 .bmp 格式的图像
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[label_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取

        # 应用 Canny 边缘检测
        edges = cv2.Canny(image, threshold1=100, threshold2=200)

        # 调整图像大小
        edges_resized = cv2.resize(edges, (32, 32))

        # 归一化
        edges_resized = edges_resized / 255.0
        edges_resized = np.expand_dims(edges_resized, axis=0)  # 增加一个维度，适应 CNN 输入

        # 转换为 PyTorch Tensor
        image_tensor = torch.tensor(edges_resized, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)

        return image_tensor, label_tensor


# 2. 定义 LeNet-5 模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 输入大小调整
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)  # 6 类：crazing, inclusion, patches, pitted, rolled, scratches

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 3. 训练模型
def train_model():
    # 创建 LeNet-5 模型实例
    model = LeNet5()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 用于多分类任务
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 设置训练数据的目录
    train_dir = r"/archive/NEU Metal Surface Defects Data/train"
    train_dataset = SteelDefectDataset(train_dir)

    # 学习率衰减：每 5 个 epoch 降低学习率 2 倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 开始训练
    epochs = 40
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()  # 清空梯度

            # 前向传播
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            running_loss += loss.item()

            # 调整学习率
        scheduler.step()  # 在每个 epoch 后更新学习率

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")
        # 打印当前学习率
        print(f"Learning rate: {scheduler.get_last_lr()[0]}")

    # 保存训练好的模型
    torch.save(model.state_dict(), "lenet5_model.pth")
    print("Model trained and saved.")


# 4. 评估模型
def evaluate(model, valid_loader):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for inputs, labels in valid_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy}%")


# 5. 主函数：训练与评估
if __name__ == "__main__":
    # 训练模型
    train_model()

    # 加载保存的模型
    model = LeNet5()
    model.load_state_dict(torch.load("lenet5_model.pth"))

    # 创建验证集加载器
    valid_dir = r"/archive/NEU Metal Surface Defects Data/valid"
    valid_dataset = SteelDefectDataset(valid_dir)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # 评估模型
    evaluate(model, valid_loader)
