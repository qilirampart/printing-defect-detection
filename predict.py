import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# =====================
# 1. 定义 LeNet5 模型
# =====================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 输入1通道，输出6通道
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)  # 输出6类

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)  # 展平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =====================
# 2. 加载模型
# =====================
model = LeNet5()
# 确保你的模型参数文件在当前目录
model.load_state_dict(torch.load("lenet5_model.pth"))
model.eval()  # 评估模式

# =====================
# 3. 图像预处理
# =====================
def preprocess_image(img_path):
    # 1) 检查文件是否存在
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图片不存在: {img_path}")

    # 2) 使用 PIL 灰度读取（解决 PNG 透明通道问题）
    try:
        image = Image.open(img_path).convert('L')
        image = np.array(image)
    except Exception as e:
        raise ValueError(f"无法读取图片: {img_path}\n错误信息: {e}")

    # 3) Canny 边缘检测
    edges = cv2.Canny(image, threshold1=100, threshold2=200)
    if edges is None or edges.size == 0:
        raise ValueError(f"Canny处理失败，图像可能为空: {img_path}")

    # 4) 缩放到32x32
    edges_resized = cv2.resize(edges, (32, 32))

    # 5) 归一化
    edges_resized = edges_resized / 255.0

    # 6) 增加通道维度 [C,H,W]
    edges_resized = np.expand_dims(edges_resized, axis=0)

    # 7) 转为 Tensor 并增加 batch 维度 [1,C,H,W]
    image_tensor = torch.tensor(edges_resized, dtype=torch.float32).unsqueeze(0)

    return image_tensor

# =====================
# 4. 预测函数
# =====================
def predict(img_path):
    image_tensor = preprocess_image(img_path)

    with torch.no_grad():  # 禁止梯度计算
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)

    class_labels = ['crazing', 'inclusion', 'patches', 'pitted', 'rolled', 'scratches']
    predicted_class = class_labels[predicted.item()]

    # 可视化
    image = Image.open(img_path).convert('L')
    plt.imshow(image, cmap='gray')
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

    return predicted_class

# =====================
# 5. 测试预测
# =====================
if __name__ == "__main__":
    # 替换为你的图片路径
    img_path = "C:/Users/psk13/Desktop/ea1bce9f-ce91-491a-a9ff-e48831cd23c2.png"
    predicted_class = predict(img_path)
    print(f"Predicted class: {predicted_class}")
