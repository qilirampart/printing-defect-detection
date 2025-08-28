👌 完美！代码已经推送到 GitHub 上了，现在只差一个漂亮的 **README.md**，让别人一眼就能明白你的实验做了什么。

---

## 🔹 README.md 模板（中英文双语）

你在项目根目录下新建一个 `README.md` 文件，把下面内容复制进去即可：

```markdown
# Printing Defect Detection / 印刷缺陷检测实验

## 📖 项目简介 | Project Overview
本项目实现了一个基于 **PyTorch + LeNet-5 + Canny 边缘检测** 的金属/印刷表面缺陷检测实验。  
This project implements a **PyTorch + LeNet-5 + Canny Edge Detection** pipeline for detecting defects on metallic/printing surfaces.  

支持的缺陷类型（NEU 数据集 6 类）：  
- crazing (裂纹)  
- inclusion (夹杂)  
- patches (斑点)  
- pitted surface (麻点)  
- rolled-in scale (轧入氧化皮)  
- scratches (划痕)  

---

## 📂 项目结构 | Project Structure
```

printing-defect-detection/
├── archive/                     # NEU Metal Surface Defects Data (6 classes)
├── steel\_defect\_dataset.py       # 数据集类 + LeNet-5 模型 + 训练脚本
├── predict.py                    # 单张图片预测脚本
├── lenet5\_model.pth              # 训练好的模型参数
├── .gitignore
└── README.md

````

---

## ⚙️ 环境依赖 | Requirements
```bash   ”“bash
pip install torch torchvisionPIP安装火炬火炬视觉
pip install opencv-pythonPIP安装openv -python
pip install matplotlib   PIP安装matplotlib
pip install pillow   PIP安装枕
````

---

## 🚀 运行方法 | Usage

### 1. 训练模型 | Train the model

```bash   ”“bash
python steel_defect_dataset.py
```

训练完成后会在当前目录下生成 `lenet5_model.pth`。

### 2. 单张图片预测 | Predict a single image

修改 `predict.py` 中的图片路径：

```python
img_path = "archive/NEU Metal Surface Defects Data/test/crazing/xxx.bmp"
predicted_class = predict(img_path)
print(f"Predicted class: {predicted_class}")
```

运行：

```bash
python predict.py
```

---

## 📊 实验结果 | Results

* 模型结构：LeNet-5
* 输入尺寸：32x32 灰度边缘图像
* 优化器：Adam, lr=0.001, StepLR 衰减
* 数据集：NEU Metal Surface Defects Data (\~1800 张图像, 6 类)

> 实验结果显示模型能够较好地区分不同类型的缺陷。

---

## ✨ TODO / 下一步

* 增加数据增强（旋转、裁剪等）
* 尝试更深的 CNN（ResNet, EfficientNet）
* 与边缘计算实验结合，构建端到端印刷缺陷检测系统

---

✍️ Author: **qilirampart**
📧 Contact: `psk1387981284@gmail.com`

