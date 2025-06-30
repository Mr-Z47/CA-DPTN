# CA-DPTN 网络（分类、三分支角点预测、自适应融合、透视矫正）



3分支角点预测模型：

1. **矩形分支** - 专门预测正常视角下的矩形
2. **梯形分支** - 统一预测俯视角和仰视角下的梯形，包含方向信息
3. **平行四边形分支** - 专门预测倾斜视角下的四边形

## 主要创新点

1. **几何约束损失函数**：
   - 矩形约束：相邻边应垂直
   - 梯形约束：对边应平行
   - 平行四边形约束：对边应平行且相等
   
2. **特殊注意力机制**：
   - 梯形注意力：根据梯形方向（上宽或下宽）动态调整特征权重
   - 平行四边形注意力：使用可变形卷积增强倾斜结构的特征提取
   
3. **自适应方向预测**：
   - 梯形分支增加方向预测，区分仰视角（下宽）和俯视角（上宽）
   
4. **分类指导矫正及更好的融合机制**：
   - 基于分类概率的自适应权重融合三个分支的预测结果
     
5. **伪标签生成器**：
   - 无需标注透视变换类型，伪标签生成器会根据设置的约束条件，自动生成伪标签，用于训练
   
## 使用方法

### 训练模型

```bash
python train.py \
  --train_dir ../data/train \
  --val_dir ../data/val \
  --annotation_file ../data/annotations.csv \
  --backbone resnet50 \
  --batch_size 8 \
  --num_epochs 50 \
  --max_size 1024 \
  --visualize_every 1 \
  --checkpoints_dir checkpoints \
  --visualize_samples 1
```

### 测试单张图片

```bash
python test.py \
  --image_path test/images/your_image.jpg \
  --model_path checkpoints/best_resnet50_model.pth \
  --backbone resnet50 \
  --output_dir test/results
```

### 批量测试

```bash
python batch_test.py \
  --input_dir test/images \
  --model_path checkpoints/best_resnet50_model.pth \
  --backbone resnet50 \
  --output_dir test/results
```

## 模型输出说明

模型的输出包含额外信息：

```python
transformed_images, pred_corners, distortion_probs, extra_info = model(images)

# extra_info包含：
# - 'trapezoid_direction': 梯形方向（0表示上宽，1表示下宽）
# - 'rectangle_corners': 矩形分支预测的角点
# - 'trapezoid_corners': 梯形分支预测的角点
# - 'parallelogram_corners': 平行四边形分支预测的角点
```

## 可视化结果示例

可视化包含六个子图：
- 第一行：原始图像+真实标注、融合预测结果、透视矫正后图像
- 第二行：矩形分支预测、梯形分支预测（含方向）、平行四边形分支预测

## 注意事项

1. 使用可变形卷积需要确保安装了最新版的PyTorch和TorchVision
2. 梯形分支的方向预测功能可用于后处理和结果分析
3. 几何约束损失可通过在训练脚本中调整权重来控制其影响力

## 项目结构
```
CA-DPTN/
├── train.py              # 训练脚本
├── test.py               # 单张图片测试脚本
├── batch_test.py         # 批量测试脚本
├── perspective_corrector.py  # 网络模型定义
├── checkpoints/          # 模型权重保存目录
│   ├── best_vgg16_model.pth
│   └── best_resnet50_model.pth
├── test/                 # 测试相关目录
│   ├── images/          # 测试图片目录
│   ├── results/         # 测试结果目录
│   └── models/          # 模型权重目录
└── README.md            # 项目说明文档
```

## 环境要求
- Python 3.6+
- PyTorch 1.7+
- torchvision
- numpy
- matplotlib
- tqdm
- pillow

## 训练指令

### 使用 VGG16 骨干网络
```bash
python train.py \
  --train_dir ../data/train \
  --val_dir ../data/val \
  --annotation_file ../data/annotations.csv \
  --backbone vgg16 \
  --batch_size 8 \
  --num_epochs 50 \
  --max_size 1024 \
  --visualize_every 1 \
  --checkpoints_dir checkpoints \
  --visualize_samples 1
```

### 使用 ResNet50 骨干网络
```bash
python train.py \
  --train_dir ../data/train \
  --val_dir ../data/val \
  --annotation_file ../data/annotations.csv \
  --backbone resnet50 \
  --batch_size 8 \
  --num_epochs 50 \
  --max_size 1024 \
  --visualize_every 1 \
  --checkpoints_dir checkpoints \
  --visualize_samples 1
```

## 测试指令

### 单张图片测试
```bash
# 使用 VGG16
python test.py \
  --image_path test/images/your_image.jpg \
  --model_path checkpoints/best_vgg16_model.pth \
  --backbone vgg16 \
  --output_dir test/results

# 使用 ResNet50
python test.py \
  --image_path test/images/your_image.jpg \
  --model_path checkpoints/best_resnet50_model.pth \
  --backbone resnet50 \
  --output_dir test/results
```

### 批量测试
```bash
# 使用 VGG16
python batch_test.py \
  --input_dir test/images \
  --model_path checkpoints/best_vgg16_model.pth \
  --backbone vgg16 \
  --output_dir test/results

# 使用 ResNet50
python batch_test.py \
  --input_dir test/images \
  --model_path checkpoints/best_resnet50_model.pth \
  --backbone resnet50 \
  --output_dir test/results
```

## 参数说明

### 训练参数
- `--train_dir`: 训练数据目录
- `--val_dir`: 验证数据目录
- `--annotation_file`: 标注文件路径
- `--backbone`: 骨干网络选择（vgg16 或 resnet50）
- `--batch_size`: 批次大小
- `--num_epochs`: 训练轮数
- `--max_size`: 最大图像尺寸
- `--visualize_every`: 每隔多少个epoch可视化一次
- `--checkpoints_dir`: 检查点保存目录
- `--visualize_samples`: 每次可视化的样本数量

### 测试参数
- `--image_path`: 测试图像路径（单张测试）
- `--input_dir`: 输入图像目录（批量测试）
- `--model_path`: 模型权重路径
- `--backbone`: 骨干网络选择（vgg16 或 resnet50）
- `--output_dir`: 输出目录
- `--corners`: 目标角点坐标 [左上x,左上y, 右上x,右上y, 右下x,右下y, 左下x,左下y]
- `--extensions`: 支持的图像扩展名（批量测试）

## 注意事项
1. 确保数据目录结构正确
2. 标注文件格式正确
3. 角点坐标必须在[0,1]范围内
4. 角点顺序必须为：左上、右上、右下、左下
5. ResNet50 模型需要更多显存，可能需要调整 batch_size 
