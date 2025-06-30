import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
import cv2
from pathlib import Path
import sys
import torch.nn.functional as F

# 设置matplotlib支持中文显示
import matplotlib

# 设置支持中文的字体，根据您的系统进行调整
# 在Windows系统下，常见的中文字体有SimHei, Microsoft YaHei等
try:
    # 尝试设置为微软雅黑字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'Arial Unicode MS',
                                       'Times New Roman']
    # 解决负号'-'显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False
    # 设置DPI以获得更清晰的图像
    plt.rcParams['figure.dpi'] = 150

    # 验证字体设置是否成功
    plt.figure(figsize=(1, 1))
    plt.text(0.5, 0.5, '测试中文字体', fontsize=12, ha='center', va='center')
    plt.close()
    print("成功设置中文字体支持")
except Exception as e:
    print(f"设置中文字体时出现问题: {e}")
    print("将使用系统默认字体")

# 添加父目录到路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入透视矫正网络
from perspective_corrector import PerspectiveCorrectionNetwork, draw_region_box


# 自定义数据集，用于加载图像和角点标注
class PerspectiveCorrectionDataset(Dataset):
    def __init__(self, img_dir, annotation_file=None, transform=None, encoding='utf-8', filename_column='filename',
                 reorder_points=True, debug_mode=False):
        self.img_dir = img_dir
        self.transform = transform
        self.annotations = None
        self.filename_column = filename_column
        self.reorder_points = reorder_points  # 是否重新排序角点
        self.debug_mode = debug_mode  # 是否显示调试信息

        # 加载图像文件列表
        self.img_files = []
        for root, _, files in os.walk(img_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.img_files.append(os.path.join(root, file))

        # 如果提供标注文件，则加载标注
        if annotation_file and os.path.exists(annotation_file):
            try:
                self.annotations = pd.read_csv(annotation_file, encoding=encoding)
                if self.debug_mode:
                    print(f"Successfully read annotation file using {encoding} encoding")
                    # 打印CSV文件的列名，以便调试
                    print(f"CSV file columns: {list(self.annotations.columns)}")

                # 检查filename_column是否存在，不存在则尝试猜测
                if self.filename_column not in self.annotations.columns:
                    # 尝试查找可能的文件名列
                    possible_filename_columns = ['filename', 'file', 'image', 'name', 'img', 'image_name', 'file_name']
                    for col in possible_filename_columns:
                        if col in self.annotations.columns:
                            self.filename_column = col
                            if self.debug_mode:
                                print(f"Using '{col}' as filename column")
                            break

                    # 如果还是找不到，使用第一列作为文件名列
                    if self.filename_column not in self.annotations.columns:
                        self.filename_column = self.annotations.columns[0]
                        if self.debug_mode:
                            print(
                                f"Standard filename column not found, using first column '{self.filename_column}' as filename column")

            except UnicodeDecodeError:
                # 尝试其他常见编码
                encodings_to_try = ['gbk', 'gb2312', 'latin1', 'cp1252', 'big5']
                for enc in encodings_to_try:
                    try:
                        self.annotations = pd.read_csv(annotation_file, encoding=enc)
                        if self.debug_mode:
                            print(f"Successfully read annotation file using {enc} encoding")
                            # 打印CSV文件的列名，以便调试
                            print(f"CSV file columns: {list(self.annotations.columns)}")

                        # 检查filename_column是否存在，不存在则尝试猜测
                        if self.filename_column not in self.annotations.columns:
                            # 尝试查找可能的文件名列
                            possible_filename_columns = ['filename', 'file', 'image', 'name', 'img', 'image_name',
                                                         'file_name']
                            for col in possible_filename_columns:
                                if col in self.annotations.columns:
                                    self.filename_column = col
                                    if self.debug_mode:
                                        print(f"Using '{col}' as filename column")
                                    break

                            # 如果还是找不到，使用第一列作为文件名列
                            if self.filename_column not in self.annotations.columns:
                                self.filename_column = self.annotations.columns[0]
                                if self.debug_mode:
                                    print(
                                        f"Standard filename column not found, using first column '{self.filename_column}' as filename column")

                        break
                    except UnicodeDecodeError:
                        continue
                if self.annotations is None:
                    print(f"Warning: Unable to read annotation file {annotation_file}, using default annotations")

    def __len__(self):
        return len(self.img_files)

    # 确保四个角点按照左上、右上、右下、左下的顺序排列
    def normalize_and_reorder_corners(self, corners, img_width, img_height):
        """
        归一化坐标并重新排序角点

        Args:
            corners: 包含[x1,y1,x2,y2,x3,y3,x4,y4]的坐标数组
            img_width: 图像宽度
            img_height: 图像高度

        Returns:
            归一化并重新排序后的角点坐标
        """
        # 转换为numpy数组并重塑为[4, 2]形状
        corners_array = np.array(corners).reshape(4, 2)

        # 检查是否需要归一化 (如果坐标大于1.0)
        if np.any(corners_array > 1.0):
            # 归一化坐标
            corners_array[:, 0] = corners_array[:, 0] / img_width  # x坐标
            corners_array[:, 1] = corners_array[:, 1] / img_height  # y坐标

        # 如果需要重新排序
        if self.reorder_points:
            # 计算四个点的质心
            center = np.mean(corners_array, axis=0)

            # 计算每个点相对于质心的角度
            angles = np.arctan2(corners_array[:, 1] - center[1],
                                corners_array[:, 0] - center[0])

            # 找到左上角点的索引(角度接近180度或-180度)
            # 将角度映射到[0, 2π]范围
            mapped_angles = np.mod(angles + 2 * np.pi, 2 * np.pi)
            # 左上角应该是角度最大的点(接近π)
            top_left_idx = np.argmax(mapped_angles)

            # 根据左上角点，按照顺时针顺序排列点
            sorted_indices = np.zeros(4, dtype=int)
            for i in range(4):
                sorted_indices[i] = (top_left_idx + i) % 4

            # 重新排序点
            corners_array = corners_array[sorted_indices]

        # 确保所有坐标在[0,1]范围内
        corners_array = np.clip(corners_array, 0, 1)

        # 重新展平为[8]形状
        return corners_array.flatten()

    def __getitem__(self, idx):
        img_path = self.img_files[idx]

        # 加载图像
        try:
            img = Image.open(img_path).convert('RGB')
            # 保存原始图像尺寸
            orig_width, orig_height = img.size
        except Exception as e:
            print(f"Unable to load image {img_path}: {e}")
            # 如果图像加载失败，使用一个黑色图像代替
            img = Image.new('RGB', (224, 224), color=(0, 0, 0))
            orig_width, orig_height = img.size

        # 应用变换
        if self.transform:
            img = self.transform(img)

        # 默认四个角点 (左上, 右上, 右下, 左下)
        corners = torch.tensor([
            0.2, 0.2,  # 左上
            0.8, 0.2,  # 右上
            0.8, 0.8,  # 右下
            0.2, 0.8  # 左下
        ], dtype=torch.float32)

        # 如果有标注，使用标注的角点
        if self.annotations is not None:
            # 查找对应图像的标注
            img_name = os.path.basename(img_path)

            # 尝试直接匹配文件名
            annotation = self.annotations[self.annotations[self.filename_column] == img_name]

            # 如果找不到，尝试通过相对路径匹配
            if annotation.empty:
                try:
                    rel_path = os.path.relpath(img_path, self.img_dir).replace('\\', '/')
                    annotation = self.annotations[self.annotations[self.filename_column] == rel_path]
                except:
                    # 路径处理出错时不中断流程
                    pass

            if not annotation.empty:
                # 检查是否有标准列名
                required_columns = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']

                if all(col in self.annotations.columns for col in required_columns):
                    # 使用命名列获取坐标
                    coords = []
                    for i in range(1, 5):
                        x_col = f'x{i}'
                        y_col = f'y{i}'
                        coords.extend([
                            float(annotation[x_col].values[0]),
                            float(annotation[y_col].values[0])
                        ])
                else:
                    # 假设坐标在第3-10列
                    coords = []
                    for i in range(2, min(10, len(annotation.columns))):
                        try:
                            coords.append(float(annotation.iloc[0, i]))
                        except:
                            # 无法解析时跳过
                            if idx < 5 and self.debug_mode:
                                print(f"Warning: Unable to convert coordinate value: {annotation.iloc[0, i]}")

                # 只有在坐标数量正确时处理
                if len(coords) == 8:
                    # 检查坐标值范围
                    if max(coords) > 1.0:
                        # 需要归一化
                        normalized_coords = []
                        for i in range(0, len(coords), 2):
                            normalized_coords.append(coords[i] / orig_width)
                            normalized_coords.append(coords[i + 1] / orig_height)
                        coords = normalized_coords

                    # 确保所有坐标在0-1范围内
                    coords = [max(0.0, min(1.0, c)) for c in coords]

                    # 使用处理后的坐标
                    corners = torch.tensor(coords, dtype=torch.float32)

        # 返回图像、角点坐标和路径
        return {"image": img, "corners": corners, "path": img_path}


# 仅限制最大尺寸的变换，保持原始宽高比
class LimitMaxSize:
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, img):
        # 获取原始尺寸
        width, height = img.size

        # 如果图像尺寸小于最大尺寸，则不做处理
        if width <= self.max_size and height <= self.max_size:
            return img

        # 否则，等比例缩放
        ratio = min(self.max_size / width, self.max_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # 调整大小但保持比例
        try:
            LANCZOS = Image.Resampling.LANCZOS
        except AttributeError:
            LANCZOS = Image.LANCZOS

        resized_img = img.resize((new_width, new_height), LANCZOS)
        return resized_img


# 计算预测角点和真实角点之间的L1距离
def corner_loss(pred_corners, true_corners):
    return nn.functional.l1_loss(pred_corners, true_corners)


# 计算预测角点和真实角点之间的欧几里得距离误差
def corner_error(pred_corners, true_corners):
    # 重塑为 [batch_size, 4, 2] 格式，方便计算每个点的欧几里得距离
    pred_corners_reshaped = pred_corners.view(-1, 4, 2)
    true_corners_reshaped = true_corners.view(-1, 4, 2)

    # 计算每个角点的欧几里得距离
    # sqrt((x1-x2)^2 + (y1-y2)^2)
    point_distances = torch.sqrt(torch.sum((pred_corners_reshaped - true_corners_reshaped) ** 2, dim=2))

    # 计算每个样本的平均角点误差
    mean_error_per_sample = torch.mean(point_distances, dim=1)

    # 返回批次的平均误差
    return torch.mean(mean_error_per_sample)


# 在 train.py 中找到 corner_loss 和 corner_error 函数，在它们之前添加下面的分类损失函数

def distortion_classification_loss(pred_logits, target_corners):
    """
    根据角点位置自动生成透视变换类型的标签并计算分类损失

    分类为三种类型：
    1. 矩形(正常视角) - 四边长度接近且角度接近90度
    2. 梯形(俯视/仰视角) - 上下边平行但宽度不同
    3. 平行四边形(倾斜视角) - 对边平行但不是矩形或梯形

    Args:
        pred_logits: 预测的分类logits
        target_corners: 真实角点坐标，形状为[batch_size, 8]

    Returns:
        loss: 分类损失
        accuracy: 分类准确率
        targets: 生成的标签，用于监控训练
    """
    batch_size = pred_logits.size(0)
    device = pred_logits.device

    # 重塑为 [batch_size, 4, 2] 格式，方便计算
    corners = target_corners.view(batch_size, 4, 2)

    # 计算四条边的向量
    edges = []
    for i in range(4):
        next_i = (i + 1) % 4
        edge = corners[:, next_i, :] - corners[:, i, :]
        edges.append(edge)

    # 计算每条边的长度
    edge_lengths = [torch.norm(edge, dim=1) for edge in edges]

    # 计算相邻边的夹角（通过点积）
    angles = []
    for i in range(4):
        next_i = (i + 1) % 4
        # 计算单位向量
        v1 = edges[i] / edge_lengths[i].unsqueeze(1)
        v2 = edges[next_i] / edge_lengths[next_i].unsqueeze(1)
        # 计算点积（夹角余弦值）
        cos_angle = torch.sum(v1 * v2, dim=1)
        angles.append(cos_angle)

    # 计算特征指标
    # 1. 矩形度：四个角接近90度(cos接近0)的程度
    rectangle_score = sum(torch.abs(angle) for angle in angles) / 4

    # 2. 梯形度：对边平行性和长度比例
    # 对边0和2，对边1和3
    top_bottom_parallel = torch.abs(edges[0][:, 0] * edges[2][:, 1] - edges[0][:, 1] * edges[2][:, 0]) / (
                edge_lengths[0] * edge_lengths[2])
    left_right_parallel = torch.abs(edges[1][:, 0] * edges[3][:, 1] - edges[1][:, 1] * edges[3][:, 0]) / (
                edge_lengths[1] * edge_lengths[3])

    # 计算上下边长度比例（用于区分梯形）
    top_bottom_ratio = torch.min(edge_lengths[0], edge_lengths[2]) / torch.max(edge_lengths[0], edge_lengths[2])

    # 初始化标签张量，用于存储每个样本的类别
    targets = torch.zeros(batch_size, dtype=torch.long, device=device)

    # 根据特征决定每个样本的类别
    for i in range(batch_size):
        # 检测矩形：四个角接近90度（rectangularity接近0）且长宽比例均衡
        if rectangle_score[i] < 0.3 and top_bottom_parallel[i] < 0.2 and left_right_parallel[i] < 0.2:
            targets[i] = 0  # 矩形
        # 检测梯形：上下边相对平行，但宽度差异明显
        elif (top_bottom_parallel[i] < 0.3 or left_right_parallel[i] < 0.3) and top_bottom_ratio[i] < 0.8:
            targets[i] = 1  # 梯形
        # 其他情况默认为平行四边形（倾斜视角）
        else:
            targets[i] = 2  # 平行四边形

    # 计算分类损失
    cls_loss = F.cross_entropy(pred_logits, targets)

    # 计算分类准确率
    pred_classes = torch.argmax(pred_logits, dim=1)
    accuracy = (pred_classes == targets).float().mean()

    return cls_loss, accuracy, targets


# 训练一个epoch
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    corner_loss_sum = 0
    class_loss_sum = 0
    geometry_loss_sum = 0
    accuracy_sum = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    # 损失权重
    corner_weight = 10.0
    class_weight = 1.0
    geometry_weight = 0.5

    for batch_idx, batch in enumerate(progress_bar):
        # 正确处理字典格式的数据
        images = batch["image"].to(device)
        corners = batch["corners"].to(device)

        # 前向传播 - 注意返回值增加了额外信息
        transformed_images, pred_corners, class_probs, extra_info = model(images, corners)

        # 计算角点损失
        loss_corners = corner_loss(pred_corners, corners)

        # 计算分类损失
        loss_class, accuracy, _ = distortion_classification_loss(class_probs, corners)

        # 计算几何约束损失 - 传递额外信息字典
        geometry_losses = model.compute_geometry_losses(extra_info)
        loss_geometry = sum(geometry_losses.values())

        # 总损失 = 角点损失 + 分类损失 + 几何约束损失
        loss = corner_weight * loss_corners + class_weight * loss_class + geometry_weight * loss_geometry

        # 记录各项损失和准确率
        total_loss += loss.item()
        corner_loss_sum += loss_corners.item()
        class_loss_sum += loss_class.item()
        geometry_loss_sum += loss_geometry.item()
        accuracy_sum += accuracy.item()

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新进度条
        progress_bar.set_description(
            f"Train Epoch: {epoch} "
            f"Loss: {loss.item():.4f} "
            f"Corner: {loss_corners.item():.4f} "
            f"Class: {loss_class.item():.4f} "
            f"Geo: {loss_geometry.item():.4f} "
            f"Acc: {accuracy.item():.4f}"
        )

    # 学习率调整
    if scheduler:
        scheduler.step()

    # 计算平均损失和准确率
    avg_loss = total_loss / len(dataloader)
    avg_corner_loss = corner_loss_sum / len(dataloader)
    avg_class_loss = class_loss_sum / len(dataloader)
    avg_geometry_loss = geometry_loss_sum / len(dataloader)
    avg_accuracy = accuracy_sum / len(dataloader)

    return avg_loss, avg_corner_loss, avg_class_loss, avg_geometry_loss, avg_accuracy


# 验证一个epoch
def validate_epoch(model, dataloader, device, epoch):
    """Validate one epoch"""
    model.eval()
    total_loss = 0
    corner_loss_sum = 0
    class_loss_sum = 0
    geometry_loss_sum = 0
    corner_error_sum = 0
    accuracy_sum = 0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch} [Valid]')

    # 损失权重
    corner_weight = 10.0
    class_weight = 1.0
    geometry_weight = 0.5

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # 正确处理字典格式的数据
            images = batch["image"].to(device)
            corners = batch["corners"].to(device)

            # 前向传播 - 注意返回值增加了额外信息
            transformed_images, pred_corners, class_probs, extra_info = model(images, corners)

            # 计算角点损失和错误率
            loss_corners = corner_loss(pred_corners, corners)
            error = corner_error(pred_corners, corners)

            # 计算分类损失
            loss_class, accuracy, _ = distortion_classification_loss(class_probs, corners)

            # 计算几何约束损失 - 传递额外信息字典
            geometry_losses = model.compute_geometry_losses(extra_info)
            loss_geometry = sum(geometry_losses.values())

            # 总损失
            loss = corner_weight * loss_corners + class_weight * loss_class + geometry_weight * loss_geometry

            # 记录各项损失和指标
            total_loss += loss.item()
            corner_loss_sum += loss_corners.item()
            class_loss_sum += loss_class.item()
            geometry_loss_sum += loss_geometry.item()
            corner_error_sum += error.item()
            accuracy_sum += accuracy.item()

            # 更新进度条
            progress_bar.set_description(
                f"Val Epoch: {epoch} "
                f"Loss: {loss.item():.4f} "
                f"Corner: {loss_corners.item():.4f} "
                f"Error: {error.item():.4f} "
                f"Class: {loss_class.item():.4f} "
                f"Geo: {loss_geometry.item():.4f} "
                f"Acc: {accuracy.item():.4f}"
            )

    # 计算平均损失和指标
    avg_loss = total_loss / len(dataloader)
    avg_corner_loss = corner_loss_sum / len(dataloader)
    avg_class_loss = class_loss_sum / len(dataloader)
    avg_geometry_loss = geometry_loss_sum / len(dataloader)
    avg_corner_error = corner_error_sum / len(dataloader)
    avg_accuracy = accuracy_sum / len(dataloader)

    print(f"\nValidation Epoch: {epoch}")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Avg Corner Loss: {avg_corner_loss:.4f}")
    print(f"Avg Corner Error: {avg_corner_error:.4f}")
    print(f"Avg Class Loss: {avg_class_loss:.4f}")
    print(f"Avg Geometry Loss: {avg_geometry_loss:.4f}")
    print(f"Avg Accuracy: {avg_accuracy:.4f}")

    return avg_loss, avg_corner_loss, avg_class_loss, avg_geometry_loss, avg_corner_error, avg_accuracy


# 可视化验证集样本
def visualize_samples(model, dataloader, device, epoch, output_dir, num_samples=3, debug_mode=False):
    """可视化当前模型的预测结果和透视矫正效果"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # 获取样本
    batch = next(iter(dataloader))
    images = batch["image"]
    corners = batch["corners"]

    if debug_mode:
        print(f"Visualizing samples with shape: {images.shape}, {corners.shape}")

    # 限制样本数量
    num_samples = min(num_samples, images.size(0))
    images = images[:num_samples]
    corners = corners[:num_samples]

    images = images.to(device)
    corners = corners.to(device)

    with torch.no_grad():
        # 前向传播
        transformed_images, pred_corners, class_probs, extra_info = model(images, corners)

        # 获取各分支预测
        rectangle_corners = extra_info['rectangle_corners']
        trapezoid_corners = extra_info['trapezoid_corners']
        parallelogram_corners = extra_info['parallelogram_corners']
        trapezoid_direction = extra_info['trapezoid_direction']

        # 获取分类结果
        _, predicted_classes = torch.max(class_probs, 1)
        class_names = ["矩形", "梯形", "平行四边形"]

    # 转换为CPU张量用于可视化
    images = images.cpu()
    transformed_images = transformed_images.cpu()
    corners = corners.cpu()
    pred_corners = pred_corners.cpu()
    rectangle_corners = rectangle_corners.cpu()
    trapezoid_corners = trapezoid_corners.cpu()
    parallelogram_corners = parallelogram_corners.cpu()
    trapezoid_direction = trapezoid_direction.cpu()

    # 可视化每个样本
    for i in range(num_samples):
        # 创建更大的图
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # 第一行: 原始图像和预测

        # 1. 原始图像和真实标注
        img_tensor = images[i].clone()
        # 反归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = img_tensor * std + mean

        # 在原始图像上绘制真实角点
        overlay = draw_region_box(img_tensor, corners[i], message="真实标注")
        axes[0, 0].imshow(overlay.permute(1, 2, 0))
        axes[0, 0].set_title("原始图像和真实标注", fontproperties='SimHei')
        axes[0, 0].axis('off')

        # 2. 融合预测
        overlay = draw_region_box(
            img_tensor,
            pred_corners[i],
            message=f"融合预测\n类别: {class_names[predicted_classes[i]]}",
            debug_info=True,
            distortion_probs=class_probs[i]
        )
        axes[0, 1].imshow(overlay.permute(1, 2, 0))
        axes[0, 1].set_title("融合预测结果", fontproperties='SimHei')
        axes[0, 1].axis('off')

        # 3. 矫正后图像
        corrected = transformed_images[i].clone()
        corrected = corrected * std + mean
        axes[0, 2].imshow(corrected.permute(1, 2, 0))
        axes[0, 2].set_title("透视矫正后图像", fontproperties='SimHei')
        axes[0, 2].axis('off')

        # 第二行: 各分支预测

        # 1. 矩形分支预测
        overlay = draw_region_box(
            img_tensor,
            rectangle_corners[i],
            message="矩形分支预测"
        )
        axes[1, 0].imshow(overlay.permute(1, 2, 0))
        axes[1, 0].set_title("矩形分支预测", fontproperties='SimHei')
        axes[1, 0].axis('off')

        # 2. 梯形分支预测
        direction_text = "上宽" if trapezoid_direction[i].item() < 0.5 else "下宽"
        overlay = draw_region_box(
            img_tensor,
            trapezoid_corners[i],
            message=f"梯形分支预测\n方向: {direction_text}"
        )
        axes[1, 1].imshow(overlay.permute(1, 2, 0))
        axes[1, 1].set_title("梯形分支预测", fontproperties='SimHei')
        axes[1, 1].axis('off')

        # 3. 平行四边形分支预测
        overlay = draw_region_box(
            img_tensor,
            parallelogram_corners[i],
            message="平行四边形分支预测"
        )
        axes[1, 2].imshow(overlay.permute(1, 2, 0))
        axes[1, 2].set_title("平行四边形分支预测", fontproperties='SimHei')
        axes[1, 2].axis('off')

        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'epoch_{epoch:03d}_sample_{i + 1}.png'), dpi=200)
        plt.close()

    print(f"Saved {num_samples} visualization samples to {output_dir}")


def plot_training_metrics(save_dir, train_losses, val_losses, train_corner_errors, val_corner_errors,
                          train_class_losses, val_class_losses, train_accuracies, val_accuracies,
                          train_geometry_losses, val_geometry_losses,
                          current_epoch, final=False, always_save_latest=True):
    """绘制训练曲线并保存"""

    # 创建包含4个子图的图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 总损失曲线
    ax1 = axes[0, 0]
    ax1.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练')
    ax1.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='验证')
    ax1.set_title('总损失曲线', fontproperties='SimHei')
    ax1.set_xlabel('轮次', fontproperties='SimHei')
    ax1.set_ylabel('损失值', fontproperties='SimHei')
    ax1.legend(prop={'family': 'SimHei'})
    ax1.grid(True)

    # 2. 角点误差曲线
    ax2 = axes[0, 1]
    ax2.plot(range(1, len(train_corner_errors) + 1), train_corner_errors, 'b-', label='训练')
    ax2.plot(range(1, len(val_corner_errors) + 1), val_corner_errors, 'r-', label='验证')
    ax2.set_title('角点误差曲线', fontproperties='SimHei')
    ax2.set_xlabel('轮次', fontproperties='SimHei')
    ax2.set_ylabel('误差', fontproperties='SimHei')
    ax2.legend(prop={'family': 'SimHei'})
    ax2.grid(True)

    # 3. 分类损失曲线
    ax3 = axes[1, 0]
    ax3.plot(range(1, len(train_class_losses) + 1), train_class_losses, 'b-', label='训练')
    ax3.plot(range(1, len(val_class_losses) + 1), val_class_losses, 'r-', label='验证')
    ax3.set_title('分类损失曲线', fontproperties='SimHei')
    ax3.set_xlabel('轮次', fontproperties='SimHei')
    ax3.set_ylabel('损失值', fontproperties='SimHei')
    ax3.legend(prop={'family': 'SimHei'})
    ax3.grid(True)

    # 4. 分类准确率曲线
    ax4 = axes[1, 1]
    ax4.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='训练')
    ax4.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'r-', label='验证')
    ax4.set_title('分类准确率曲线', fontproperties='SimHei')
    ax4.set_xlabel('轮次', fontproperties='SimHei')
    ax4.set_ylabel('准确率', fontproperties='SimHei')
    ax4.legend(prop={'family': 'SimHei'})
    ax4.grid(True)

    plt.tight_layout()

    # 保存图表
    if final:
        # 最终图表
        plt.savefig(os.path.join(save_dir, 'final_training_metrics.png'), dpi=200)
    elif always_save_latest:
        # 保存最新的图表
        plt.savefig(os.path.join(save_dir, 'latest_training_metrics.png'), dpi=200)

    # 每10轮保存一次，或者是最后一轮
    if current_epoch % 10 == 0 or final:
        plt.savefig(os.path.join(save_dir, f'training_metrics_epoch_{current_epoch:03d}.png'), dpi=200)

    plt.close()

    # 创建额外的图表，显示几何约束损失
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, len(train_geometry_losses) + 1), train_geometry_losses, 'b-', label='训练')
    ax.plot(range(1, len(val_geometry_losses) + 1), val_geometry_losses, 'r-', label='验证')
    ax.set_title('几何约束损失曲线', fontproperties='SimHei')
    ax.set_xlabel('轮次', fontproperties='SimHei')
    ax.set_ylabel('损失值', fontproperties='SimHei')
    ax.legend(prop={'family': 'SimHei'})
    ax.grid(True)

    # 保存几何约束损失图表
    if final:
        plt.savefig(os.path.join(save_dir, 'final_geometry_loss.png'), dpi=200)
    elif always_save_latest:
        plt.savefig(os.path.join(save_dir, 'latest_geometry_loss.png'), dpi=200)

    if current_epoch % 10 == 0 or final:
        plt.savefig(os.path.join(save_dir, f'geometry_loss_epoch_{current_epoch:03d}.png'), dpi=200)

    plt.close()


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train Perspective Correction Network')
    parser.add_argument('--train_dir', type=str, required=True, help='Training data directory')
    parser.add_argument('--val_dir', type=str, default=None, help='Validation data directory')
    parser.add_argument('--annotation_file', type=str, default=None, help='Annotation file path')
    parser.add_argument('--encoding', type=str, default='utf-8', help='Annotation file encoding, default is utf-8')
    parser.add_argument('--filename_column', type=str, default='filename',
                        help='Column name representing filename in CSV file')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay rate')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'],
                        help='Backbone network for feature extraction')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume training')
    parser.add_argument('--checkpoints_dir', type=str, default='./perspective_correction_only/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--max_size', type=int, default=1024, help='Maximum image size')
    parser.add_argument('--visualize_every', type=int, default=5, help='Visualize every N epochs')
    parser.add_argument('--visualize_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--reorder_points', action='store_true',
                        help='Reorder annotated points to follow top-left, top-right, bottom-right, bottom-left order')
    parser.add_argument('--debug_mode', action='store_false', help='Enable debug mode to print more information')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode, do not display image information')
    return parser.parse_args()


# 主函数
def main():
    args = parse_args()

    # 创建检查点目录
    os.makedirs(args.checkpoints_dir, exist_ok=True)

    # 创建日志目录
    log_dir = os.path.join(args.checkpoints_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件
    log_file = os.path.join(log_dir, f'training_log_{args.backbone}.csv')
    with open(log_file, 'w') as f:
        f.write(
            'epoch,train_loss,val_loss,train_corner_error,val_corner_error,train_class_loss,val_class_loss,train_accuracy,val_accuracy\n')

    # 创建统一的可视化目录
    vis_dir = os.path.join(args.checkpoints_dir, f'visualizations_{args.backbone}')
    os.makedirs(vis_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 定义数据变换
    transform = transforms.Compose([
        LimitMaxSize(args.max_size),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建数据集
    train_dataset = PerspectiveCorrectionDataset(
        args.train_dir,
        annotation_file=args.annotation_file,
        transform=transform,
        encoding=args.encoding,
        filename_column=args.filename_column,
        reorder_points=args.reorder_points,
        debug_mode=args.debug_mode
    )

    # 验证数据集（如果提供）
    val_dataset = None
    if args.val_dir:
        val_dataset = PerspectiveCorrectionDataset(
            args.val_dir,
            annotation_file=args.annotation_file,
            transform=transform,
            encoding=args.encoding,
            filename_column=args.filename_column,
            reorder_points=args.reorder_points,
            debug_mode=args.debug_mode
        )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 创建模型
    model = PerspectiveCorrectionNetwork(backbone=args.backbone).to(device)
    print(f"Created perspective correction network using {args.backbone} as feature extractor")

    # 定义优化器 - 使用更快的Adam优化器，并提高学习率
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 学习率调度器 - 使用OneCycleLR调度器，帮助快速收敛
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,  # 不再放大10倍
        epochs=args.num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # 前30%的时间提高学习率，然后降低
        div_factor=10,  # 起始学习率 = max_lr/10
        final_div_factor=100,  # 最终学习率 = max_lr/100
    )

    # 记录最佳验证损失
    best_val_loss = float('inf')
    start_epoch = 0

    # 用于收集训练和验证指标的列表
    train_losses = []
    val_losses = []
    train_corner_errors = []
    val_corner_errors = []
    train_class_losses = []
    val_class_losses = []
    train_accuracies = []  # 新增：训练准确率列表
    val_accuracies = []  # 新增：验证准确率列表
    train_geometry_losses = []
    val_geometry_losses = []

    # 如果需要从检查点恢复
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from checkpoint {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 如果检查点包含调度器状态，加载它
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state restored")

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']

        # 如果检查点包含训练历史，则恢复
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
        if 'train_corner_errors' in checkpoint:
            train_corner_errors = checkpoint['train_corner_errors']
            val_corner_errors = checkpoint['val_corner_errors']
        if 'train_class_losses' in checkpoint:
            train_class_losses = checkpoint['train_class_losses']
            val_class_losses = checkpoint['val_class_losses']
        if 'train_accuracies' in checkpoint:
            train_accuracies = checkpoint['train_accuracies']
            val_accuracies = checkpoint['val_accuracies']
        if 'train_geometry_losses' in checkpoint:
            train_geometry_losses = checkpoint['train_geometry_losses']
            val_geometry_losses = checkpoint['val_geometry_losses']

        print(f"Resumed to epoch {start_epoch}, best validation loss: {best_val_loss:.4f}")

    # 训练循环
    for epoch in range(start_epoch, args.num_epochs):
        # 训练一个轮次
        train_loss, train_corner_loss, train_class_loss, train_geometry_loss, train_accuracy = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch)

        # 验证
        val_loss, val_corner_loss, val_class_loss, val_geometry_loss, val_corner_error, val_accuracy = validate_epoch(
            model, val_loader, device, epoch)

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_corner_errors.append(train_corner_loss)  # 使用损失作为替代指标
        val_corner_errors.append(val_corner_error)  # 注意这里使用val_corner_error
        train_class_losses.append(train_class_loss)
        val_class_losses.append(val_class_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_geometry_losses.append(train_geometry_loss)
        val_geometry_losses.append(val_geometry_loss)

        # 写入训练日志文件
        with open(log_file, 'a') as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_corner_loss:.6f},{val_corner_error:.6f},{train_class_loss:.6f},{val_class_loss:.6f},{train_accuracy:.6f},{val_accuracy:.6f}\n")

        # 绘制训练指标图
        plot_training_metrics(
            args.checkpoints_dir,
            train_losses, val_losses,
            train_corner_errors, val_corner_errors,
            train_class_losses, val_class_losses,
            train_accuracies, val_accuracies,
            train_geometry_losses, val_geometry_losses,
            epoch
        )

        # 检查是否为最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_corner_errors': train_corner_errors,
                'val_corner_errors': val_corner_errors,
                'train_class_losses': train_class_losses,
                'val_class_losses': val_class_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'train_geometry_losses': train_geometry_losses,
                'val_geometry_losses': val_geometry_losses
            }, os.path.join(args.checkpoints_dir, f'best_model_{args.backbone}.pth'))
            print(
                f"Saved best model, validation loss: {best_val_loss:.4f}, corner error: {val_corner_loss:.4f}, class loss: {val_class_loss:.4f}, accuracy: {val_accuracy:.4f}")

        # 定期保存模型检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss if val_loader else train_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_corner_errors': train_corner_errors,
                'val_corner_errors': val_corner_errors,
                'train_class_losses': train_class_losses,
                'val_class_losses': val_class_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'train_geometry_losses': train_geometry_losses,
                'val_geometry_losses': val_geometry_losses
            }, os.path.join(args.checkpoints_dir, f'{args.backbone}_model_epoch_{epoch + 1}.pth'))
            print(f"Saved checkpoint for epoch {epoch + 1}")

        # 可视化样本
        if (epoch + 1) % args.visualize_every == 0 or epoch == 0:
            # 可视化样本
            dataset_to_visualize = val_dataset if val_dataset else train_dataset
            loader_to_visualize = DataLoader(dataset_to_visualize, batch_size=1, shuffle=True)

            visualize_samples(
                model,
                loader_to_visualize,
                device,
                epoch,
                vis_dir,
                num_samples=args.visualize_samples,
                debug_mode=args.debug_mode
            )
            print(f"Saved visualization results for epoch {epoch + 1}")

    # 训练完成后保存最终模型
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss if val_loader else train_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_corner_errors': train_corner_errors,
        'val_corner_errors': val_corner_errors,
        'train_class_losses': train_class_losses,
        'val_class_losses': val_class_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_geometry_losses': train_geometry_losses,
        'val_geometry_losses': val_geometry_losses
    }, os.path.join(args.checkpoints_dir, f'final_{args.backbone}_model.pth'))
    print("Training completed, final model saved")

    # 最终绘制训练指标图
    plot_training_metrics(
        args.checkpoints_dir,
        train_losses, val_losses,
        train_corner_errors, val_corner_errors,
        train_class_losses, val_class_losses,
        train_accuracies, val_accuracies,
        train_geometry_losses, val_geometry_losses,
        args.num_epochs - 1,
        final=True
    )
    print("Saved final training metrics")


if __name__ == "__main__":
    main() 