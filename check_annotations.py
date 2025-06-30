import os
import sys
import cv2
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description='检查标注文件中的角点')
    parser.add_argument('--image_dir', type=str, required=True, help='图像目录')
    parser.add_argument('--annotation_file', type=str, required=True, help='标注文件路径')
    parser.add_argument('--output_dir', type=str, default='annotation_check', help='输出目录')
    parser.add_argument('--max_images', type=int, default=10, help='最大检查图像数量')
    return parser.parse_args()

def draw_annotation(img, corners, title="", show_points=True):
    """在图像上绘制标注角点"""
    # 复制图像以便在上面绘制
    img_draw = img.copy()
    height, width = img_draw.shape[:2]
    
    # 将标注坐标转换为像素坐标
    points = []
    corner_array = np.array(corners).reshape(4, 2)
    
    # 检查坐标是否为相对值(0-1)或绝对像素值
    if np.max(corner_array) <= 1.0:
        # 将相对坐标转换为像素坐标
        for i in range(4):
            x = int(corner_array[i, 0] * width)
            y = int(corner_array[i, 1] * height)
            points.append((x, y))
    else:
        # 使用绝对像素坐标
        for i in range(4):
            x = int(corner_array[i, 0])
            y = int(corner_array[i, 1])
            points.append((x, y))
    
    # 绘制四边形边框
    for i in range(4):
        pt1 = points[i]
        pt2 = points[(i + 1) % 4]
        cv2.line(img_draw, pt1, pt2, (0, 255, 0), 2)
    
    # 用不同颜色标记四个角点
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (128, 0, 128)]  # 蓝、红、绿、紫
    labels = ["1:左上", "2:右上", "3:右下", "4:左下"]
    
    if show_points:
        for i, point in enumerate(points):
            cv2.circle(img_draw, point, 5, colors[i], -1)
            cv2.putText(img_draw, labels[i], (point[0]+10, point[1]+10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
    
    # 添加标题
    if title:
        cv2.putText(img_draw, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img_draw

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载标注文件
    try:
        annotations = pd.read_csv(args.annotation_file)
        print(f"成功加载标注文件: {args.annotation_file}")
        print(f"标注文件列名: {list(annotations.columns)}")
        print(f"标注文件前5行: \n{annotations.head()}")
    except Exception as e:
        print(f"加载标注文件出错: {e}")
        return
    
    # 确定文件名列
    filename_column = 'filename'
    if filename_column not in annotations.columns:
        # 尝试常见的文件名列
        possible_columns = ['filename', 'file', 'image', 'name', 'img', 'image_path']
        for col in possible_columns:
            if col in annotations.columns:
                filename_column = col
                break
        
        # 如果还是找不到，使用第一列
        if filename_column not in annotations.columns:
            filename_column = annotations.columns[0]
    
    print(f"使用 '{filename_column}' 作为文件名列")
    
    # 获取图像文件列表
    img_files = []
    for root, _, files in os.walk(args.image_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_files.append(os.path.join(root, file))
    
    print(f"找到 {len(img_files)} 个图像文件")
    
    # 限制检查的图像数量
    img_files = img_files[:args.max_images]
    
    # 检查每个图像
    processed_count = 0
    for img_path in img_files:
        img_name = os.path.basename(img_path)
        print(f"\n处理图像: {img_name}")
        
        # 尝试查找标注
        annotation = annotations[annotations[filename_column] == img_name]
        
        # 如果找不到，尝试相对路径
        if annotation.empty:
            rel_path = os.path.relpath(img_path, args.image_dir).replace('\\', '/')
            annotation = annotations[annotations[filename_column] == rel_path]
            print(f"通过相对路径查找: {rel_path}")
        
        if annotation.empty:
            print(f"未找到图像 {img_name} 的标注，跳过")
            continue
        
        # 加载图像
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法加载图像 {img_path}")
                continue
                
            height, width = img.shape[:2]
            print(f"图像尺寸: {width}x{height}")
        except Exception as e:
            print(f"处理图像出错: {e}")
            continue
        
        # 提取角点坐标
        corners = None
        
        # 检查是否有命名的坐标列
        required_columns = ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        if all(col in annotations.columns for col in required_columns):
            print("标注文件使用命名列")
            corners = []
            for i in range(1, 5):
                x_col = f'x{i}'
                y_col = f'y{i}'
                corners.extend([
                    float(annotation[x_col].values[0]),
                    float(annotation[y_col].values[0])
                ])
        else:
            # 尝试使用索引获取坐标
            print("标注文件使用默认列顺序")
            corners = []
            for i in range(2, min(10, len(annotation.columns))):
                try:
                    corners.append(float(annotation.iloc[0, i]))
                except:
                    print(f"无法转换坐标值: {annotation.iloc[0, i]}")
            
            if len(corners) != 8:
                print(f"无效的角点数量: {len(corners)}，应为8")
                continue
        
        print(f"读取的坐标: {corners}")
        
        # 检查坐标值范围
        print(f"坐标范围: min={min(corners)}, max={max(corners)}")
        
        # 绘制原始坐标和归一化坐标进行比较
        img_with_orig = img.copy()
        img_with_norm = img.copy()
        
        # 绘制原始坐标
        try:
            img_with_orig = draw_annotation(img, corners, "原始坐标")
        except Exception as e:
            print(f"绘制原始坐标出错: {e}")
        
        # 归一化坐标并绘制
        if max(corners) > 1.0:
            norm_corners = []
            for i in range(0, len(corners), 2):
                norm_corners.append(corners[i] / width)
                norm_corners.append(corners[i+1] / height)
            
            try:
                img_with_norm = draw_annotation(img, norm_corners, "归一化坐标")
            except Exception as e:
                print(f"绘制归一化坐标出错: {e}")
        else:
            img_with_norm = draw_annotation(img, corners, "已归一化坐标")
        
        # 将两个图像水平拼接
        combined = np.hstack((img_with_orig, img_with_norm))
        
        # 保存结果
        output_path = os.path.join(args.output_dir, f"check_{img_name}")
        cv2.imwrite(output_path, combined)
        print(f"保存结果到: {output_path}")
        
        processed_count += 1
    
    print(f"\n处理完成，共检查了 {processed_count} 个图像")

if __name__ == "__main__":
    main() 