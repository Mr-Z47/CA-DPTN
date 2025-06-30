import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2
from pathlib import Path
import sys



# 添加父目录到路径，以便导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入透视矫正网络
from perspective_corrector import PerspectiveCorrectionNetwork, draw_region_box

def parse_args():
    parser = argparse.ArgumentParser(description="使用透视矫正网络进行推理")
    parser.add_argument('--input', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='results', help='输出目录路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='特征提取器骨干网络')
    parser.add_argument('--no_visualize', action='store_true', help='禁用可视化结果')
    parser.add_argument('--max_size', type=int, default=1280, help='处理前的最大图像尺寸')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    return parser.parse_args()

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

def load_and_preprocess_image(img_path, max_size):
    """加载并预处理图像
    
    Args:
        img_path: 图像路径
        max_size: 最大尺寸
        
    Returns:
        tuple: (原始图像, 预处理后的张量)
    """
    try:
        # 加载图像
        img = Image.open(img_path).convert('RGB')
        
        # 保存原始尺寸
        orig_width, orig_height = img.size
        
        # 定义变换
        transform = transforms.Compose([
            LimitMaxSize(max_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 应用变换
        img_tensor = transform(img)
        
        return img, img_tensor
    except Exception as e:
        print(f"无法加载图像 {img_path}: {e}")
        return None, None

def denormalize_tensor(tensor, device):
    """反归一化张量
    
    Args:
        tensor: 输入张量
        device: 计算设备
        
    Returns:
        反归一化后的张量
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1)

def save_tensor_as_image(tensor, save_path):
    """将张量保存为图像
    
    Args:
        tensor: 输入张量，形状为 [C, H, W]
        save_path: 保存路径
    """
    # 确保是CPU张量
    tensor = tensor.cpu().detach()
    
    # 转换为PIL图像
    img = transforms.ToPILImage()(tensor)
    
    # 保存图像
    img.save(save_path)

def process_image(model, image_path, output_dir, args, device):
    """处理单个图像"""
    try:
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # 调整图像大小，保持宽高比
        if max(image.size) > args.max_size:
            ratio = args.max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
        
        # 定义图像变换
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 转换图像为张量
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 进行推理
        with torch.no_grad():
            transformed_image, predicted_corners, distortion_probs = model(image_tensor)
        
        # 反归一化图像
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        input_image = image_tensor[0] * std + mean
        input_image = torch.clamp(input_image, 0, 1)
        transformed_image = transformed_image[0] * std + mean
        transformed_image = torch.clamp(transformed_image, 0, 1)
        
        # 获取分类结果
        distortion_types = ["正常视角", "俯视角", "仰视角", "倾斜视角"]
        probs = distortion_probs[0].cpu().numpy()
        max_prob_idx = np.argmax(probs)
        predicted_type = distortion_types[max_prob_idx]
        
        # 获取文件名（不含扩展名）用于保存结果
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 保存矫正后的图像
        result_pil = transforms.ToPILImage()(transformed_image.cpu())
        
        # 如果原始图像较大，还原到原始大小
        if original_size != image.size:
            result_pil = result_pil.resize(original_size, Image.LANCZOS)
        
        result_pil.save(os.path.join(output_dir, f'corrected_{base_name}.png'))
        
        # 如果需要可视化，创建并保存可视化结果
        if not args.no_visualize:
            # 在原始图像上绘制预测的四边形区域和分类结果
            overlay_image = draw_region_box(
                input_image, 
                predicted_corners[0], 
                message=f"预测区域", 
                debug_info=args.debug,
                distortion_probs=distortion_probs[0]
            )
            
            overlay_pil = transforms.ToPILImage()(overlay_image.cpu())
            
            # 创建可视化结果
            plt.figure(figsize=(12, 6))
            
            # 显示原始图像和预测区域
            plt.subplot(1, 2, 1)
            plt.imshow(overlay_pil)
            plt.title(f'预测区域：{predicted_type}')
            plt.axis('off')
            
            # 显示矫正后的图像
            plt.subplot(1, 2, 2)
            plt.imshow(result_pil)
            plt.title('透视矫正后图像')
            plt.axis('off')
            
            # 保存可视化结果
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'visualization_{base_name}.png'), dpi=150)
            plt.close()
        
        return True, predicted_type, probs
    
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return False, None, None

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = PerspectiveCorrectionNetwork(backbone=args.backbone).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"已加载模型: {args.model_path}")
    
    # 处理输入（单个图像或目录）
    if os.path.isfile(args.input):
        # 处理单个图像
        print(f"处理图像: {args.input}")
        success, predicted_type, _ = process_image(model, args.input, args.output, args, device)
        if success:
            print(f"处理完成，预测视角类型: {predicted_type}")
            print(f"结果已保存到: {args.output}")
        else:
            print("处理失败")
    
    elif os.path.isdir(args.input):
        # 处理目录中的所有图像
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(
                [os.path.join(args.input, f) for f in os.listdir(args.input) 
                 if f.lower().endswith(ext)]
            )
        
        if not image_files:
            print(f"警告: 目录 {args.input} 中没有找到图像文件")
            return
        
        # 创建分类统计字典
        classification_stats = {
            "正常视角": 0,
            "俯视角": 0,
            "仰视角": 0,
            "倾斜视角": 0
        }
        
        # 使用tqdm显示进度
        print(f"开始处理 {len(image_files)} 个图像...")
        successful = 0
        
        for image_path in tqdm(image_files, desc="处理图像"):
            success, predicted_type, _ = process_image(model, image_path, args.output, args, device)
            if success:
                successful += 1
                if predicted_type:
                    classification_stats[predicted_type] += 1
        
        # 打印处理结果
        print(f"处理完成: {successful}/{len(image_files)} 个图像成功")
        print(f"结果已保存到: {args.output}")
        
        # 打印分类统计
        print("\n视角类型分布:")
        for view_type, count in classification_stats.items():
            percentage = (count / successful * 100) if successful > 0 else 0
            print(f"{view_type}: {count} ({percentage:.1f}%)")
        
        # 生成分类统计图表
        if successful > 0:
            plt.figure(figsize=(10, 6))
            plt.bar(classification_stats.keys(), classification_stats.values())
            plt.title('视角类型分布')
            plt.xlabel('视角类型')
            plt.ylabel('图像数量')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output, 'classification_stats.png'), dpi=150)
            plt.close()
    
    else:
        print(f"错误: 输入路径 {args.input} 不存在")

if __name__ == "__main__":
    main() 