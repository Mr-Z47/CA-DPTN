import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from perspective_corrector import PerspectiveCorrectionNetwork, draw_region_box

def parse_args():
    parser = argparse.ArgumentParser(description='测试透视矫正网络')
    parser.add_argument('--image_path', type=str, required=True, help='测试图像路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--backbone', type=str, default='vgg16', choices=['vgg16', 'resnet50'], help='特征提取器骨干网络')
    parser.add_argument('--output_dir', type=str, default='test_results', help='输出目录')
    parser.add_argument('--corners', type=float, nargs=8, default=[0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8], 
                        help='目标角点坐标 [左上x,左上y, 右上x,右上y, 右下x,右下y, 左下x,左下y]')
    parser.add_argument('--debug', action='store_true', help='显示详细调试信息')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = PerspectiveCorrectionNetwork(backbone=args.backbone).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"已加载模型: {args.model_path}")
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载图像
    image = Image.open(args.image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # 添加batch维度
    
    # 准备角点
    corners = torch.tensor(args.corners, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 进行推理
    with torch.no_grad():
        transformed_image, predicted_corners, distortion_probs = model(image_tensor, corners)
    
    # 反归一化图像
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    input_image = image_tensor[0] * std + mean
    input_image = torch.clamp(input_image, 0, 1)
    transformed_image = transformed_image[0] * std + mean
    transformed_image = torch.clamp(transformed_image, 0, 1)
    
    # 在原图上绘制目标区域和预测区域
    # 获取分类结果
    distortion_types = ["正常视角", "俯视角", "仰视角", "倾斜视角"]
    probs = distortion_probs[0].cpu().numpy()
    max_prob_idx = np.argmax(probs)
    predicted_type = distortion_types[max_prob_idx]
    
    # 在原始图像上绘制预测的四边形区域和分类结果
    original_with_overlay = draw_region_box(
        input_image, 
        predicted_corners[0], 
        message=f"预测区域", 
        debug_info=args.debug,
        distortion_probs=distortion_probs[0]
    )
    
    # 转换为PIL图像
    original_pil = transforms.ToPILImage()(original_with_overlay.cpu())
    result_pil = transforms.ToPILImage()(transformed_image.cpu())
    
    # 创建可视化结果
    plt.figure(figsize=(12, 6))
    
    # 显示原始图像和预测区域
    plt.subplot(1, 2, 1)
    plt.imshow(original_pil)
    plt.title(f'原始图像与预测区域 ({predicted_type})')
    plt.axis('off')
    
    # 显示矫正后的图像
    plt.subplot(1, 2, 2)
    plt.imshow(result_pil)
    plt.title('透视矫正后图像')
    plt.axis('off')
    
    # 保存结果
    plt.tight_layout()
    output_path = os.path.join(args.output_dir, f'test_result_{os.path.basename(args.image_path)}')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"测试结果已保存到: {output_path}")
    
    # 保存变换后的图像
    result_pil.save(os.path.join(args.output_dir, f'transformed_{os.path.basename(args.image_path)}'))
    
    # 打印预测的角点坐标
    print("\n预测的角点坐标:")
    predicted_corners = predicted_corners[0].cpu().numpy()
    print(f"左上: ({predicted_corners[0]:.4f}, {predicted_corners[1]:.4f})")
    print(f"右上: ({predicted_corners[2]:.4f}, {predicted_corners[3]:.4f})")
    print(f"右下: ({predicted_corners[4]:.4f}, {predicted_corners[5]:.4f})")
    print(f"左下: ({predicted_corners[6]:.4f}, {predicted_corners[7]:.4f})")
    
    # 打印预测的透视类型和概率
    print("\n预测的透视类型:")
    for i, (dtype, prob) in enumerate(zip(distortion_types, probs)):
        print(f"{dtype}: {prob:.4f}" + (" ←" if i == max_prob_idx else ""))

if __name__ == "__main__":
    main() 